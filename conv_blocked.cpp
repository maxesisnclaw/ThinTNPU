#include <torch/all.h>
#include <torch/library.h>
#include <acl/acl_rt.h>
#include "kernel_operator.h"
using namespace AscendC;
namespace thin { aclrtStream stream(); void sync(); void select(c10::Device); }
namespace blocked_conv {
__aicore__ inline int imin(int a, int b) { return a < b ? a : b; }

#include "layout_blocked.h"

template<int CI>
__aicore__ inline void stage(LocalTensor<half> image, GlobalTensor<half> input,
                              int cb, int bank, int ci, int maxP, int n,
                              int pos, int count, int plane) {
    int channels = imin(CI, ci - cb);
    DataCopy(image[bank * CI * maxP], input[cb * n + pos * 16],
             DataCopyParams{(uint16_t)(channels / 16), (uint16_t)count,
                            (uint16_t)(n - count), (uint16_t)(plane - count)});
    SetFlag<HardEvent::MTE2_MTE1>(bank);
}

template<int TN, bool HalfOut, bool Fused>
__aicore__ inline void finish(LocalTensor<float> value, LocalTensor<float> scratch,
                               LocalTensor<half> halves, LocalTensor<float> bias,
                               LocalTensor<float> accum, GlobalTensor<half> outH,
                               GlobalTensor<float> outF, int oc, int n, int pos, int count,
                               bool hasBias) {
    for (int group = 0; group < TN; group += 16) {
        DataCopyEnhancedParams ep; ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(value, accum[group * count], DataCopyParams{1, (uint16_t)(count / 16), 0, 0}, ep);
        PipeBarrier<PIPE_V>();
        if (hasBias) {
            for (int off = 0; off < count; off += 128) {
                int repeats = imin(128, count - off);
                Add(value[off * 16], value[off * 16], bias[group],
                    16, repeats, {1, 1, 1, 2, 2, 0});
            }
            PipeBarrier<PIPE_V>();
        }
        if constexpr (Fused) {
            Muls(scratch, value, -1.0f, count * 16); PipeBarrier<PIPE_V>();
            Exp(scratch, scratch, count * 16); PipeBarrier<PIPE_V>();
            Adds(scratch, scratch, 1.0f, count * 16); PipeBarrier<PIPE_V>();
            Div(value, value, scratch, count * 16); PipeBarrier<PIPE_V>();
        }
        if constexpr (HalfOut) {
            Cast(halves, value, RoundMode::CAST_NONE, count * 16);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(2); WaitFlag<HardEvent::V_MTE3>(2);
            DataCopy(outH[(oc + group) * n + pos * 16], halves, count * 16);
        } else {
            SetFlag<HardEvent::V_MTE3>(2); WaitFlag<HardEvent::V_MTE3>(2);
            DataCopy(outF[(oc + group) * n + pos * 16], value, count * 16);
        }
        SetFlag<HardEvent::MTE3_V>(2); WaitFlag<HardEvent::MTE3_V>(2);
    }
}

template<int TM, int TN, int CI, int TK, int KERNEL, bool HalfOut, bool Fused>
__global__ __aicore__ void convolution(GM_ADDR x, GM_ADDR pw, GM_ADDR bp, GM_ADDR y,
                                      int ci, int co, int h, int width, int stride,
                                      int oh, int ow, int maxP, bool hasBias) {
    constexpr int banks = TK > 128 ? 1 : 2;
    static_assert(banks * TM * TK * 2 <= 65536, "double-buffered L0A exceeds 310P capacity");
    TPipe pipe;
    TBuf<TPosition::A1> imageBuf;
    TBuf<TPosition::B1> weightBuf;
    TBuf<TPosition::A2> aBuf;
    TBuf<TPosition::B2> bBuf;
    TBuf<TPosition::CO1> cBuf;
    TBuf<TPosition::VECCALC> valueBuf, scratchBuf, halfBuf, biasBuf;
    pipe.InitBuffer(imageBuf, (ci > CI ? 2 : 1) * CI * maxP * 2);
    pipe.InitBuffer(weightBuf, TN * ci * KERNEL * KERNEL * 2);
    pipe.InitBuffer(aBuf, banks * TM * TK * 2);
    pipe.InitBuffer(bBuf, banks * TN * TK * 2);
    pipe.InitBuffer(cBuf, TN * TM * 4);
    pipe.InitBuffer(valueBuf, 16 * TM * 4);
    pipe.InitBuffer(scratchBuf, 16 * TM * 4);
    pipe.InitBuffer(halfBuf, 16 * TM * 2);
    pipe.InitBuffer(biasBuf, TN * 6);
    GlobalTensor<half> input, weights, biases, outH;
    GlobalTensor<float> outF;
    input.SetGlobalBuffer((__gm__ half*)x);
    weights.SetGlobalBuffer((__gm__ half*)pw);
    biases.SetGlobalBuffer((__gm__ half*)bp);
    outH.SetGlobalBuffer((__gm__ half*)y);
    outF.SetGlobalBuffer((__gm__ float*)y);
    auto image = imageBuf.Get<half>(), weight = weightBuf.Get<half>();
    auto value = valueBuf.Get<float>(), scratch = scratchBuf.Get<float>();
    int n = oh * ow, kk = KERNEL * KERNEL;
    int groups = co / TN, channelCores = 1;
    while (channelCores * 2 <= groups && channelCores * 2 <= GetBlockNum()) channelCores *= 2;
    int spatialCores = GetBlockNum() / channelCores;
    int tiles = (n + TM - 1) / TM;
    for (int oc = GetBlockIdx() % channelCores * TN; oc < co; oc += channelCores * TN) {
        DataCopy(weight, weights[oc * 16],
                 DataCopyParams{(uint16_t)(ci / 16 * kk), (uint16_t)TN,
                                (uint16_t)((co + 63) / 64 * 64 - TN), 0});
        SetFlag<HardEvent::MTE2_MTE1>(2); WaitFlag<HardEvent::MTE2_MTE1>(2);
        if (hasBias) {
            DataCopy(biasBuf.Get<half>()[TN * 2], biases[oc], TN);
            SetFlag<HardEvent::MTE2_V>(2); WaitFlag<HardEvent::MTE2_V>(2);
            Cast(biasBuf.Get<float>(), biasBuf.Get<half>()[TN * 2], RoundMode::CAST_NONE, TN);
            PipeBarrier<PIPE_V>();
        }
        for (int tile = GetBlockIdx() / channelCores; tile < tiles; tile += spatialCores) {
            int pos = tile * TM, count = imin(TM, n - pos);
            int begin = 0, end = 1, length = TM;
            if constexpr (KERNEL == 3) {
                int row = pos / ow, lastRow = (pos + count - 1) / ow;
                int group = stride;
                begin = (row * stride - 1 > 0 ? row * stride - 1 : 0) / group * group;
                end = imin(h, (lastRow * stride + 2 + group - 1) / group * group);
                length = (end - begin) * width;
            }
            int sourcePos = KERNEL == 1 ? pos : begin * width;
            int copied = KERNEL == 1 ? count : length;
            int plane = KERNEL == 1 ? TM : length;
            stage<CI>(image, input, 0, 0, ci, maxP, h * width, sourcePos, copied, plane);
            for (int bank = 0; bank < banks; ++bank) SetFlag<HardEvent::M_MTE1>(bank);
            for (int cb = 0; cb < ci; cb += CI) {
                int channels = imin(CI, ci - cb), imageBank = cb / CI % 2;
                WaitFlag<HardEvent::MTE2_MTE1>(imageBank);
                for (int k = 0; k < channels * kk; k += TK) {
                    int bank = (cb / CI * (CI * kk / TK) + k / TK) % banks;
                    auto a = aBuf.Get<half>()[bank * TM * TK];
                    auto b = bBuf.Get<half>()[bank * TN * TK];
                    WaitFlag<HardEvent::M_MTE1>(bank);
                    LoadData3DParamsV2<half> p;
                    p.l1H = KERNEL == 1 ? 1 : end - begin;
                    p.l1W = KERNEL == 1 ? TM : width;
                    p.channelSize = channels; p.kExtension = TK; p.kStartPt = k; p.mExtension = count;
                    p.mStartPt = KERNEL == 1 ? 0 : (pos / ow - begin / stride) * ow + pos % ow;
                    p.filterW = KERNEL; p.filterH = KERNEL; p.strideW = stride; p.strideH = stride;
                    if constexpr (KERNEL == 3)
                        for (int i = 0; i < 4; ++i) p.padList[i] = 1;
                    LoadData(a, image[imageBank * CI * maxP], p);
                    LoadData2DParams ld; ld.repeatTimes = TK / 16 * TN / 16; ld.srcStride = 1;
                    LoadData(b, weight[(cb * kk + k) * TN], ld);
                    SetFlag<HardEvent::MTE1_M>(bank); WaitFlag<HardEvent::MTE1_M>(bank);
                    MmadParams mm; mm.m = count; mm.n = TN; mm.k = TK;
                    mm.isBias = cb != 0 || k != 0;
                    Mmad(cBuf.Get<float>(), a, b, mm);
                    SetFlag<HardEvent::M_MTE1>(bank);
                    if (k == 0 && cb + CI < ci) {
                        SetFlag<HardEvent::MTE1_MTE2>(2); WaitFlag<HardEvent::MTE1_MTE2>(2);
                        stage<CI>(image, input, cb + CI, imageBank ^ 1, ci, maxP, h * width, sourcePos, copied, plane);
                    }
                }
                SetFlag<HardEvent::MTE1_MTE2>(2); WaitFlag<HardEvent::MTE1_MTE2>(2);
            }
            for (int bank = 0; bank < banks; ++bank) WaitFlag<HardEvent::M_MTE1>(bank);
            SetFlag<HardEvent::M_V>(2); WaitFlag<HardEvent::M_V>(2);
            finish<TN, HalfOut, Fused>(value, scratch, halfBuf.Get<half>(), biasBuf.Get<float>(),
                cBuf.Get<float>(), outH, outF, oc, n, pos, count, hasBias);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

at::Tensor conv(const at::Tensor& x, const at::Tensor& packed, const std::optional<at::Tensor>& bias,
                int64_t co, int64_t kernel, int64_t stride, bool fp32, bool fused) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && x.dim() == 5 &&
                x.size(0) == 1 && x.size(4) == 16 && x.scalar_type() == at::kHalf &&
                x.is_contiguous() && x.storage_offset() % 16 == 0 && !x.requires_grad(),
                "expected aligned contiguous N=1 FP16 NC1HWC0 input, C0=16");
    int ci = x.size(1) * 16, h = x.size(2), w = x.size(3);
    bool point = kernel == 1 && stride == 1 && h == 40 && w == 40 &&
                 co == 256 && (ci == 256 || ci == 512);
    bool spatial = kernel == 3 && stride == 1 && h == 40 && w == 40 && ci == 128 && co == 128;
    TORCH_CHECK(point || spatial, "expected a model.6 C2f convolution shape");
    TORCH_CHECK(!fp32 && fused, "blocked C2f requires FP16 output and fused FP32 SiLU");
    TORCH_CHECK(packed.device() == x.device() && packed.scalar_type() == at::kHalf &&
                packed.is_contiguous() && packed.storage_offset() == 0 && !packed.requires_grad() &&
                packed.sizes() == c10::IntArrayRef({ci / 16, kernel * kernel, (co + 63) / 64 * 64, 16}),
                "expected prepared FRACTAL_Z weights matching the convolution");
    if (bias)
        TORCH_CHECK(bias->device() == x.device() && bias->scalar_type() == at::kHalf &&
                    bias->is_contiguous() && bias->storage_offset() == 0 && !bias->requires_grad() &&
                    bias->dim() == 1 && bias->numel() == co, "expected same-device FP16 bias");
    thin::select(x.device());
    int oh = (h - 1) / stride + 1, ow = (w - 1) / stride + 1;
    auto y = at::empty({1, co / 16, oh, ow, 16}, x.options().dtype(fp32 ? at::kFloat : at::kHalf));
    GM_ADDR bp = bias ? (GM_ADDR)bias->data_ptr() : nullptr;
    auto s = thin::stream();
#define RUN(M, N, C, K, F, H, S) convolution<M, N, C, K, F, H, S><<<8, nullptr, s>>>( \
    (GM_ADDR)x.data_ptr(), (GM_ADDR)packed.data_ptr(), bp, (GM_ADDR)y.data_ptr(), ci, co, h, w, stride, oh, ow, maxP, bias.has_value())
#define OUTPUT(M, N, C, K, F) RUN(M,N,C,K,F,true,true)
    if (point) { int maxP = 400; OUTPUT(400,64,128,32,1); }
    else { int maxP = 560; OUTPUT(400,64,32,16,3); }
#undef OUTPUT
#undef RUN
    thin::sync();
    return y;
}
}
TORCH_LIBRARY(thin_blocked, m) {
    m.def("pack(Tensor x) -> Tensor");
    m.def("unpack(Tensor x) -> Tensor");
    m.def("conv(Tensor x, Tensor packed, Tensor? bias, int co, int kernel, int stride, bool fp32_output=False, bool fused=False) -> Tensor");
}
TORCH_LIBRARY_IMPL(thin_blocked, PrivateUse1, m) {
    m.impl("conv", TORCH_FN(blocked_conv::conv));
    m.impl("pack", TORCH_FN(blocked_conv::pack));
    m.impl("unpack", TORCH_FN(blocked_conv::unpack));
}
TORCH_LIBRARY_IMPL(thin_blocked, AutogradPrivateUse1, m) {
    m.impl("conv", torch::CppFunction::makeFallthrough());
    m.impl("pack", torch::CppFunction::makeFallthrough());
    m.impl("unpack", torch::CppFunction::makeFallthrough());
}
