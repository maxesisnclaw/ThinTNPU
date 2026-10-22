#include <torch/all.h>
#include <limits>
#include <acl/acl_rt.h>
#include "kernel_operator.h"
#include "cube_layout.h"
using namespace AscendC;
namespace thin {
aclrtStream stream();
void sync();
void select(c10::Device);
}
namespace image_c4 {
constexpr int N = 32, K = 48;
constexpr int overhead = 32 + N * K * 8 + (K + 32) * 2 + N * 18;
constexpr int ubM = (262144 - overhead) / (3 * N * 4) / 16 * 16;
constexpr int l0M = 65536 / (K * 2) / 16 * 16;
constexpr int M = ubM < l0M ? ubM : l0M;
__aicore__ inline int imin(int a, int b) { return a < b ? a : b; }
__aicore__ inline int imax(int a, int b) { return a > b ? a : b; }

template <bool HalfOut, bool Fused, bool Blocked = false>
__global__ __aicore__ void stem(GM_ADDR x, GM_ADDR w, GM_ADDR bias, GM_ADDR y, GM_ADDR index, int h, int width, int co,
                                bool hasBias) {
    const int ow = width / 2;
    const int rowsPerTile = ow <= M ? M / ow : 1;
    const int stripWidth = imin(width, 2 * M + 16);
    const int P = imin(h, 2 * rowsPerTile + 1) * stripWidth;
    TPipe pipe;
    TBuf<TPosition::VECCALC> arenaBuf, maskBuf, weightIndexBuf, weightRawBuf, weightPackedBuf, scratchBuf,
        biasBuf, broadcastBuf;
    TBuf<TPosition::A1> imageBuf;
    TBuf<TPosition::B1> weightBuf;
    TBuf<TPosition::A2> aBuf;
    TBuf<TPosition::B2> bBuf;
    TBuf<TPosition::CO1> cBuf;
    pipe.InitBuffer(arenaBuf, (3 * N * M * 4 > 24 * P * 2 ? 3 * N * M * 4 : 24 * P * 2));
    pipe.InitBuffer(maskBuf, 32);
    pipe.InitBuffer(weightIndexBuf, N * K * 4);
    pipe.InitBuffer(weightRawBuf, N * K * 2);
    pipe.InitBuffer(weightPackedBuf, N * K * 2);
    pipe.InitBuffer(scratchBuf, (K + 32) * 2);
    pipe.InitBuffer(biasBuf, N * 2);
    if constexpr (Blocked) pipe.InitBuffer(broadcastBuf, N * 16);
    pipe.InitBuffer(imageBuf, 4 * P * 2);
    pipe.InitBuffer(weightBuf, N * K * 2);
    pipe.InitBuffer(aBuf, M * K * 2);
    pipe.InitBuffer(bBuf, N * K * 2);
    pipe.InitBuffer(cBuf, N * M * 4);
    GlobalTensor<half> input, weight, biases, outH;
    GlobalTensor<float> outF;
    GlobalTensor<uint32_t> indices;
    input.SetGlobalBuffer((__gm__ half *)x);
    weight.SetGlobalBuffer((__gm__ half *)w);
    biases.SetGlobalBuffer((__gm__ half *)bias);
    outH.SetGlobalBuffer((__gm__ half *)y);
    outF.SetGlobalBuffer((__gm__ float *)y);
    indices.SetGlobalBuffer((__gm__ uint32_t *)index);
    auto raw = arenaBuf.Get<half>(), packed16 = raw[4 * P], packed = raw[20 * P];
    auto cUb = arenaBuf.Get<float>();
    auto halfOut = arenaBuf.Get<half>();
    auto wr = weightRawBuf.Get<half>(), wp = weightPackedBuf.Get<half>(), scratch = scratchBuf.Get<half>();
    auto wi = weightIndexBuf.Get<uint32_t>();
    auto mask = maskBuf.Get<uint16_t>();
    DataCopy(wi, indices, N * K);
    Duplicate(mask, (uint16_t)0x000f, 16);
    PipeBarrier<PIPE_ALL>();
    int oh = (h + 1) / 2, n = oh * ow, rowTiles = (ow + M - 1) / M, spatialTiles = ((oh + rowsPerTile - 1) / rowsPerTile) * rowTiles, channelTiles = (co + N - 1) / N,
        cachedOc = -1;
    for (int tile = GetBlockIdx(); tile < spatialTiles * channelTiles; tile += GetBlockNum()) {
        int row = (tile % spatialTiles) / rowTiles * rowsPerTile;
        int col = (tile % rowTiles) * M;
        int tileCols = imin(M, ow - col);
        int tileM = imin(rowsPerTile, oh - row) * tileCols;
        int pos = row * ow + col, oc = (tile / spatialTiles) * N;
        int realM = tileM, realN = imin(N, co - oc);
        auto out = cUb[N * tileM], tmp = cUb[2 * N * tileM];
        if (cachedOc != oc) {
            Duplicate(wr, (half)0, N * K);
            PipeBarrier<PIPE_ALL>();
            for (int no = 0; no < realN; no++) {
                int off = (oc + no) * 27, align = off / 16 * 16, skew = off - align;
                DataCopy(scratch, weight[align], (skew + 27 + 15) / 16 * 16);
                PipeBarrier<PIPE_ALL>();
                Adds(wr[no * K], scratch[skew], (half)0, 27);
                PipeBarrier<PIPE_ALL>();
            }
            Gather(wp, wr, wi, 0, N * K);
            PipeBarrier<PIPE_ALL>();
            DataCopy(weightBuf.Get<half>(), wp, N * K);
            if (hasBias) {
                DataCopy(biasBuf.Get<half>(), biases[oc], (realN + 15) / 16 * 16);
                PipeBarrier<PIPE_ALL>();
                Cast(scratchBuf.Get<float>(), biasBuf.Get<half>(), RoundMode::CAST_NONE, N);
                if constexpr (Blocked) {
                    PipeBarrier<PIPE_ALL>();
                    for (int group = 0; group < N; group += 16)
                        for (int repeat = 0; repeat < 4; ++repeat)
                            DataCopy(broadcastBuf.Get<float>()[group * 4 + repeat * 16], scratchBuf.Get<float>()[group], 16);
                }
            }
            PipeBarrier<PIPE_ALL>();
            cachedOc = oc;
        }
        int lastRow = row + imin(rowsPerTile, oh - row) - 1;
        int begin = imax(0, row * 2 - 1), end = imin(h, lastRow * 2 + 2);
        int colBegin = imax(0, col * 2 - 1) / 16 * 16;
        int colEnd = imin(width, (2 * (col + tileCols - 1) + 2 + 15) / 16 * 16);
        int localWidth = colEnd - colBegin;
        int length = (end - begin) * localWidth;
        Duplicate(raw[3 * P], (half)0, P);
        PipeBarrier<PIPE_ALL>();
        if (localWidth == width && (h * width - length) / 16 <= 65535) {
            DataCopy(raw, input[begin * width],
                     DataCopyParams{3, uint16_t(length / 16), uint16_t((h * width - length) / 16),
                                    uint16_t((P - length) / 16)});
        } else {
            for (int c = 0; c < 3; ++c)
                for (int r = begin; r < end; ++r)
                    DataCopy(raw[c * P + (r - begin) * localWidth],
                             input[c * h * width + r * width + colBegin], localWidth);
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        uint64_t dstList[16], srcList[16];
        for (int i = 0; i < 16; i++) {
            dstList[i] = (uint64_t)packed16[i * 16].GetPhyAddr();
            srcList[i] = (uint64_t)raw[(i < 3 ? i : 3) * P].GetPhyAddr();
        }
        TransDataTo5HDParams trans;
        trans.repeatTimes = P / 16;
        trans.dstRepStride = 16;
        trans.srcRepStride = 1;
        TransDataTo5HD<half>(dstList, srcList, trans);
        PipeBarrier<PIPE_V>();
        uint64_t selected = 0;
        GatherMask(packed, packed16, mask, true, 16 * P, {1, 1, 8, 0}, selected);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(imageBuf.Get<half>(), packed, length * 4);
        SetFlag<HardEvent::MTE3_MTE1>(0);
        WaitFlag<HardEvent::MTE3_MTE1>(0);
        LoadData3DParamsV2<half> p;
        p.l1H = end - begin;
        p.l1W = localWidth;
        p.channelSize = 4;
        p.kExtension = K;
        p.mExtension = tileM;
        p.mStartPt = col - colBegin / 2;
        p.filterW = 3;
        p.filterH = 3;
        p.strideW = 2;
        p.strideH = 2;
        p.padList[0] = 1;
        p.padList[1] = 1;
        p.padList[2] = imax(0, 1 - row * 2);
        p.padList[3] = imax(0, lastRow * 2 + 2 - h);
        LoadData(aBuf.Get<half>(), imageBuf.Get<half>(), p);
        LoadData2DParams ld;
        ld.repeatTimes = K / 16 * N / 16;
        ld.srcStride = 1;
        LoadData(bBuf.Get<half>(), weightBuf.Get<half>(), ld);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        MmadParams mm;
        mm.m = tileM;
        mm.n = N;
        mm.k = K;
        Mmad(cBuf.Get<float>(), aBuf.Get<half>(), bBuf.Get<half>(), mm);
        SetFlag<HardEvent::M_V>(0);
        WaitFlag<HardEvent::M_V>(0);
        DataCopyEnhancedParams ep;
        ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(cUb, cBuf.Get<float>(), DataCopyParams{N / 16, uint16_t(tileM / 16), 0, 0}, ep);
        PipeBarrier<PIPE_ALL>();
        if constexpr (HalfOut) {
            if constexpr (Blocked) {
                if (hasBias) image_layout::add_bias(cUb, broadcastBuf.Get<float>(), tileM, N);
                image_layout::activate<Fused>(cUb, tmp, scratchBuf.Get<float>(), tileM, N, false);
                Cast(arenaBuf.Get<half>()[2 * N * tileM], cUb, RoundMode::CAST_NONE, N * tileM);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
                for (int group = 0; group < realN; group += 16)
                    DataCopy(outH[(oc + group) * n + pos * 16],
                        arenaBuf.Get<half>()[2 * N * tileM + group * tileM], realM * 16);
            } else {
            image_layout::finish_half<Fused>(halfOut, arenaBuf.Get<half>()[2 * N * tileM],
                cUb, tmp, scratchBuf.Get<float>(), tileM, N, hasBias);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopy(outH[oc * n + pos], halfOut,
                     DataCopyParams{(uint16_t)realN, (uint16_t)(realM / 16), (uint16_t)((tileM - realM) / 16),
                                    (uint16_t)((n - realM) / 16)});
            }
        } else {
        image_layout::transpose(out, cUb, tileM, N);
        PipeBarrier<PIPE_ALL>();
        if (hasBias) {
            image_layout::broadcast(cUb, scratchBuf.Get<float>(), tileM, N);
            PipeBarrier<PIPE_V>();
            Add(out, out, cUb, N * tileM);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (Fused) {
            Muls(tmp, out, -1.0f, N * tileM);
            PipeBarrier<PIPE_V>();
            Exp(tmp, tmp, N * tileM);
            PipeBarrier<PIPE_V>();
            Adds(tmp, tmp, 1.0f, N * tileM);
            PipeBarrier<PIPE_V>();
            Div(out, out, tmp, N * tileM);
            PipeBarrier<PIPE_V>();
        }
            PipeBarrier<PIPE_ALL>();
            DataCopy(outF[oc * n + pos], out,
                     DataCopyParams{(uint16_t)realN, (uint16_t)(realM / 8), (uint16_t)((tileM - realM) / 8),
                                    (uint16_t)((n - realM) / 8)});
        }
        PipeBarrier<PIPE_ALL>();
    }
}

at::Tensor indices(at::TensorOptions options) {
    static thread_local at::Tensor cached;
    if (cached.defined())
        return cached;
    auto cpu = at::empty({N * K}, at::TensorOptions().dtype(at::kInt));
    auto p = cpu.data_ptr<int>();
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) {
            int src = k < 36 ? (k % 4) * 9 + k / 4 : 36;
            p[(k / 16) * N * 16 + n * 16 + k % 16] = (n * K + src) * 2;
        }
    cached = cpu.to(options.dtype(at::kInt));
    return cached;
}

at::Tensor run(const at::Tensor &x, const at::Tensor &w, const std::optional<at::Tensor> &bias, bool fp32,
                bool fused, bool blocked) {
    TORCH_CHECK(x.numel() <= std::numeric_limits<int>::max(), "stem input exceeds index range");
    auto idx = indices(x.options());
    auto out = blocked ? at::empty({1, w.size(0) / 16, (x.size(2) + 1) / 2, x.size(3) / 2, 16}, x.options()) :
        at::empty({1, w.size(0), (x.size(2) + 1) / 2, x.size(3) / 2}, x.options().dtype(fp32 ? at::kFloat : at::kHalf));
    auto s = thin::stream();
    auto bp = bias ? (uint8_t *)bias->data_ptr() : nullptr;
#define RUN(H, F)                                                                                            \
    stem<H, F><<<8, nullptr, s>>>((uint8_t *)x.data_ptr(), (uint8_t *)w.data_ptr(), bp,                      \
                                  (uint8_t *)out.data_ptr(), (uint8_t *)idx.data_ptr(), x.size(2), x.size(3), \
                                  w.size(0), bias.has_value())
    if (blocked) {
        stem<true, true, true><<<8, nullptr, s>>>((uint8_t *)x.data_ptr(), (uint8_t *)w.data_ptr(), bp,
            (uint8_t *)out.data_ptr(), (uint8_t *)idx.data_ptr(), x.size(2), x.size(3), w.size(0), bias.has_value());
    } else if (fp32) {
        if (fused) {
            RUN(false, true);
        } else {
            RUN(false, false);
        }
    } else {
        if (fused) {
            RUN(true, true);
        } else {
            RUN(true, false);
        }
    }
#undef RUN
    thin::sync();
    return out;
}

at::Tensor conv(const at::Tensor &x, const at::Tensor &w, const std::optional<at::Tensor> &bias,
                bool fp32, bool fused) {
    return run(x, w, bias, fp32, fused, false);
}

at::Tensor conv_blocked(const at::Tensor &x, const at::Tensor &w, const std::optional<at::Tensor> &bias) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && x.dim() == 4 &&
        x.size(0) == 1 && x.size(1) == 3 && x.size(2) > 0 && x.size(3) > 0 &&
        x.size(2) % 32 == 0 && x.size(3) % 32 == 0 &&
        x.scalar_type() == at::kHalf && x.is_contiguous() && !x.requires_grad(), "expected FP16 NCHW 1x3xHxW with positive /32 H/W");
    TORCH_CHECK(w.device() == x.device() && w.scalar_type() == at::kHalf && w.is_contiguous() &&
        w.sizes() == c10::IntArrayRef({32,3,3,3}) && !w.requires_grad(), "expected stem 32x3x3x3 weights");
    if (bias) TORCH_CHECK(bias->device() == x.device() && bias->scalar_type() == at::kHalf &&
        bias->numel() == 32 && bias->is_contiguous() && !bias->requires_grad(), "expected stem FP16 bias");
    thin::select(x.device());
    return run(x, w, bias, false, true, true);
}
}
