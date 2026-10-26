#include <torch/all.h>
#include <torch/library.h>
#include <acl/acl_rt.h>
#include "kernel_operator.h"
#include "cube_layout.h"
#include <map>
using namespace AscendC;
namespace thin {
aclrtStream stream();
void sync();
void select(c10::Device);
}
namespace thin_ops { at::Tensor& copy_strided(at::Tensor&, const at::Tensor&, bool); }
namespace thin_aten { at::Tensor allocation_base(const at::Tensor&); }
namespace image_c4 {
at::Tensor conv(const at::Tensor &, const at::Tensor &, const std::optional<at::Tensor> &, bool, bool);
}
namespace image_conv {
constexpr int TM = 128;
__aicore__ inline int imin(int a, int b) { return a < b ? a : b; }
__aicore__ inline int imax(int a, int b) { return a > b ? a : b; }
__aicore__ inline int round16(int a) { return (a + 15) / 16 * 16; }

__aicore__ inline void transpose_image(LocalTensor<half> dst, LocalTensor<half> src, int plane,
                                       int srcPlane) {
    uint64_t d[16], s[16];
    for (int i = 0; i < 16; i++) {
        d[i] = (uint64_t)dst[i * 16].GetPhyAddr();
        s[i] = (uint64_t)src[i * srcPlane].GetPhyAddr();
    }
    TransDataTo5HDParams p;
    p.repeatTimes = plane / 16;
    p.dstRepStride = plane == 16 ? 0 : 16;
    p.srcRepStride = plane == 16 ? 0 : 1;
    TransDataTo5HD<half>(d, s, p);
}

#include "conv_point.h"
#include "conv_spatial.h"
#include "conv_spatial_stream.h"

template <int TN, bool HalfOut, bool Fused>
__global__ __aicore__ void cube_image(GM_ADDR x, GM_ADDR w, GM_ADDR bias, GM_ADDR y, GM_ADDR index, int ci,
                                      int h, int width, int co, int kernel, int stride, int pad, int oh,
                                      int ow, int dstPlane, bool hasBias, bool wholePoint, bool cacheInput,
                                      int inputChunk, GM_ADDR packedWeight) {
    const int kk = kernel * kernel, kc = kk * 16, n = oh * ow;
    const int step = wholePoint ? ci : 16;
    const int maxH = (ow % TM == 0 ? (TM - 1) / ow : (TM + ow - 2) / ow) * stride + kernel;
    const int maxP = cacheInput ? inputChunk : round16(maxH * width);
    const int rawChannels = cacheInput ? 16 : step;
    TPipe pipe;
    TBuf<TPosition::VECCALC> rawBuf, packBuf, wrBuf, wpBuf, scratchBuf, indexBuf, transBuf, halfBuf, biasBuf;
    TBuf<TPosition::A1> imageBuf;
    TBuf<TPosition::B1> weightBuf;
    TBuf<TPosition::A2> aBuf;
    TBuf<TPosition::B2> bBuf;
    TBuf<TPosition::CO1> cBuf;
    TBuf<TPosition::CO2> cUbBuf;
    pipe.InitBuffer(rawBuf, rawChannels * maxP * 2);
    pipe.InitBuffer(packBuf, imax(rawChannels * maxP * 2, Fused ? TN * TM * 4 : 0));
    pipe.InitBuffer(wrBuf, TN * kc * 2);
    pipe.InitBuffer(wpBuf, TN * kc * 2);
    pipe.InitBuffer(scratchBuf, (imax(maxP, kc) + 32) * 2);
    pipe.InitBuffer(indexBuf, imax(TN * kc * 4, TN * TM * 4));
    pipe.InitBuffer(transBuf, TN * TM * 4);
    pipe.InitBuffer(halfBuf, TN * TM * 2);
    pipe.InitBuffer(biasBuf, TN * 2);
    pipe.InitBuffer(imageBuf, cacheInput ? round16(ci) * h * width * 2 : step * maxP * 2);
    pipe.InitBuffer(weightBuf, TN * kc * ((ci + 15) / 16) * 2);
    pipe.InitBuffer(aBuf, TM * (wholePoint ? ci : kc) * 2);
    pipe.InitBuffer(bBuf, TN * (wholePoint ? ci : kc) * 2);
    pipe.InitBuffer(cBuf, TN * TM * 4);
    pipe.InitBuffer(cUbBuf, TN * TM * 4);
    GlobalTensor<half> input, weights, biases, outH, prepared;
    GlobalTensor<float> outF;
    GlobalTensor<uint32_t> indices;
    input.SetGlobalBuffer((__gm__ half *)x);
    weights.SetGlobalBuffer((__gm__ half *)w);
    biases.SetGlobalBuffer((__gm__ half *)bias);
    outH.SetGlobalBuffer((__gm__ half *)y);
    outF.SetGlobalBuffer((__gm__ float *)y);
    indices.SetGlobalBuffer((__gm__ uint32_t *)index);
    prepared.SetGlobalBuffer((__gm__ half *)packedWeight);
    auto raw = rawBuf.Get<half>(), packed = packBuf.Get<half>(), wr = wrBuf.Get<half>(),
         wp = wpBuf.Get<half>(), scratch = scratchBuf.Get<half>();
    auto wi = indexBuf.Get<uint32_t>();
    Duplicate(raw, (half)0, rawChannels * maxP);
    PipeBarrier<PIPE_ALL>();
    if (cacheInput) {
        for (int cb = 0; cb < ci; cb += 16) {
            int channels = imin(16, ci - cb);
            if (channels != 16) {
                Duplicate(raw[channels * maxP], (half)0, (16 - channels) * maxP);
                PipeBarrier<PIPE_ALL>();
            }
            for (int off = 0; off < h * width; off += maxP) {
                int len = imin(maxP, h * width - off);
                DataCopy(raw, input[cb * h * width + off],
                         DataCopyParams{(uint16_t)channels, (uint16_t)(len / 16),
                                        (uint16_t)((h * width - len) / 16), (uint16_t)((maxP - len) / 16)});
                PipeBarrier<PIPE_ALL>();
                transpose_image(packed, raw, len, maxP);
                PipeBarrier<PIPE_ALL>();
                DataCopy(imageBuf.Get<half>()[cb * h * width + off * 16], packed, len * 16);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
    int spatialTiles = (n + TM - 1) / TM, channelTiles = (co + TN - 1) / TN;
    int cachedOc = -1;
    for (int tile = GetBlockIdx(); tile < spatialTiles * channelTiles; tile += GetBlockNum()) {
        int pos = (tile % spatialTiles) * TM, oc = (tile / spatialTiles) * TN;
        int realM = imin(TM, n - pos), realN = imin(TN, co - oc);
        int row = pos / ow, lastRow = imin(oh - 1, (pos + TM - 1) / ow);
        int begin = imax(0, row * stride - pad), end = imin(h, lastRow * stride - pad + kernel);
        int length = (end - begin) * width, plane = round16(length);
        if (cachedOc != oc) {
            if (packedWeight != nullptr) {
                int coPad = (co + 63) / 64 * 64;
                DataCopy(weightBuf.Get<half>(), prepared[oc * 16],
                         DataCopyParams{(uint16_t)(((ci + 15) / 16) * kk), (uint16_t)TN,
                                        (uint16_t)(coPad - TN), 0});
                PipeBarrier<PIPE_ALL>();
            } else {
                DataCopy(wi, indices, TN * kc);
                PipeBarrier<PIPE_ALL>();
                for (int cb = 0; cb < ci; cb += 16) {
                    int channels = imin(16, ci - cb);
                    if (channels != 16 || realN != TN) {
                        Duplicate(wr, (half)0, TN * kc);
                        PipeBarrier<PIPE_ALL>();
                    }
                    if (ci % 16 == 0) {
                        DataCopy(wr, weights[oc * ci * kk + cb * kk],
                                 DataCopyParams{(uint16_t)realN, (uint16_t)(kc / 16),
                                                (uint16_t)((ci * kk - kc) / 16), 0});
                        PipeBarrier<PIPE_ALL>();
                    } else {
                        for (int no = 0; no < realN; no++) {
                            int off = (oc + no) * ci * kk + cb * kk, aligned = off / 16 * 16,
                                skew = off - aligned;
                            DataCopy(scratch, weights[aligned], round16(channels * kk + skew));
                            PipeBarrier<PIPE_ALL>();
                            Adds(wr[no * kc], scratch[skew], (half)0, channels * kk);
                            PipeBarrier<PIPE_ALL>();
                        }
                    }
                    Gather(wp, wr, wi, 0, TN * kc);
                    PipeBarrier<PIPE_ALL>();
                    DataCopy(weightBuf.Get<half>()[(cb / 16) * TN * kc], wp, TN * kc);
                    PipeBarrier<PIPE_ALL>();
                }
            }
            if (hasBias) {
                DataCopy(biasBuf.Get<half>(), biases[oc], round16(realN));
                PipeBarrier<PIPE_ALL>();
                Cast(scratchBuf.Get<float>(), biasBuf.Get<half>(), RoundMode::CAST_NONE, TN);
                PipeBarrier<PIPE_V>();
                image_layout::broadcast(indexBuf.Get<float>(), scratchBuf.Get<float>(), TM, TN);
                PipeBarrier<PIPE_ALL>();
            }
            cachedOc = oc;
        }
        for (int cb = 0; cb < ci; cb += step) {
            int channels = imin(step, ci - cb);
            if (!cacheInput) {
                if (channels != 16 && cb != 0) {
                    Duplicate(raw[channels * maxP], (half)0, (16 - channels) * maxP);
                    PipeBarrier<PIPE_ALL>();
                }
                if (h * width % 16 == 0 && begin * width % 16 == 0 && length % 16 == 0) {
                    DataCopy(raw, input[cb * h * width + begin * width],
                             DataCopyParams{(uint16_t)channels, (uint16_t)(length / 16),
                                            (uint16_t)((h * width - length) / 16),
                                            (uint16_t)((maxP - length) / 16)});
                    SetFlag<HardEvent::MTE2_V>(0);
                    WaitFlag<HardEvent::MTE2_V>(0);
                } else {
                    for (int c = 0; c < channels; c++) {
                        int off = (cb + c) * h * width + begin * width, aligned = off / 16 * 16,
                            skew = off - aligned;
                        DataCopy(scratch, input[aligned], round16(length + skew));
                        PipeBarrier<PIPE_ALL>();
                        Adds(raw[c * maxP], scratch[skew], (half)0, length);
                        PipeBarrier<PIPE_ALL>();
                    }
                }
                for (int c1 = 0; c1 < step; c1 += 16)
                    transpose_image(packed[c1 * plane], raw[c1 * maxP], plane, maxP);
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopy(imageBuf.Get<half>(), packed, plane * step);
                SetFlag<HardEvent::MTE3_MTE1>(0);
                WaitFlag<HardEvent::MTE3_MTE1>(0);
            }
            LoadData3DParamsV2<half> p;
            p.l1H = cacheInput ? h : end - begin;
            p.l1W = width;
            p.channelSize = step;
            p.kExtension = wholePoint ? ci : kc;
            p.mExtension = TM;
            p.mStartPt = cacheInput ? pos : pos % ow;
            p.filterW = kernel;
            p.filterH = kernel;
            p.strideW = stride;
            p.strideH = stride;
            p.padList[0] = pad;
            p.padList[1] = pad;
            p.padList[2] = cacheInput ? pad : imax(0, pad - row * stride);
            p.padList[3] = cacheInput ? pad : imax(0, lastRow * stride - pad + kernel - h);
            if (cb != 0)
                WaitFlag<HardEvent::M_MTE1>(0);
            LoadData(aBuf.Get<half>(), imageBuf.Get<half>()[cacheInput ? cb * h * width : 0], p);
            LoadData2DParams ld;
            ld.repeatTimes = (wholePoint ? ci / 16 : kk) * TN / 16;
            ld.srcStride = 1;
            LoadData(bBuf.Get<half>(), weightBuf.Get<half>()[(cb / 16) * TN * kc], ld);
            SetFlag<HardEvent::MTE1_M>(0);
            WaitFlag<HardEvent::MTE1_M>(0);
            MmadParams mm;
            mm.m = TM;
            mm.n = TN;
            mm.k = wholePoint ? ci : kc;
            mm.isBias = cb != 0;
            Mmad(cBuf.Get<float>(), aBuf.Get<half>(), bBuf.Get<half>(), mm);
            SetFlag<HardEvent::M_MTE1>(0);
        }
        WaitFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::M_V>(0);
        WaitFlag<HardEvent::M_V>(0);
        DataCopyEnhancedParams ep;
        ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(cUbBuf.Get<float>(), cBuf.Get<float>(),
                 DataCopyParams{(uint16_t)(TN / 16), (uint16_t)(TM / 16), 0, 0}, ep);
        PipeBarrier<PIPE_ALL>();
        auto trans = transBuf.Get<float>();
        image_layout::transpose(trans, cUbBuf.Get<float>(), TM, TN);
        PipeBarrier<PIPE_ALL>();
        if (hasBias) {
            Add(trans, trans, indexBuf.Get<float>(), TN * TM);
            PipeBarrier<PIPE_ALL>();
        }
        if constexpr (Fused) {
            auto tmp = packBuf.Get<float>();
            Muls(tmp, trans, -1.0f, TN * TM);
            PipeBarrier<PIPE_V>();
            Exp(tmp, tmp, TN * TM);
            PipeBarrier<PIPE_V>();
            Adds(tmp, tmp, 1.0f, TN * TM);
            PipeBarrier<PIPE_V>();
            Div(trans, trans, tmp, TN * TM);
            PipeBarrier<PIPE_ALL>();
        }
        if constexpr (HalfOut) {
            Cast(halfBuf.Get<half>(), trans, RoundMode::CAST_NONE, TN * TM);
            PipeBarrier<PIPE_ALL>();
            int copied = round16(realM);
            DataCopy(outH[oc * dstPlane + pos], halfBuf.Get<half>(),
                     DataCopyParams{(uint16_t)realN, (uint16_t)(copied / 16), (uint16_t)((TM - copied) / 16),
                                    (uint16_t)((dstPlane - copied) / 16)});
        } else {
            int copied = round16(realM);
            DataCopy(outF[oc * dstPlane + pos], trans,
                     DataCopyParams{(uint16_t)realN, (uint16_t)(copied / 8), (uint16_t)((TM - copied) / 8),
                                    (uint16_t)((dstPlane - copied) / 8)});
        }
        PipeBarrier<PIPE_ALL>();
    }
}

at::Tensor indices(int kernel, int tn, at::TensorOptions opts) {
    static thread_local std::map<int, at::Tensor> cache;
    int key = kernel * 100 + tn;
    if (cache.count(key))
        return cache.at(key);
    int kk = kernel * kernel, kc = kk * 16;
    auto cpu = at::empty({tn * kc}, at::TensorOptions().dtype(at::kInt));
    auto p = cpu.data_ptr<int>();
    for (int tap = 0; tap < kk; tap++)
        for (int n = 0; n < tn; n++)
            for (int c = 0; c < 16; c++)
                p[tap * tn * 16 + (n / 16) * 256 + (n % 16) * 16 + c] = (n * kc + c * kk + tap) * 2;
    auto device = cpu.to(opts.dtype(at::kInt));
    cache.emplace(key, device);
    return device;
}

__global__ __aicore__ void trim(GM_ADDR src, GM_ADDR dst, int co, int n, int np, bool halfOut) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> rawBuf, outBuf, halfBuf;
    pipe.InitBuffer(rawBuf, 4096 * 4);
    pipe.InitBuffer(outBuf, 2048 * 4);
    pipe.InitBuffer(halfBuf, 2048 * 2);
    GlobalTensor<float> in, out;
    GlobalTensor<half> outH;
    in.SetGlobalBuffer((__gm__ float *)src);
    out.SetGlobalBuffer((__gm__ float *)dst);
    outH.SetGlobalBuffer((__gm__ half *)dst);
    for (int base = GetBlockIdx() * 2048; base < co * n; base += GetBlockNum() * 2048) {
        int count = imin(2048, co * n - base);
        for (int i = 0; i < count;) {
            int c = (base + i) / n, pos = (base + i) % n, len = imin(n - pos, count - i), off = c * np + pos,
                align = off / 8 * 8, skew = off - align;
            DataCopy(rawBuf.Get<float>(), in[align], (len + skew + 7) / 8 * 8);
            PipeBarrier<PIPE_ALL>();
            Adds(outBuf.Get<float>()[i], rawBuf.Get<float>()[skew], 0.0f, len);
            PipeBarrier<PIPE_ALL>();
            i += len;
        }
        if (halfOut) {
            Cast(halfBuf.Get<half>(), outBuf.Get<float>(), RoundMode::CAST_NONE, (count + 63) / 64 * 64);
            PipeBarrier<PIPE_ALL>();
            DataCopy(outH[base], halfBuf.Get<half>(), (count + 15) / 16 * 16);
        } else
            DataCopy(out[base], outBuf.Get<float>(), (count + 7) / 8 * 8);
        PipeBarrier<PIPE_ALL>();
    }
}

__global__ __aicore__ void pack_weights(GM_ADDR w, GM_ADDR packedWeight, GM_ADDR index, int ci, int co,
                                        int kernel) {
    int kk = kernel * kernel, kc = kk * 16, coPad = (co + 63) / 64 * 64, cbCount = (ci + 15) / 16;
    TPipe pipe;
    TBuf<TPosition::VECCALC> rawBuf, packedBuf, scratchBuf, indexBuf;
    pipe.InitBuffer(rawBuf, 16 * kc * 2);
    pipe.InitBuffer(packedBuf, 16 * kc * 2);
    pipe.InitBuffer(scratchBuf, (kc + 32) * 2);
    pipe.InitBuffer(indexBuf, 16 * kc * 4);
    GlobalTensor<half> input, output;
    GlobalTensor<uint32_t> indices;
    input.SetGlobalBuffer((__gm__ half *)w);
    output.SetGlobalBuffer((__gm__ half *)packedWeight);
    indices.SetGlobalBuffer((__gm__ uint32_t *)index);
    auto raw = rawBuf.Get<half>(), scratch = scratchBuf.Get<half>(), packed = packedBuf.Get<half>();
    auto wi = indexBuf.Get<uint32_t>();
    DataCopy(wi, indices, 16 * kc);
    PipeBarrier<PIPE_ALL>();
    for (int tile = GetBlockIdx(); tile < cbCount * (coPad / 16); tile += GetBlockNum()) {
        int cb = tile / (coPad / 16) * 16, oc = tile % (coPad / 16) * 16;
        int channels = imin(16, ci - cb), outputs = imax(0, imin(16, co - oc));
        Duplicate(raw, (half)0, 16 * kc);
        PipeBarrier<PIPE_ALL>();
        if (outputs > 0) {
            if (ci % 16 == 0) {
                DataCopy(raw, input[oc * ci * kk + cb * kk],
                         DataCopyParams{(uint16_t)outputs, (uint16_t)(kc / 16),
                                        (uint16_t)((ci * kk - kc) / 16), 0});
                PipeBarrier<PIPE_ALL>();
            } else {
                for (int no = 0; no < outputs; no++) {
                    int off = (oc + no) * ci * kk + cb * kk, aligned = off / 16 * 16, skew = off - aligned;
                    DataCopy(scratch, input[aligned], round16(channels * kk + skew));
                    PipeBarrier<PIPE_ALL>();
                    Adds(raw[no * kc], scratch[skew], (half)0, channels * kk);
                    PipeBarrier<PIPE_ALL>();
                }
            }
        }
        Gather(packed, raw, wi, 0, 16 * kc);
        PipeBarrier<PIPE_ALL>();
        DataCopy(output[(cb / 16) * kk * coPad * 16 + oc * 16], packed,
                 DataCopyParams{(uint16_t)kk, 16, 0, (uint16_t)(coPad - 16)});
        PipeBarrier<PIPE_ALL>();
    }
}

at::Tensor prepare(const at::Tensor &w) {
    TORCH_CHECK(w.device().type() == c10::DeviceType::PrivateUse1 && w.dim() == 4 &&
                    w.scalar_type() == at::kHalf && w.is_contiguous() && w.storage_offset() == 0 &&
                    !w.requires_grad(),
                "expected allocation-base fp16 OIHW inference weights on NPU");
    int ci = w.size(1), co = w.size(0), k = w.size(2);
    TORCH_CHECK(ci > 0 && co > 0 && (k == 1 || k == 3 || k == 6) && w.size(3) == k,
                "expected supported square positive OIHW shape");
    thin::select(w.device());
    auto idx = indices(k, 16, w.options());
    auto packed = at::empty({(ci + 15) / 16, k * k, (co + 63) / 64 * 64, 16}, w.options());
    pack_weights<<<8, nullptr, thin::stream()>>>((uint8_t *)w.data_ptr(), (uint8_t *)packed.data_ptr(),
                                                 (uint8_t *)idx.data_ptr(), ci, co, k);
    thin::sync();
    return packed;
}

at::Tensor run(const at::Tensor &x, const at::Tensor &w, const std::optional<at::Tensor> &bias,
               int64_t stride, int64_t pad, bool fp32, bool fused,
               const std::optional<at::Tensor> &prepared) {
    TORCH_CHECK(x.dim() == 4 && w.dim() == 4 && x.size(0) == 1 && x.size(1) == w.size(1) &&
                    w.size(2) == w.size(3),
                "expected N=1 NCHW/OIHW");
    TORCH_CHECK(x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf && x.is_contiguous() &&
                    w.is_contiguous(),
                "expected contiguous fp16 input and weight");
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && w.device() == x.device(),
                "expected same NPU device");
    TORCH_CHECK(!x.requires_grad() && !w.requires_grad(), "convolution is inference-only");
    TORCH_CHECK(x.storage_offset() % 16 == 0 && w.storage_offset() == 0,
                "expected aligned input and allocation-base weights");
    int ci = x.size(1), h = x.size(2), width = x.size(3), co = w.size(0), k = w.size(2);
    TORCH_CHECK(ci > 0 && h > 0 && width > 0 && co > 0, "C/H/W/Cout must be positive");
    TORCH_CHECK((k == 1 || k == 3) && (stride == 1 || stride == 2) && pad == k / 2,
                "expected k1/k3 s1/s2 same padding");
    if (bias)
        TORCH_CHECK(bias->device() == x.device() && bias->scalar_type() == at::kHalf &&
                        bias->is_contiguous() && bias->storage_offset() == 0 && bias->dim() == 1 &&
                        bias->numel() == co && !bias->requires_grad(),
                    "expected contiguous fp16 Cout-vector bias on the same device");
    if (prepared)
        TORCH_CHECK(prepared->device() == x.device() && prepared->scalar_type() == at::kHalf &&
                        prepared->is_contiguous() && prepared->storage_offset() == 0 &&
                        prepared->sizes() ==
                            c10::IntArrayRef({(ci + 15) / 16, k * k, (co + 63) / 64 * 64, 16}),
                    "prepared weights do not match the convolution shape");
    if (k == 1 && stride == 1 && h * width % 16 != 0) {
        int plane = h * width, paddedPlane = (plane + 15) / 16 * 16;
        auto buffer = at::empty({1, ci, paddedPlane / 16, 16}, x.options());
        auto valid = buffer.as_strided({1, ci, h, width}, {ci * paddedPlane, paddedPlane, width, 1});
        thin_ops::copy_strided(valid, x, false);
        auto result = run(buffer, w, bias, stride, pad, fp32, fused, prepared);
        auto view = result.as_strided({1, co, h, width}, {co * paddedPlane, paddedPlane, width, 1});
        auto output = at::empty({1, co, h, width}, result.options());
        thin_ops::copy_strided(output, view, false);
        return output;
    }
    // A point convolution is spatially independent; fold wide planes to bound L1/UB strips.
    if (k == 1 && stride == 1 && width > TM && h * width % 16 == 0)
        return run(x.view({1, ci, h * width / 16, 16}), w, bias, stride, pad, fp32, fused,
                   prepared).view({1, co, h, width});
    if (ci == 3 && width == 640 && k == 3 && stride == 2)
        return image_c4::conv(x, w, bias, fp32, fused);
    int oh = (h + 2 * pad - k) / stride + 1, ow = (width + 2 * pad - k) / stride + 1, n = oh * ow,
        np = (n + 15) / 16 * 16;
    if (ci == 256 && h == 40 && width == 40 && k == 3 && co % 64 == 0) {
        auto packed = prepared ? *prepared : prepare(w);
        auto y = at::empty({1, co, oh, ow}, x.options().dtype(fp32 ? at::kFloat : at::kHalf));
        bool down = stride == 2 && co % 128 == 0;
        int maxP = down ? 880 : (stride == 1 ? 560 : 1600);
        GM_ADDR bp = bias ? (GM_ADDR)bias->data_ptr() : nullptr;
        auto s = thin::stream();
#define STREAM(M, N, C, K, H, F) cube_spatial_stream<M, N, C, K, H, F><<<8, nullptr, s>>>((GM_ADDR)x.data_ptr(), \
        (GM_ADDR)packed.data_ptr(), bp, (GM_ADDR)y.data_ptr(), ci, co, h, width, stride, oh, ow, maxP, bias.has_value())
#define STREAM_OUTPUT(M, N, C, K) \
        if (fp32) { \
            if (fused) { STREAM(M, N, C, K, false, true); } else { STREAM(M, N, C, K, false, false); } \
        } else { \
            if (fused) { STREAM(M, N, C, K, true, true); } else { STREAM(M, N, C, K, true, false); } \
        }
        if (down) { STREAM_OUTPUT(208, 128, 256, 64); } else { STREAM_OUTPUT(400, 64, 32, 16); }
#undef STREAM_OUTPUT
#undef STREAM
        thin::sync();
        return y;
    }
    bool pointWide = ci >= 384 && n == 1600;
    int pointM = pointWide ? 256 : 128;
    int pointN = pointWide ? 64 : (co % 128 == 0 ? 128 : (co % 64 == 0 ? 64 : 32));
    int pointWeightCo = std::min(co, (1048576 - 4 * ci * pointM) / (ci * 2) / pointN * pointN);
    int pointTiles = (n + pointM - 1) / pointM;
    if (pointTiles < 8)
        pointWeightCo = std::min(pointWeightCo, (co / ((8 + pointTiles - 1) / pointTiles) + pointN - 1) / pointN * pointN);
    if (!(ci == 256 && n == 1600) && k == 1 && stride == 1 && ci >= 32 && ci % 16 == 0 && co % pointN == 0 && n % 16 == 0 &&
        pointWeightCo >= pointN) {
        auto packed = prepared ? *prepared : prepare(w);
        auto y = at::empty({1, co, oh, ow}, x.options().dtype(fp32 ? at::kFloat : at::kHalf));
        GM_ADDR bp = bias ? (GM_ADDR)bias->data_ptr() : nullptr;
        auto s = thin::stream();
#define POINT(N, H, F, W) cube_point<N, H, F, W><<<8, nullptr, s>>>((GM_ADDR)x.data_ptr(), \
        (GM_ADDR)packed.data_ptr(), bp, (GM_ADDR)y.data_ptr(), ci, co, n, pointWeightCo, bias.has_value())
#define POINT_OUTPUT(N, W) \
        if (fp32) { \
            if (fused) { POINT(N, false, true, W); } else { POINT(N, false, false, W); } \
        } else { \
            if (fused) { POINT(N, true, true, W); } else { POINT(N, true, false, W); } \
        }
        if (pointWide) { POINT_OUTPUT(64, true); }
        else if (pointN == 128) { POINT_OUTPUT(128, false); }
        else if (pointN == 64) { POINT_OUTPUT(64, false); }
        else { POINT_OUTPUT(32, false); }
#undef POINT_OUTPUT
#undef POINT
        thin::sync();
        return y;
    }
    int rowGroup = 1;
    while (width * rowGroup % 16 != 0 || rowGroup % stride != 0) ++rowGroup;
    int spatialN = width >= 320 || co == 32 ? 32 : 64;
    int spatialH = ((112 + ow - 2) / ow) * stride + 3 + 2 * (rowGroup - 1);
    int spatialP = std::min(h, (spatialH + rowGroup - 1) / rowGroup * rowGroup) * width;
    bool spatialCache = ci * h * width * 2 + spatialN * ci * 18 <= 1048576 &&
                        h * width % 16 == 0;
    int spatialL1 = spatialCache ? ci * h * width * 2 : 4 * ci * spatialP;
    if (spatialCache) spatialP = std::min(1024, h * width);
    int weightCo = (1048576 - spatialL1) / (ci * 18) / spatialN * spatialN;
    weightCo = std::min(co, weightCo);
    int stageBuffers = spatialCache || spatialP <= 1536 ? 2 : 1;
    int spatialUB = 2 * std::max(stageBuffers * 16 * spatialP * 2, spatialN * 112 * 4) +
                    spatialN * 112 * 4 + spatialN * 6;
    bool spatialClass = ci >= 16;
    if (spatialClass && k == 3 && ci % 16 == 0 && co % spatialN == 0 && n % 16 == 0 &&
        h % rowGroup == 0 && weightCo >= spatialN && spatialUB <= 262144) {
        auto packed = prepared ? *prepared : prepare(w);
        auto y = at::empty({1, co, oh, ow}, x.options().dtype(fp32 ? at::kFloat : at::kHalf));
        GM_ADDR bp = bias ? (GM_ADDR)bias->data_ptr() : nullptr;
        auto s = thin::stream();
#define SPATIAL(N, H, F) cube_spatial<N, H, F><<<8, nullptr, s>>>((GM_ADDR)x.data_ptr(), \
        (GM_ADDR)packed.data_ptr(), bp, (GM_ADDR)y.data_ptr(), ci, co, h, width, stride, oh, ow, \
        spatialP, weightCo, bias.has_value(), spatialCache)
#define SPATIAL_OUTPUT(N) \
        if (fp32) { \
            if (fused) { SPATIAL(N, false, true); } else { SPATIAL(N, false, false); } \
        } else { \
            if (fused) { SPATIAL(N, true, true); } else { SPATIAL(N, true, false); } \
        }
        if (spatialN == 32) { SPATIAL_OUTPUT(32); } else { SPATIAL_OUTPUT(64); }
#undef SPATIAL_OUTPUT
#undef SPATIAL
        thin::sync();
        return y;
    }
    int tn = (width >= 320 || co <= 32) ? 16 : 64;
    bool cacheInput = h * width % 16 == 0 && ((ci + 15) / 16 * 16 * h * width * 2 +
                                              tn * k * k * ((ci + 15) / 16) * 16 * 2) <= 1048576;
    int maxH = (ow % TM == 0 ? (TM - 1) / ow : (TM + ow - 2) / ow) * stride + k,
        maxP = cacheInput ? std::min(255 * 16, h * width) : (maxH * width + 15) / 16 * 16;
    auto ubBytes = [&](int step) {
        int raw = (cacheInput ? 16 : step) * maxP * 2;
        return std::max(raw, fused ? tn * TM * 4 : 0) + raw + tn * k * k * 16 * 4 +
               std::max(tn * k * k * 16 * 4, tn * TM * 4) + (std::max(maxP, k * k * 16) + 32) * 2 +
               tn * TM * 10 + tn * 2;
    };
    if (cacheInput)
        while (maxP > 16 && ubBytes(16) > 262144)
            maxP -= 16;
    bool wholePoint = k == 1 && stride == 1 && ci % 16 == 0 && ubBytes(ci) <= 262144 &&
                      TM * ci * 2 <= 65536 && tn * ci * 2 <= 65536;
    int ub = ubBytes(wholePoint ? ci : 16);
    TORCH_CHECK(ub <= 262144, "UB allocation exceeds 310P capacity: ", ub,
                " for input ", x.sizes(), " and weight ", w.sizes());
    auto idx = indices(k, tn, x.options());
    auto y = at::empty({1, co, oh, ow}, x.options().dtype(fp32 ? at::kFloat : at::kHalf));
    bool needsTrim = n % 16 != 0;
    auto dest = needsTrim ? at::empty({co, np}, x.options().dtype(at::kFloat)) : y;
    GM_ADDR bp = bias ? (GM_ADDR)bias->data_ptr() : nullptr;
    auto pw = prepared ? (uint8_t *)prepared->data_ptr() : nullptr;
    auto s = thin::stream();
#define LAUNCH(T, H, F)                                                                                      \
    cube_image<T, H, F><<<8, nullptr, s>>>((GM_ADDR)x.data_ptr(), (GM_ADDR)w.data_ptr(), bp,                 \
                                           (GM_ADDR)dest.data_ptr(), (GM_ADDR)idx.data_ptr(), ci, h, width,  \
                                           co, k, stride, pad, oh, ow, needsTrim ? np : n, bias.has_value(), \
                                           wholePoint, cacheInput, maxP, pw)
    if (tn == 16) {
        if (!fp32 && !needsTrim) {
            if (fused) {
                LAUNCH(16, true, true);
            } else {
                LAUNCH(16, true, false);
            }
        } else {
            if (fused) {
                LAUNCH(16, false, true);
            } else {
                LAUNCH(16, false, false);
            }
        }
    } else {
        if (!fp32 && !needsTrim) {
            if (fused) {
                LAUNCH(64, true, true);
            } else {
                LAUNCH(64, true, false);
            }
        } else {
            if (fused) {
                LAUNCH(64, false, true);
            } else {
                LAUNCH(64, false, false);
            }
        }
    }
#undef LAUNCH
    if (needsTrim)
        trim<<<8, nullptr, s>>>((GM_ADDR)dest.data_ptr(), (GM_ADDR)y.data_ptr(), co, n, np, !fp32);
    thin::sync();
    return y;
}
at::Tensor conv(const at::Tensor &x, const at::Tensor &w, const std::optional<at::Tensor> &bias,
                int64_t stride, int64_t pad, bool fp32, bool fused) {
    return run(x, w, bias, stride, pad, fp32, fused, std::nullopt);
}
at::Tensor conv_prepared(const at::Tensor &x, const at::Tensor &w, const at::Tensor &packed,
                         const std::optional<at::Tensor> &bias, int64_t stride, int64_t pad, bool fp32,
                         bool fused) {
    auto input = !x.requires_grad() && x.is_contiguous() && x.storage_offset() % 16 == 0
                     ? x.detach()
                     : thin_aten::allocation_base(x);
    return run(input, w, bias, stride, pad, fp32, fused, packed);
}
}
