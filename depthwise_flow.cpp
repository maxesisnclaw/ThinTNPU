#include "kernel_operator.h"
#include <acl/acl_rt.h>
#include <torch/all.h>
#include <torch/library.h>
#include <algorithm>
#include <limits>
#include <optional>
using namespace AscendC;

namespace thin {
aclrtStream stream();
void sync();
void select(c10::Device);
} // namespace thin

namespace depthwise_flow {
constexpr int C0 = 16;
constexpr int64_t UB_BYTES = 262144;

__aicore__ inline int imin(int a, int b) { return a < b ? a : b; }
__aicore__ inline int imax(int a, int b) { return a > b ? a : b; }

template <bool Fused>
__global__ __aicore__ void depthwise_kernel(GM_ADDR x, GM_ADDR pw, GM_ADDR bp, GM_ADDR y,
                                            int c1Count, int h, int w, int kernel, int stride,
                                            int pad, int oh, int ow, int tileRows, int numTiles,
                                            int padW, bool hasBias) {
    const int padHmax = (tileRows - 1) * stride + kernel;
    const int kk = kernel * kernel;
    const uint8_t srcRep = (uint8_t)(stride * 2);

    TPipe pipe;
    TBuf<TPosition::VECCALC> padBuf, inBuf, accBuf, scratchBuf, outBuf, wHalfBuf, wF32Buf, bHalfBuf, bF32Buf, broadcastBuf;
    pipe.InitBuffer(padBuf, padHmax * padW * C0 * (int)sizeof(half));
    pipe.InitBuffer(inBuf, padHmax * padW * C0 * (int)sizeof(float));
    pipe.InitBuffer(accBuf, tileRows * ow * C0 * (int)sizeof(float));
    pipe.InitBuffer(scratchBuf, tileRows * ow * C0 * (int)sizeof(float));
    pipe.InitBuffer(outBuf, tileRows * ow * C0 * (int)sizeof(half));
    pipe.InitBuffer(wHalfBuf, kk * C0 * (int)sizeof(half));
    pipe.InitBuffer(wF32Buf, kk * C0 * (int)sizeof(float));
    pipe.InitBuffer(bHalfBuf, C0 * (int)sizeof(half));
    pipe.InitBuffer(bF32Buf, C0 * (int)sizeof(float));
    pipe.InitBuffer(broadcastBuf, 64 * (int)sizeof(float));

    auto padHalf = padBuf.Get<half>();
    auto inF32 = inBuf.Get<float>();
    auto acc = accBuf.Get<float>();
    auto scratch = scratchBuf.Get<float>();
    auto halfOut = outBuf.Get<half>();
    auto wHalf = wHalfBuf.Get<half>();
    auto wF32 = wF32Buf.Get<float>();
    auto bHalf = bHalfBuf.Get<half>();
    auto bF32 = bF32Buf.Get<float>();
    auto broadcast = broadcastBuf.Get<float>();

    GlobalTensor<half> input, weights, biases, outH;
    input.SetGlobalBuffer((__gm__ half *)x);
    weights.SetGlobalBuffer((__gm__ half *)pw);
    if (hasBias) biases.SetGlobalBuffer((__gm__ half *)bp);
    outH.SetGlobalBuffer((__gm__ half *)y);

    const int64_t jobs = (int64_t)c1Count * numTiles;
    for (int64_t job = GetBlockIdx(); job < jobs; job += GetBlockNum()) {
        const int c1 = job / numTiles;
        const int tile = job % numTiles;
        const int r0 = tile * tileRows;
        const int rows = imin(tileRows, oh - r0);
        if (rows <= 0) continue;
        const int padH = (rows - 1) * stride + kernel;
        const int outPixels = rows * ow;

        DataCopy(wHalf, weights[c1 * kk * C0], kk * C0);
        if (hasBias) DataCopy(bHalf, biases[c1 * C0], C0);

        Duplicate(padHalf, (half)0, padH * padW * C0);
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);

        const int topIr = r0 * stride - pad;
        const int firstIr = imax(0, topIr);
        const int lastIr = imin(h - 1, topIr + padH - 1);
        const int nrows = lastIr - firstIr + 1;
        if (nrows > 0) {
            const int bufOff = firstIr - topIr;
            DataCopy(padHalf[bufOff * padW * C0 + pad * C0], input[(c1 * h + firstIr) * w * C0],
                     DataCopyParams{(uint16_t)nrows, (uint16_t)w, 0, (uint16_t)(padW - w)});
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        Cast(wF32, wHalf, RoundMode::CAST_NONE, kk * C0);
        if (hasBias) Cast(bF32, bHalf, RoundMode::CAST_NONE, C0);
        Cast(inF32, padHalf, RoundMode::CAST_NONE, padH * padW * C0);
        PipeBarrier<PIPE_V>();

        Duplicate(acc, 0.0f, outPixels * C0);
        PipeBarrier<PIPE_V>();

        for (int ky = 0; ky < kernel; ++ky) {
            for (int kx = 0; kx < kernel; ++kx) {
                auto w16 = wF32[(ky * kernel + kx) * C0];
                for (int c = 0; c < 4; ++c) Adds(broadcast[c * C0], w16, 0.0f, C0);
                PipeBarrier<PIPE_V>();
                for (int lr = 0; lr < rows; ++lr) {
                    const int srcRowBase = (lr * stride + ky) * padW * C0 + kx * C0;
                    const int dstRowBase = lr * ow * C0;
                    if (stride == 1) {
                        for (int c = 0; c < ow / 4 * 4; c += 4 * 255) {
                            int cnt = imin(255, ow / 4 - c / 4);
                            Mul(scratch[dstRowBase + c * C0], inF32[srcRowBase + c * C0], broadcast,
                                64, cnt, {1, 1, 1, 8, 8, 0});
                        }
                        if (ow % 4) Mul(scratch[dstRowBase + ow / 4 * 64], inF32[srcRowBase + ow / 4 * 64],
                            broadcast, (ow % 4) * C0);
                    } else {
                        for (int c = 0; c < ow; c += 255) {
                            const int cnt = imin(255, ow - c);
                            Mul(scratch[dstRowBase + c * C0], inF32[srcRowBase + c * stride * C0], w16,
                                C0, cnt, {1, 1, 1, 2, srcRep, 0});
                        }
                    }
                }
                PipeBarrier<PIPE_V>();
                for (int lr = 0; lr < rows; ++lr) {
                    const int base = lr * ow * C0;
                    Add(acc[base], acc[base], scratch[base], ow * C0);
                }
                PipeBarrier<PIPE_V>();
            }
        }

        if (hasBias) {
            for (int p = 0; p < outPixels; p += 255) {
                const int cnt = imin(255, outPixels - p);
                Add(acc[p * C0], acc[p * C0], bF32, C0, cnt, {1, 1, 1, 2, 2, 0});
            }
            PipeBarrier<PIPE_V>();
        }

        if constexpr (Fused) {
            const int n = outPixels * C0;
            Muls(scratch, acc, -1.0f, n);
            PipeBarrier<PIPE_V>();
            Exp(scratch, scratch, n);
            PipeBarrier<PIPE_V>();
            Adds(scratch, scratch, 1.0f, n);
            PipeBarrier<PIPE_V>();
            Div(acc, acc, scratch, n);
            PipeBarrier<PIPE_V>();
        }

        Cast(halfOut, acc, RoundMode::CAST_NONE, outPixels * C0);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(outH[(c1 * oh + r0) * ow * C0], halfOut, DataCopyParams{(uint16_t)rows, (uint16_t)ow, 0, 0});
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        // The next tile's MTE2 weight DMA must wait for this tile's vector reads.
        PipeBarrier<PIPE_ALL>();
    }
}

at::Tensor depthwise(const at::Tensor &x, const at::Tensor &packed,
                     const std::optional<at::Tensor> &bias, int64_t kernel, int64_t stride, bool fused) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && x.dim() == 5 && x.size(0) == 1 &&
                    x.size(4) == 16 && x.scalar_type() == at::kHalf && x.is_contiguous() &&
                    x.storage_offset() % 16 == 0 && !x.requires_grad(),
                "expected aligned contiguous N=1 FP16 NC1HWC0 input, C0=16");
    const int c1 = x.size(1), c = c1 * 16, h = x.size(2), w = x.size(3);
    TORCH_CHECK(c1 > 0 && h > 0 && w > 0, "depthwise requires positive C1/H/W");
    TORCH_CHECK(x.numel() <= std::numeric_limits<int>::max(), "depthwise input exceeds index range");
    TORCH_CHECK(kernel >= 1 && kernel % 2 == 1, "depthwise kernel must be a positive odd integer");
    TORCH_CHECK(stride == 1 || stride == 2, "depthwise supports stride 1 or 2");
    const int pad = (int(kernel) - 1) / 2;
    TORCH_CHECK(h + 2 * pad >= int(kernel) && w + 2 * pad >= int(kernel), "kernel exceeds padded input");
    const int oh = (h + 2 * pad - int(kernel)) / int(stride) + 1;
    const int ow = (w + 2 * pad - int(kernel)) / int(stride) + 1;
    TORCH_CHECK(oh > 0 && ow > 0, "depthwise output is empty for the given shape");
    const int padW = w + 2 * pad;
    TORCH_CHECK(w <= 65535 && ow <= 65535 && padW <= 65535, "depthwise width exceeds DataCopy burst range");

    TORCH_CHECK(packed.device() == x.device() && packed.scalar_type() == at::kHalf &&
                    packed.is_contiguous() && packed.storage_offset() == 0 && !packed.requires_grad() &&
                    packed.dim() == 4 && packed.size(0) == c1 && packed.size(1) == kernel &&
                    packed.size(2) == kernel && packed.size(3) == 16,
                "expected packed depthwise weight [C/16, K, K, 16] FP16 on the input device");
    const bool hasBias = bias.has_value();
    if (hasBias)
        TORCH_CHECK(bias->device() == x.device() && bias->scalar_type() == at::kHalf &&
                        bias->is_contiguous() && bias->storage_offset() == 0 && !bias->requires_grad() &&
                        bias->dim() == 1 && bias->numel() == c,
                    "expected same-device contiguous FP16 bias of length C");

    auto fits = [&](int tr) {
        int64_t padHmax = (int64_t)(tr - 1) * stride + kernel;
        int64_t padBytes = padHmax * padW * 16 * ((int)sizeof(half) + (int)sizeof(float));
        int64_t outBytes = (int64_t)tr * ow * 16 * (2 * (int)sizeof(float) + (int)sizeof(half));
        int64_t miscBytes = (int64_t)kernel * kernel * 16 * ((int)sizeof(half) + (int)sizeof(float)) +
                            16 * ((int)sizeof(half) + (int)sizeof(float)) + 64 * (int)sizeof(float);
        return padBytes + outBytes + miscBytes <= UB_BYTES;
    };
    TORCH_CHECK(fits(1), "depthwise row working set exceeds unified buffer; width too large");
    int tileRows = oh;
    while (tileRows > 1 && !fits(tileRows)) --tileRows;
    const int numTiles = (oh + tileRows - 1) / tileRows;

    thin::select(x.device());
    auto y = at::empty({1, c1, oh, ow, 16}, x.options());
    GM_ADDR bp = hasBias ? (GM_ADDR)bias->data_ptr() : nullptr;
    auto s = thin::stream();
    if (fused)
        depthwise_kernel<true><<<8, nullptr, s>>>(
            (GM_ADDR)x.data_ptr(), (GM_ADDR)packed.data_ptr(), bp, (GM_ADDR)y.data_ptr(), c1, h, w,
            int(kernel), int(stride), pad, oh, ow, tileRows, numTiles, padW, hasBias);
    else
        depthwise_kernel<false><<<8, nullptr, s>>>(
            (GM_ADDR)x.data_ptr(), (GM_ADDR)packed.data_ptr(), bp, (GM_ADDR)y.data_ptr(), c1, h, w,
            int(kernel), int(stride), pad, oh, ow, tileRows, numTiles, padW, hasBias);
    thin::sync();
    return y;
}
} // namespace depthwise_flow

TORCH_LIBRARY_FRAGMENT(thin_native_flow, m) {
    m.def("depthwise(Tensor x, Tensor packed, Tensor? bias, int kernel, int stride, bool fused=True) -> Tensor");
}
TORCH_LIBRARY_IMPL(thin_native_flow, PrivateUse1, m) {
    m.impl("depthwise", TORCH_FN(depthwise_flow::depthwise));
}
TORCH_LIBRARY_IMPL(thin_native_flow, AutogradPrivateUse1, m) {
    m.impl("depthwise", torch::CppFunction::makeFallthrough());
}
