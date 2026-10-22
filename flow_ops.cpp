#include "kernel_operator.h"
#include <acl/acl_rt.h>
#include <torch/all.h>
#include <torch/library.h>
using namespace AscendC;

namespace thin {
aclrtStream stream();
void sync();
void select(c10::Device);
} // namespace thin

namespace blocked_flow {
constexpr uint32_t C0 = 16;
constexpr uint32_t UB_BYTES = 262144;

__aicore__ inline half neg_inf() {
    union {
        uint32_t bits;
        float value;
    } ninf;
    ninf.bits = 0xff800000u;
    return static_cast<half>(ninf.value);
}

template <typename T>
__global__ __aicore__ void maxpool_kernel(GM_ADDR x, GM_ADDR y, uint32_t c1, uint32_t height, uint32_t width,
                                          uint32_t radius) {
    const uint32_t rowLen = width * C0;
    const uint32_t padLen = (width + 2 * radius) * C0;
    const T NEG = neg_inf();
    const int32_t R = static_cast<int32_t>(radius);
    const int32_t H = static_cast<int32_t>(height);

    TPipe pipe;
    TBuf<TPosition::VECCALC> rowBuf, padBuf, outBuf;
    pipe.InitBuffer(rowBuf, rowLen * sizeof(T));
    pipe.InitBuffer(padBuf, padLen * sizeof(T));
    pipe.InitBuffer(outBuf, rowLen * sizeof(T));
    auto row = rowBuf.Get<T>();
    auto pad = padBuf.Get<T>();
    auto out = outBuf.Get<T>();
    GlobalTensor<T> xg, yg;
    xg.SetGlobalBuffer((__gm__ T *)x);
    yg.SetGlobalBuffer((__gm__ T *)y);

    const uint64_t jobs = uint64_t(c1) * height;
    for (uint64_t job = GetBlockIdx(); job < jobs; job += GetBlockNum()) {
        const uint32_t cb = job / height;
        const int32_t h = static_cast<int32_t>(job % height);
        const uint64_t planeBase = uint64_t(cb) * height * rowLen;
        auto mid = pad[radius * C0];

        Duplicate(pad, NEG, padLen);
        PipeBarrier<PIPE_V>();
        for (int32_t dy = -R; dy <= R; ++dy) {
            const int32_t hh = h + dy;
            if (hh < 0 || hh >= H)
                continue;
            DataCopy(row, xg[planeBase + uint64_t(hh) * rowLen], rowLen);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            Max(mid, mid, row, rowLen);
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }
        PipeBarrier<PIPE_V>();
        Duplicate(out, NEG, rowLen);
        PipeBarrier<PIPE_V>();
        for (int32_t dx = -R; dx <= R; ++dx) {
            Max(out, out, pad[uint32_t(R + dx) * C0], rowLen);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(yg[planeBase + uint64_t(h) * rowLen], out, rowLen);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
}

__global__ __aicore__ void maxpool_tiles(GM_ADDR x, GM_ADDR y, uint32_t c1, uint32_t height,
                                         uint32_t width, uint32_t radius, uint32_t tileRows) {
    const uint32_t pitchPixels = (width + 2 * radius + 15) / 16 * 16;
    const uint32_t pitch = pitchPixels * C0, active = tileRows * pitch;
    TPipe pipe;
    TBuf<TPosition::VECCALC> padBuf, verticalBuf, resultBuf, outBuf;
    pipe.InitBuffer(padBuf, (tileRows + 2 * radius) * pitch * 2);
    pipe.InitBuffer(verticalBuf, (active + 2 * radius * C0) * 2);
    pipe.InitBuffer(resultBuf, active * 2);
    pipe.InitBuffer(outBuf, tileRows * width * C0 * 2);
    auto pad = padBuf.Get<half>(), vertical = verticalBuf.Get<half>();
    auto result = resultBuf.Get<half>(), out = outBuf.Get<half>();
    GlobalTensor<half> input, output;
    input.SetGlobalBuffer((__gm__ half*)x);
    output.SetGlobalBuffer((__gm__ half*)y);
    const uint32_t tiles = (height + tileRows - 1) / tileRows;
    for (uint32_t job = GetBlockIdx(); job < c1 * tiles; job += GetBlockNum()) {
        int row = (job % tiles) * tileRows;
        int rows = height - row < tileRows ? height - row : tileRows;
        int begin = row > radius ? row - radius : 0;
        int end = row + rows + radius < height ? row + rows + radius : height;
        Duplicate(pad, neg_inf(), (tileRows + 2 * radius) * pitch);
        PipeBarrier<PIPE_ALL>();
        DataCopy(pad[(begin - row + radius) * pitch + radius * C0],
                 input[((job / tiles) * height + begin) * width * C0],
                 {uint16_t(end - begin), uint16_t(width), 0, uint16_t(pitchPixels - width)});
        PipeBarrier<PIPE_ALL>();
        Duplicate(vertical, neg_inf(), rows * pitch);
        PipeBarrier<PIPE_V>();
        for (int offset = 0; offset <= 2 * radius; ++offset) {
            Max(vertical, vertical, pad[offset * pitch], rows * pitch);
            PipeBarrier<PIPE_V>();
        }
        Duplicate(result, neg_inf(), rows * pitch);
        PipeBarrier<PIPE_V>();
        for (int offset = 0; offset <= 2 * radius; ++offset) {
            Max(result, result, vertical[offset * C0], rows * pitch);
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(out, result, {uint16_t(rows), uint16_t(width), uint16_t(pitchPixels - width), 0});
        PipeBarrier<PIPE_ALL>();
        DataCopy(output[((job / tiles) * height + row) * width * C0], out, rows * width * C0);
        PipeBarrier<PIPE_ALL>();
    }
}

__global__ __aicore__ void upsample_kernel(GM_ADDR x, GM_ADDR y, uint32_t c1,
                                           uint32_t height, uint32_t width, uint32_t tileRows) {
    const uint32_t inRow = width * C0, outRow = inRow * 2;
    const uint32_t tileElements = tileRows * inRow;
    const uint32_t tilesPerPlane = (height + tileRows - 1) / tileRows;
    TPipe pipe;
    TBuf<TPosition::VECCALC> inBuf, horizontalBuf, expandedBuf;
    pipe.InitBuffer(inBuf, tileElements * sizeof(uint16_t));
    pipe.InitBuffer(horizontalBuf, tileElements * 2 * sizeof(uint16_t));
    pipe.InitBuffer(expandedBuf, tileElements * 4 * sizeof(uint16_t));
    auto in = inBuf.Get<uint16_t>();
    auto horizontal = horizontalBuf.Get<uint16_t>();
    auto expanded = expandedBuf.Get<uint16_t>();
    GlobalTensor<uint16_t> xg, yg;
    xg.SetGlobalBuffer((__gm__ uint16_t *)x);
    yg.SetGlobalBuffer((__gm__ uint16_t *)y);
    const uint64_t jobs = uint64_t(c1) * tilesPerPlane;
    for (uint64_t job = GetBlockIdx(); job < jobs; job += GetBlockNum()) {
        const uint32_t cb = job / tilesPerPlane;
        const uint32_t h = (job % tilesPerPlane) * tileRows;
        const uint32_t rows = height - h < tileRows ? height - h : tileRows;
        const uint32_t elements = rows * inRow;
        const uint64_t inBase = (uint64_t(cb) * height + h) * inRow;
        DataCopy(in, xg[inBase], elements);
        PipeBarrier<PIPE_ALL>();
        DataCopy(horizontal, in, {uint16_t(rows * width), 1, 0, 1});
        DataCopy(horizontal[C0], in, {uint16_t(rows * width), 1, 0, 1});
        PipeBarrier<PIPE_ALL>();
        DataCopy(expanded, horizontal, {uint16_t(rows), uint16_t(width * 2), 0, uint16_t(width * 2)});
        DataCopy(expanded[outRow], horizontal,
                 {uint16_t(rows), uint16_t(width * 2), 0, uint16_t(width * 2)});
        PipeBarrier<PIPE_ALL>();
        DataCopy(yg[inBase * 4], expanded, elements * 4);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__global__ __aicore__ void upsample_row_kernel(GM_ADDR x, GM_ADDR y, uint32_t c1, uint32_t height,
                                           uint32_t width) {
    const uint32_t inRow = width * C0;
    const uint32_t outRow = width * 2 * C0;
    const uint32_t outH = height * 2;

    TPipe pipe;
    TBuf<TPosition::VECCALC> inBuf;
    pipe.InitBuffer(inBuf, inRow * sizeof(T));
    auto in = inBuf.Get<T>();
    GlobalTensor<T> xg, yg;
    xg.SetGlobalBuffer((__gm__ T *)x);
    yg.SetGlobalBuffer((__gm__ T *)y);

    const DataCopyParams dup{static_cast<uint16_t>(width), 1, 0, 1};
    const uint64_t jobs = uint64_t(c1) * height;
    for (uint64_t job = GetBlockIdx(); job < jobs; job += GetBlockNum()) {
        const uint32_t cb = job / height;
        const uint32_t h = job % height;
        const uint64_t inBase = (uint64_t(cb) * height + h) * inRow;
        const uint64_t outBase = (uint64_t(cb) * outH + uint64_t(h) * 2) * outRow;

        DataCopy(in, xg[inBase], inRow);
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
        DataCopy(yg[outBase], in, dup);
        DataCopy(yg[outBase + C0], in, dup);
        DataCopy(yg[outBase + outRow], in, dup);
        DataCopy(yg[outBase + outRow + C0], in, dup);
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }
}

void check_blocked(const at::Tensor &x) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && x.dim() == 5 && x.size(0) == 1 &&
                    x.size(4) == 16 && x.scalar_type() == at::kHalf && x.is_contiguous() &&
                    x.storage_offset() % 16 == 0 && !x.requires_grad(),
                "expected aligned contiguous N=1 FP16 NC1HWC0 input, C0=16");
}

at::Tensor maxpool(const at::Tensor &x, int64_t kernel) {
    check_blocked(x);
    TORCH_CHECK(kernel >= 1 && kernel % 2 == 1, "maxpool kernel must be a positive odd integer");
    const int64_t c1 = x.size(1), h = x.size(2), w = x.size(3);
    TORCH_CHECK(c1 > 0 && h > 0 && w > 0, "blocked spatial operators require positive C1/H/W");
    const int64_t radius = (kernel - 1) / 2;
    const int64_t pad_bytes = (3 * w + 2 * radius) * int64_t(C0) * int64_t(sizeof(at::Half));
    TORCH_CHECK(pad_bytes <= UB_BYTES, "maxpool row tile exceeds unified buffer; width too large");
    thin::select(x.device());
    auto y = at::empty({1, c1, h, w, 16}, x.options());
    auto s = thin::stream();
    const int64_t pitch = (w + 2 * radius + 15) / 16 * 16;
    const int64_t available = UB_BYTES / (C0 * 2) - 2 * radius * (pitch + 1);
    const int64_t rows = std::min(h, available / (3 * pitch + w));
    if (rows > 0 && pitch <= 65535 && rows <= 65535) {
        maxpool_tiles<<<8, nullptr, s>>>((GM_ADDR)x.data_ptr(), (GM_ADDR)y.data_ptr(), uint32_t(c1),
                                        uint32_t(h), uint32_t(w), uint32_t(radius), uint32_t(rows));
    } else {
        maxpool_kernel<half><<<8, nullptr, s>>>((GM_ADDR)x.data_ptr(), (GM_ADDR)y.data_ptr(), uint32_t(c1),
                                             uint32_t(h), uint32_t(w), uint32_t(radius));
    }
    thin::sync();
    return y;
}

at::Tensor upsample(const at::Tensor &x) {
    check_blocked(x);
    const int64_t c1 = x.size(1), h = x.size(2), w = x.size(3);
    TORCH_CHECK(c1 > 0 && h > 0 && w > 0, "blocked spatial operators require positive C1/H/W");
    TORCH_CHECK(w * 2 <= 65535, "upsample width out of DataCopy burst range");
    TORCH_CHECK(w * int64_t(C0) * int64_t(sizeof(at::Half)) <= UB_BYTES,
                "upsample row exceeds unified buffer; width too large");
    const int64_t tile_rows = std::min<int64_t>(h, UB_BYTES / (7 * w * C0 * sizeof(at::Half)));
    thin::select(x.device());
    auto y = at::empty({1, c1, h * 2, w * 2, 16}, x.options());
    auto s = thin::stream();
    if (tile_rows > 0 && tile_rows * w <= 65535)
        upsample_kernel<<<8, nullptr, s>>>((GM_ADDR)x.data_ptr(), (GM_ADDR)y.data_ptr(), uint32_t(c1),
                                             uint32_t(h), uint32_t(w), uint32_t(tile_rows));
    else
        upsample_row_kernel<half><<<8, nullptr, s>>>((GM_ADDR)x.data_ptr(), (GM_ADDR)y.data_ptr(), uint32_t(c1),
                                                    uint32_t(h), uint32_t(w));
    thin::sync();
    return y;
}
} // namespace blocked_flow

TORCH_LIBRARY_FRAGMENT(thin_native_flow, m) {
    m.def("maxpool(Tensor x, int kernel) -> Tensor");
    m.def("upsample(Tensor x) -> Tensor");
}
TORCH_LIBRARY_IMPL(thin_native_flow, PrivateUse1, m) {
    m.impl("maxpool", TORCH_FN(blocked_flow::maxpool));
    m.impl("upsample", TORCH_FN(blocked_flow::upsample));
}
TORCH_LIBRARY_IMPL(thin_native_flow, AutogradPrivateUse1, m) {
    m.impl("maxpool", torch::CppFunction::makeFallthrough());
    m.impl("upsample", torch::CppFunction::makeFallthrough());
}
