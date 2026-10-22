#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t TILE = 128;

template <typename T>
__aicore__ inline void maxpool_impl(GM_ADDR x, GM_ADDR y, uint64_t length,
                                  uint64_t height, uint64_t width) {
    constexpr uint32_t ALIGN = 32 / sizeof(T);
    constexpr T SIGN = sizeof(T) == 2 ? 0x8000u : 0x80000000u;
    constexpr T INF = sizeof(T) == 2 ? 0x7c00u : 0x7f800000u;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input;
    TQue<QuePosition::VECOUT, 1> output;
    pipe.InitBuffer(input, 1, (TILE + 2 * ALIGN) * sizeof(T));
    pipe.InitBuffer(output, 1, TILE * sizeof(T));
    GlobalTensor<T> xg, yg;
    xg.SetGlobalBuffer((__gm__ T*)x);
    yg.SetGlobalBuffer((__gm__ T*)y);
    for (uint64_t offset = GetBlockIdx() * TILE; offset < length;
         offset += GetBlockNum() * TILE) {
        uint32_t count = length - offset < TILE ? length - offset : TILE;
        auto out = output.AllocTensor<T>();
        for (uint32_t i = 0; i < TILE; ++i)
            out.SetValue(i, static_cast<T>(SIGN | INF));
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                int64_t shifted = static_cast<int64_t>(offset) + dy * static_cast<int64_t>(width) + dx;
                int64_t begin = shifted > 0 ? shifted : 0;
                int64_t end = shifted + count;
                if (end > static_cast<int64_t>(length)) end = length;
                if (begin >= end) continue;
                uint64_t aligned = begin / ALIGN * ALIGN;
                uint32_t copied = (end - aligned + ALIGN - 1) / ALIGN * ALIGN;
                auto in = input.AllocTensor<T>();
                DataCopy(in, xg[aligned], copied);
                input.EnQue(in);
                in = input.DeQue<T>();
                for (uint32_t i = 0; i < count; ++i) {
                    uint64_t pos = offset + i;
                    int64_t row = (pos / width) % height;
                    int64_t col = pos % width;
                    if (row + dy < 0 || row + dy >= static_cast<int64_t>(height) ||
                        col + dx < 0 || col + dx >= static_cast<int64_t>(width)) continue;
                    T v = in.GetValue(shifted + i - aligned);
                    T best = out.GetValue(i);
                    // IEEE sign-magnitude ordering avoids unsupported scalar half arithmetic.
                    T vk = (v & SIGN) ? static_cast<T>(~v) : static_cast<T>(v ^ SIGN);
                    T bk = (best & SIGN) ? static_cast<T>(~best) : static_cast<T>(best ^ SIGN);
                    if ((v & ~SIGN) > INF || ((best & ~SIGN) <= INF && vk >= bk))
                        out.SetValue(i, v);
                }
                input.FreeTensor(in);
            }
        }
        output.EnQue(out);
        out = output.DeQue<T>();
        DataCopy(yg[offset], out, (count + ALIGN - 1) / ALIGN * ALIGN);
        output.FreeTensor(out);
    }
}
template <typename T>
__aicore__ inline bool vector_eligible(uint64_t height, uint64_t width) {
    constexpr uint64_t align = 32 / sizeof(T);
    uint64_t a = width, b = align;
    while (b) { uint64_t t = a % b; a = b; b = t; }
    uint64_t row_step = align / a;
    uint64_t pad = (2 * width + align - 1) / align * align;
    return height * width % align == 0 &&
        2 * pad * sizeof(T) + row_step * width * (20 + 4 * sizeof(T)) <= 262144;
}

template <typename T>
__aicore__ inline void vector_planes(GM_ADDR x, GM_ADDR y, uint64_t length,
                                    uint32_t height, uint32_t width) {
    constexpr uint32_t ALIGN = 32 / sizeof(T);
    const uint32_t pad = (2 * width + ALIGN - 1) / ALIGN * ALIGN;
    uint32_t a = width, b = ALIGN;
    while (b) { uint32_t t = a % b; a = b; b = t; }
    const uint32_t row_step = ALIGN / a;
    const uint32_t max_rows = (262144 - 2 * pad * sizeof(T)) / (width * (20 + 4 * sizeof(T)));
    uint32_t rows = max_rows;
    rows = rows / row_step * row_step;
    if (rows > height) rows = height;
    const uint32_t size = rows * width;
    const uint32_t tiles = (height + rows - 1) / rows;
    const uint32_t input_size = size + 2 * pad;
    TPipe pipe;
    TBuf<TPosition::VECCALC> index_buf, maps_buf, temp_buf, middle_buf;
    TQue<QuePosition::VECIN, 1> input;
    TQue<QuePosition::VECOUT, 1> output;
    pipe.InitBuffer(index_buf, size * 4);
    pipe.InitBuffer(maps_buf, 4 * size * 4);
    pipe.InitBuffer(temp_buf, size * sizeof(T));
    pipe.InitBuffer(middle_buf, size * sizeof(T));
    pipe.InitBuffer(input, 1, input_size * sizeof(T));
    pipe.InitBuffer(output, 1, size * sizeof(T));
    auto index = index_buf.Get<int32_t>();
    auto maps = maps_buf.Get<int32_t>();
    auto temp = temp_buf.Get<T>();
    auto middle = middle_buf.Get<T>();
    CreateVecIndex(index, int32_t(0), size);
    PipeBarrier<PIPE_V>();
    Muls(index, index, int32_t(sizeof(T)), size);
    PipeBarrier<PIPE_V>();
    for (uint32_t k = 0; k < 4; ++k) {
        const int32_t dx = k < 2 ? int32_t(k) - 2 : int32_t(k) - 1;
        Adds(maps[k * size], index, dx * int32_t(sizeof(T)), size);
    }
    PipeBarrier<PIPE_ALL>();
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t k = 0; k < 4; ++k) {
            const int32_t dx = k < 2 ? int32_t(k) - 2 : int32_t(k) - 1;
            for (uint32_t col = 0; col < 2; ++col) {
                int32_t c = dx < 0 ? int32_t(col) : int32_t(width) - 1 - int32_t(col);
                if (c < 0 || c >= int32_t(width)) continue;
                int32_t shifted = c + dx;
                if (shifted < 0) shifted = 0;
                if (shifted >= int32_t(width)) shifted = width - 1;
                maps.SetValue(k * size + row * width + c, (row * width + shifted) * sizeof(T));
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<T> xg, yg;
    xg.SetGlobalBuffer((__gm__ T*)x);
    yg.SetGlobalBuffer((__gm__ T*)y);
    union { uint32_t bits; float value; } neg_inf;
    neg_inf.bits = 0xff800000u;
    const uint64_t plane_size = uint64_t(height) * width;
    for (uint64_t job = GetBlockIdx(); job < length / plane_size * tiles; job += GetBlockNum()) {
        const uint64_t plane = job / tiles;
        const uint32_t row = job % tiles * rows;
        const uint32_t valid_rows = height - row < rows ? height - row : rows;
        const uint32_t count = valid_rows * width;
        const int64_t virtual_begin = int64_t(plane * plane_size + uint64_t(row) * width) - pad;
        const uint64_t begin = virtual_begin > int64_t(plane * plane_size) ? virtual_begin : plane * plane_size;
        const uint64_t end = virtual_begin + input_size < int64_t((plane + 1) * plane_size) ? virtual_begin + input_size : (plane + 1) * plane_size;
        auto in = input.AllocTensor<T>();
        Duplicate(in, T(neg_inf.value), input_size);
        PipeBarrier<PIPE_ALL>();
        DataCopy(in[begin - virtual_begin], xg[begin], end - begin);
        input.EnQue(in);
        in = input.DeQue<T>();
        PipeBarrier<PIPE_ALL>();
        DataCopy(middle, in[pad], size);
        PipeBarrier<PIPE_ALL>();
        PipeBarrier<PIPE_V>();
        for (int32_t dy = -2; dy <= 2; ++dy) {
            if (dy == 0) continue;
            if ((dy * int32_t(width)) % int32_t(ALIGN) == 0) {
                Max(middle, middle, in[pad + dy * int32_t(width)], size);
            } else {
                Gather(temp, in, index.ReinterpretCast<uint32_t>(), (pad + dy * int32_t(width)) * sizeof(T), size);
                PipeBarrier<PIPE_V>();
                Max(middle, middle, temp, size);
            }
            PipeBarrier<PIPE_V>();
        }
        auto out = output.AllocTensor<T>();
        PipeBarrier<PIPE_ALL>();
        DataCopy(out, middle, size);
        PipeBarrier<PIPE_ALL>();
        PipeBarrier<PIPE_V>();
        for (uint32_t k = 0; k < 4; ++k) {
            Gather(temp, middle, maps[k * size].ReinterpretCast<uint32_t>(), 0, size);
            PipeBarrier<PIPE_V>();
            Max(out, out, temp, size);
            PipeBarrier<PIPE_V>();
        }
        input.FreeTensor(in);
        output.EnQue(out);
        out = output.DeQue<T>();
        DataCopy(yg[plane * plane_size + uint64_t(row) * width], out, count);
        output.FreeTensor(out);
    }
}

extern "C" __global__ __aicore__ void maxpool_fp32(GM_ADDR x, GM_ADDR y,
        uint64_t length, uint64_t height, uint64_t width) {
    if (vector_eligible<float>(height, width))
        vector_planes<float>(x, y, length, height, width);
    else
        maxpool_impl<uint32_t>(x, y, length, height, width);
}
extern "C" __global__ __aicore__ void maxpool_fp16(GM_ADDR x, GM_ADDR y,
        uint64_t length, uint64_t height, uint64_t width) {
    if (vector_eligible<half>(height, width))
        vector_planes<half>(x, y, length, height, width);
    else
        maxpool_impl<uint16_t>(x, y, length, height, width);
}
