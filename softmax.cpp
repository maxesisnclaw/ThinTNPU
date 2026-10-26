#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr uint32_t TILE = 2048;

template <typename T, typename U, bool Last>
class Softmax {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input;
    TQue<QuePosition::VECOUT, 1> output;
    TBuf<QuePosition::VECCALC> values, maxima, sums, scratch, reduced;
    GlobalTensor<T> x;
    GlobalTensor<U> y;

    __aicore__ inline LocalTensor<float> load(uint64_t offset, uint32_t count) {
        auto in = input.AllocTensor<T>();
        DataCopy(in, x[offset], count);
        input.EnQue(in);
        in = input.DeQue<T>();
        auto a = values.Get<float>();
        if constexpr (std::is_same<T, half>::value) {
            // dav-2002 count-mode Cast loses lanes; repeat-mode preserves them.
            Cast(a, in, RoundMode::CAST_NONE, 64, (count + 63) / 64, {1, 1, 8, 4});
        } else {
            Adds(a, in, 0.0f, count);
        }
        PipeBarrier<PIPE_V>();
        input.FreeTensor(in);
        return a;
    }
    __aicore__ inline void store(uint64_t offset, LocalTensor<float> a, uint32_t count) {
        auto out = output.AllocTensor<U>();
        if constexpr (std::is_same<U, half>::value) {
            Cast(out, a, RoundMode::CAST_NONE, 64, (count + 63) / 64, {1, 1, 4, 8});
            PipeBarrier<PIPE_V>();
            // dav-2002 float-to-half Cast saturates NaN; restore its IEEE encoding.
            auto nan_bits = scratch.Get<uint16_t>();
            auto mask = reduced.Get<uint8_t>();
            Duplicate(nan_bits, static_cast<uint16_t>(0x7e00), count);
            // Compare emits whole 64-float repeats on dav-2002, even for short tiles.
            Compare(mask, a, a, CMPMODE::EQ, (count + 63) / 64 * 64);
            PipeBarrier<PIPE_V>();
            Select(out, mask, out, scratch.Get<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        } else {
            Adds(out, a, 0.0f, count);
        }
        output.EnQue(out);
        out = output.DeQue<U>();
        DataCopy(y[offset], out, count);
        output.FreeTensor(out);
    }
public:
    __aicore__ inline void run(GM_ADDR src, GM_ADDR dst, uint64_t outer, uint64_t reduction, uint64_t inner, uint64_t pitch) {
        pipe.InitBuffer(input, 1, TILE * sizeof(T));
        pipe.InitBuffer(output, 1, TILE * sizeof(U));
        pipe.InitBuffer(values, TILE * sizeof(float));
        pipe.InitBuffer(maxima, TILE * sizeof(float));
        pipe.InitBuffer(sums, TILE * sizeof(float));
        pipe.InitBuffer(scratch, TILE * sizeof(float));
        pipe.InitBuffer(reduced, 64 * sizeof(float));
        x.SetGlobalBuffer((__gm__ T*)src);
        y.SetGlobalBuffer((__gm__ U*)dst);
        auto m = maxima.Get<float>();
        auto s = sums.Get<float>();
        auto tmp = scratch.Get<float>();
        auto scalar = reduced.Get<float>();
        if constexpr (Last) {
            for (uint64_t row = GetBlockIdx(); row < outer; row += GetBlockNum()) {
                if (reduction <= TILE) {
                    uint32_t padded = (reduction + 15) / 16 * 16;
                    auto a = load(row * pitch, padded);
                    ReduceMax(scalar, a, tmp, reduction, false);
                    PipeBarrier<PIPE_ALL>();
                    float maximum = -__builtin_inff();
                    float v = scalar.GetValue(0);
                    maximum = maximum > v ? maximum : v;
                    Adds(a, a, -maximum, padded);
                    PipeBarrier<PIPE_V>();
                    Exp(a, a, padded);
                    PipeBarrier<PIPE_V>();
                    ReduceSum(scalar, a, tmp, reduction);
                    PipeBarrier<PIPE_ALL>();
                    float denominator = 0.0f;
                    denominator += scalar.GetValue(0);
                    Muls(a, a, 1.0f / denominator, padded);
                    PipeBarrier<PIPE_V>();
                    store(row * pitch, a, padded);
                    continue;
                }
                float maximum = -__builtin_inff();
                for (uint64_t k = 0; k < reduction; k += TILE) {
                    uint32_t n = reduction - k < TILE ? reduction - k : TILE;
                    uint32_t padded = (n + 15) / 16 * 16;
                    auto a = load(row * pitch + k, padded);
                    ReduceMax(scalar, a, tmp, n, false);
                    PipeBarrier<PIPE_ALL>();
                    float v = scalar.GetValue(0);
                    maximum = maximum > v ? maximum : v;
                }
                float denominator = 0.0f;
                for (uint64_t k = 0; k < reduction; k += TILE) {
                    uint32_t n = reduction - k < TILE ? reduction - k : TILE;
                    uint32_t padded = (n + 15) / 16 * 16;
                    auto a = load(row * pitch + k, padded);
                    Adds(a, a, -maximum, padded);
                    PipeBarrier<PIPE_V>();
                    Exp(a, a, padded);
                    PipeBarrier<PIPE_V>();
                    ReduceSum(scalar, a, tmp, n);
                    PipeBarrier<PIPE_ALL>();
                    denominator += scalar.GetValue(0);
                }
                for (uint64_t k = 0; k < reduction; k += TILE) {
                    uint32_t n = reduction - k < TILE ? reduction - k : TILE;
                    uint32_t padded = (n + 15) / 16 * 16;
                    auto a = load(row * pitch + k, padded);
                    Adds(a, a, -maximum, padded);
                    PipeBarrier<PIPE_V>();
                    Exp(a, a, padded);
                    PipeBarrier<PIPE_V>();
                    Muls(a, a, 1.0f / denominator, padded);
                    PipeBarrier<PIPE_V>();
                    store(row * pitch + k, a, padded);
                }
            }
        } else {
            uint64_t tiles = (inner + TILE - 1) / TILE;
            for (uint64_t task = GetBlockIdx(); task < outer * tiles; task += GetBlockNum()) {
                uint64_t row = task / tiles, column = (task % tiles) * TILE;
                uint32_t n = inner - column < TILE ? inner - column : TILE;
                uint32_t padded = (n + 15) / 16 * 16;
                Duplicate(m, -__builtin_inff(), padded);
                Duplicate(s, 0.0f, padded);
                PipeBarrier<PIPE_V>();
                for (uint64_t r = 0; r < reduction; ++r) {
                    auto a = load((row * reduction + r) * pitch + column, padded);
                    Max(m, m, a, padded);
                    PipeBarrier<PIPE_V>();
                }
                for (uint64_t r = 0; r < reduction; ++r) {
                    auto a = load((row * reduction + r) * pitch + column, padded);
                    Sub(a, a, m, padded);
                    PipeBarrier<PIPE_V>();
                    Exp(a, a, padded);
                    PipeBarrier<PIPE_V>();
                    Add(s, s, a, padded);
                    PipeBarrier<PIPE_V>();
                }
                for (uint64_t r = 0; r < reduction; ++r) {
                    auto offset = (row * reduction + r) * pitch + column;
                    auto a = load(offset, padded);
                    Sub(a, a, m, padded);
                    PipeBarrier<PIPE_V>();
                    Exp(a, a, padded);
                    PipeBarrier<PIPE_V>();
                    Div(a, a, s, padded);
                    PipeBarrier<PIPE_V>();
                    store(offset, a, padded);
                }
            }
        }
    }
};
#define ENTRY(name, T, U, last) \
extern "C" __global__ __aicore__ void name(GM_ADDR x, GM_ADDR y, uint64_t outer, uint64_t reduction, uint64_t inner, uint64_t pitch) { \
    Softmax<T, U, last> op; op.run(x, y, outer, reduction, inner, pitch); \
}
ENTRY(softmax_last_fp32, float, float, true)
ENTRY(softmax_last_fp16, half, half, true)
ENTRY(softmax_last_fp16_float, half, float, true)
ENTRY(softmax_inner_fp32, float, float, false)
ENTRY(softmax_inner_fp16, half, half, false)
ENTRY(softmax_inner_fp16_float, half, float, false)

// Eight tiles per 8400-anchor box balance DFL over the eight vector cores.
constexpr uint32_t DFL_TILE = 1056;
constexpr uint32_t DFL_REDUCTION = 16;

template <typename T, typename U>
class DflSoftmax {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input;
    TQue<QuePosition::VECOUT, 1> output;
    TBuf<QuePosition::VECCALC> values, maxima, sums, nanBits, nanMask;
    GlobalTensor<T> x;
    GlobalTensor<U> y;
public:
    __aicore__ inline void run(GM_ADDR src, GM_ADDR dst, uint64_t batches, uint64_t width,
                               uint64_t batchStride, uint64_t reduceStride, uint64_t boxStride) {
        pipe.InitBuffer(input, 1, DFL_REDUCTION * DFL_TILE * sizeof(T));
        pipe.InitBuffer(output, 1, DFL_REDUCTION * DFL_TILE * sizeof(U));
        pipe.InitBuffer(values, DFL_REDUCTION * DFL_TILE * sizeof(float));
        pipe.InitBuffer(maxima, DFL_TILE * sizeof(float));
        pipe.InitBuffer(sums, DFL_TILE * sizeof(float));
        pipe.InitBuffer(nanBits, DFL_REDUCTION * DFL_TILE * sizeof(uint16_t));
        pipe.InitBuffer(nanMask, DFL_REDUCTION * DFL_TILE / 8);
        x.SetGlobalBuffer((__gm__ T*)src);
        y.SetGlobalBuffer((__gm__ U*)dst);
        auto a = values.Get<float>();
        auto m = maxima.Get<float>();
        auto s = sums.Get<float>();
        uint64_t tiles = (width + DFL_TILE - 1) / DFL_TILE;
        for (uint64_t task = GetBlockIdx(); task < batches * 4 * tiles; task += GetBlockNum()) {
            uint64_t row = task / tiles, column = (task % tiles) * DFL_TILE;
            uint32_t n = width - column < DFL_TILE ? width - column : DFL_TILE;
            uint32_t count = DFL_REDUCTION * n;
            uint64_t srcOffset = (row / 4) * batchStride + (row % 4) * boxStride + column;
            uint64_t dstOffset = (row / 4) * DFL_REDUCTION * 4 * width + (row % 4) * width + column;
            auto in = input.AllocTensor<T>();
            DataCopy(in, x[srcOffset], {DFL_REDUCTION, static_cast<uint16_t>(n * sizeof(T) / 32),
                     static_cast<uint16_t>((reduceStride - n) * sizeof(T) / 32), 0});
            input.EnQue(in);
            in = input.DeQue<T>();
            if constexpr (std::is_same<T, half>::value) {
                Cast(a, in, RoundMode::CAST_NONE, 64, count / 128, {1, 1, 8, 4});
                Cast(a[count / 2], in[count / 2], RoundMode::CAST_NONE, 64, count / 128, {1, 1, 8, 4});
            } else {
                Adds(a, in, 0.0f, count);
            }
            PipeBarrier<PIPE_V>();
            input.FreeTensor(in);
            Duplicate(m, -__builtin_inff(), n);
            Duplicate(s, 0.0f, n);
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < DFL_REDUCTION; ++r) {
                Max(m, m, a[r * n], n);
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t r = 0; r < DFL_REDUCTION; ++r) {
                Sub(a[r * n], a[r * n], m, n);
                PipeBarrier<PIPE_V>();
                Exp(a[r * n], a[r * n], n);
                PipeBarrier<PIPE_V>();
                Add(s, s, a[r * n], n);
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t r = 0; r < DFL_REDUCTION; ++r) {
                Div(a[r * n], a[r * n], s, n);
                PipeBarrier<PIPE_V>();
            }
            auto out = output.AllocTensor<U>();
            if constexpr (std::is_same<U, half>::value) {
                Cast(out, a, RoundMode::CAST_NONE, 64, count / 128, {1, 1, 4, 8});
                Cast(out[count / 2], a[count / 2], RoundMode::CAST_NONE, 64, count / 128, {1, 1, 4, 8});
                PipeBarrier<PIPE_V>();
                // Match the generic dav-2002 path's IEEE NaN restoration after Cast.
                auto bits = nanBits.Get<uint16_t>();
                auto mask = nanMask.Get<uint8_t>();
                Duplicate(bits, static_cast<uint16_t>(0x7e00), count);
                Compare(mask, a, a, CMPMODE::EQ, count);
                PipeBarrier<PIPE_V>();
                Select(out, mask, out, nanBits.Get<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
            } else {
                Adds(out, a, 0.0f, count);
            }
            output.EnQue(out);
            out = output.DeQue<U>();
            DataCopy(y[dstOffset], out, {DFL_REDUCTION, static_cast<uint16_t>(n * sizeof(U) / 32), 0,
                     static_cast<uint16_t>((4 * width - n) * sizeof(U) / 32)});
            output.FreeTensor(out);
        }
    }
};
#define DFL_ENTRY(name, T, U) \
extern "C" __global__ __aicore__ void name(GM_ADDR x, GM_ADDR y, uint64_t batches, uint64_t width, \
                                          uint64_t batchStride, uint64_t reduceStride, uint64_t boxStride) { \
    DflSoftmax<T, U> op; op.run(x, y, batches, width, batchStride, reduceStride, boxStride); \
}
DFL_ENTRY(softmax_dfl_fp32, float, float)
DFL_ENTRY(softmax_dfl_fp16, half, half)
DFL_ENTRY(softmax_dfl_fp16_float, half, float)
