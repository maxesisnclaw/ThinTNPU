#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr uint32_t TILE = 2048;

template <typename T, int Op>
__aicore__ inline void binary_impl(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t length, float alpha) {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> left, right;
    TQue<QuePosition::VECOUT, 1> output;
    pipe.InitBuffer(left, 1, TILE * sizeof(T));
    pipe.InitBuffer(right, 1, TILE * sizeof(T));
    pipe.InitBuffer(output, 1, TILE * sizeof(T));
    GlobalTensor<T> xg, yg, zg;
    xg.SetGlobalBuffer((__gm__ T*)x);
    yg.SetGlobalBuffer((__gm__ T*)y);
    zg.SetGlobalBuffer((__gm__ T*)z);
    for (uint64_t offset = GetBlockIdx() * TILE; offset < length; offset += GetBlockNum() * TILE) {
        uint32_t count = length - offset < TILE ? length - offset : TILE;
        constexpr uint32_t align = 32 / sizeof(T);
        uint32_t copied = (count + align - 1) / align * align;
        auto a = left.AllocTensor<T>();
        auto b = right.AllocTensor<T>();
        DataCopy(a, xg[offset], copied);
        DataCopy(b, yg[offset], copied);
        left.EnQue(a); right.EnQue(b);
        a = left.DeQue<T>(); b = right.DeQue<T>();
        auto out = output.AllocTensor<T>();
        if constexpr (Op == 0 || Op == 3) {
            if (alpha != 1.0f) {
                Muls(b, b, static_cast<T>(alpha), copied);
                PipeBarrier<PIPE_V>();
            }
        }
        if constexpr (Op == 0) Sub(out, a, b, copied);
        if constexpr (Op == 1) Mul(out, a, b, copied);
        if constexpr (Op == 2) Div(out, a, b, copied);
        if constexpr (Op == 3) Add(out, a, b, copied);
        output.EnQue(out);
        left.FreeTensor(a); right.FreeTensor(b);
        out = output.DeQue<T>();
        DataCopy(zg[offset], out, copied);
        output.FreeTensor(out);
    }
}

template <typename T>
__aicore__ inline void silu_impl(GM_ADDR x, GM_ADDR y, uint64_t length) {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input;
    TQue<QuePosition::VECOUT, 1> output;
    TBuf<QuePosition::VECCALC> fp_input, fp_output;
    pipe.InitBuffer(input, 1, TILE * sizeof(T));
    pipe.InitBuffer(output, 1, TILE * sizeof(T));
    if constexpr (std::is_same<T, half>::value) {
        pipe.InitBuffer(fp_input, TILE * sizeof(float));
        pipe.InitBuffer(fp_output, TILE * sizeof(float));
    }
    GlobalTensor<T> xg, yg;
    xg.SetGlobalBuffer((__gm__ T*)x);
    yg.SetGlobalBuffer((__gm__ T*)y);
    for (uint64_t offset = GetBlockIdx() * TILE; offset < length; offset += GetBlockNum() * TILE) {
        uint32_t count = length - offset < TILE ? length - offset : TILE;
        constexpr uint32_t align = 32 / sizeof(T);
        uint32_t copied = (count + align - 1) / align * align;
        auto in = input.AllocTensor<T>();
        DataCopy(in, xg[offset], copied);
        input.EnQue(in);
        in = input.DeQue<T>();
        auto out = output.AllocTensor<T>();
        if constexpr (std::is_same<T, half>::value) {
            auto a = fp_input.Get<float>();
            auto b = fp_output.Get<float>();
            uint8_t repeats = (copied + 63) / 64;
            // dav-2002 count-mode Cast drops lanes; repeat-mode keeps all lanes.
            Cast(a, in, RoundMode::CAST_NONE, 64, repeats, {1, 1, 8, 4});
            PipeBarrier<PIPE_V>();
            Muls(b, a, -1.0f, copied);
            PipeBarrier<PIPE_V>();
            Exp(b, b, copied);
            PipeBarrier<PIPE_V>();
            Adds(b, b, 1.0f, copied);
            PipeBarrier<PIPE_V>();
            Div(b, a, b, copied);
            PipeBarrier<PIPE_V>();
            Cast(out, b, RoundMode::CAST_NONE, 64, repeats, {1, 1, 4, 8});
        } else {
            Muls(out, in, static_cast<T>(-1), copied);
            PipeBarrier<PIPE_V>();
            Exp(out, out, copied);
            PipeBarrier<PIPE_V>();
            Adds(out, out, static_cast<T>(1), copied);
            PipeBarrier<PIPE_V>();
            Div(out, in, out, copied);
        }
        output.EnQue(out);
        input.FreeTensor(in);
        out = output.DeQue<T>();
        DataCopy(yg[offset], out, copied);
        output.FreeTensor(out);
    }
}

#define BINARY_ENTRY(name, type, op) \
extern "C" __global__ __aicore__ void name(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t length, float alpha) { \
    binary_impl<type, op>(x, y, z, length, alpha); \
}
BINARY_ENTRY(sub_fp32, float, 0)
BINARY_ENTRY(sub_fp16, half, 0)
BINARY_ENTRY(mul_fp32, float, 1)
BINARY_ENTRY(mul_fp16, half, 1)
BINARY_ENTRY(div_fp32, float, 2)
BINARY_ENTRY(div_fp16, half, 2)
BINARY_ENTRY(add_fp32, float, 3)
BINARY_ENTRY(add_fp16, half, 3)
extern "C" __global__ __aicore__ void silu_fp32(GM_ADDR x, GM_ADDR y, uint64_t length) {
    silu_impl<float>(x, y, length);
}
extern "C" __global__ __aicore__ void silu_fp16(GM_ADDR x, GM_ADDR y, uint64_t length) {
    silu_impl<half>(x, y, length);
}
