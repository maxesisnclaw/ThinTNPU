#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t TILE = 2048;

template <typename T>
__aicore__ inline void sigmoid_impl(GM_ADDR x, GM_ADDR y, uint64_t length) {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input;
    TQue<QuePosition::VECOUT, 1> output;
    TBuf<QuePosition::VECCALC> ones;
    pipe.InitBuffer(input, 1, TILE * sizeof(T));
    pipe.InitBuffer(output, 1, TILE * sizeof(T));
    pipe.InitBuffer(ones, TILE * sizeof(T));
    GlobalTensor<T> xg, yg;
    xg.SetGlobalBuffer((__gm__ T*)x);
    yg.SetGlobalBuffer((__gm__ T*)y);
    auto one = ones.Get<T>();
    Duplicate(one, static_cast<T>(1), TILE);
    for (uint64_t offset = GetBlockIdx() * TILE; offset < length;
         offset += GetBlockNum() * TILE) {
        uint32_t count = length - offset < TILE ? length - offset : TILE;
        constexpr uint32_t align = 32 / sizeof(T);
        uint32_t copied = (count + align - 1) / align * align;
        auto in = input.AllocTensor<T>();
        DataCopy(in, xg[offset], copied);
        input.EnQue(in);
        in = input.DeQue<T>();
        auto out = output.AllocTensor<T>();
        Muls(out, in, static_cast<T>(-1), copied);
        PipeBarrier<PIPE_V>();
        Exp(out, out, copied);
        PipeBarrier<PIPE_V>();
        Adds(out, out, static_cast<T>(1), copied);
        PipeBarrier<PIPE_V>();
        Div(out, one, out, copied);
        output.EnQue(out);
        input.FreeTensor(in);
        out = output.DeQue<T>();
        // Storage owns a complete 32-byte tail block, not just logical elements.
        DataCopy(yg[offset], out, copied);
        output.FreeTensor(out);
    }
}

extern "C" __global__ __aicore__ void sigmoid_fp32(GM_ADDR x, GM_ADDR y, uint64_t length) {
    sigmoid_impl<float>(x, y, length);
}
extern "C" __global__ __aicore__ void sigmoid_fp16(GM_ADDR x, GM_ADDR y, uint64_t length) {
    sigmoid_impl<half>(x, y, length);
}
