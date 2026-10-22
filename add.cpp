#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t TILE = 2048;

extern "C" __global__ __aicore__ void add_fp32(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t length) {
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> left, right;
    TQue<QuePosition::VECOUT, 1> output;
    pipe.InitBuffer(left, 1, TILE * sizeof(float));
    pipe.InitBuffer(right, 1, TILE * sizeof(float));
    pipe.InitBuffer(output, 1, TILE * sizeof(float));
    GlobalTensor<float> xg, yg, zg;
    xg.SetGlobalBuffer((__gm__ float*)x);
    yg.SetGlobalBuffer((__gm__ float*)y);
    zg.SetGlobalBuffer((__gm__ float*)z);
    for (uint64_t offset = GetBlockIdx() * TILE; offset < length;
         offset += GetBlockNum() * TILE) {
        uint32_t count = length - offset < TILE ? length - offset : TILE;
        uint32_t copied = (count + 7) / 8 * 8;
        auto a = left.AllocTensor<float>();
        auto b = right.AllocTensor<float>();
        DataCopy(a, xg[offset], copied);
        DataCopy(b, yg[offset], copied);
        left.EnQue(a);
        right.EnQue(b);
        a = left.DeQue<float>();
        b = right.DeQue<float>();
        auto out = output.AllocTensor<float>();
        Add(out, a, b, copied);
        output.EnQue(out);
        left.FreeTensor(a);
        right.FreeTensor(b);
        out = output.DeQue<float>();
        DataCopy(zg[offset], out, copied);
        output.FreeTensor(out);
    }
}
