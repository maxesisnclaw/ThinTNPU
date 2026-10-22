#include "kernel_operator.h"
using namespace AscendC;
// Gather uses an 8-bit repeat count: fp32 must stay within 255 * 64 lanes.
constexpr uint32_t OUTPUT_TILE = 255 * 64;
template <typename T>
__aicore__ inline void upsample_impl(GM_ADDR x, GM_ADDR y, uint64_t length,
                                    uint64_t height, uint64_t width) {
    constexpr uint32_t ALIGN = 32 / sizeof(T);
    TPipe pipe;
    TBuf<QuePosition::VECCALC> input, output, offsets;
    pipe.InitBuffer(input, (OUTPUT_TILE + 2 * ALIGN) * sizeof(T));
    pipe.InitBuffer(output, OUTPUT_TILE * sizeof(T));
    pipe.InitBuffer(offsets, OUTPUT_TILE * sizeof(uint32_t));
    auto in = input.Get<T>();
    auto out = output.Get<T>();
    auto idx = offsets.Get<uint32_t>();
    GlobalTensor<T> xg, yg;
    xg.SetGlobalBuffer((__gm__ T*)x);
    yg.SetGlobalBuffer((__gm__ T*)y);
    if ((width * 2) % ALIGN == 0 && width <= OUTPUT_TILE / 4) {
        uint32_t rows = (OUTPUT_TILE / (width * 4)) * 2;
        uint32_t inputTile = rows * width;
        uint32_t outputCount = inputTile * 2;
        for (uint32_t j = 0; j < 8; ++j) idx.SetValue(j, (j / 2) * sizeof(T));
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        auto signedIdx = offsets.Get<int32_t>();
        for (uint32_t built = 8; built < outputCount;) {
            uint32_t n = outputCount - built < built ? outputCount - built : built;
            Adds(signedIdx[built], signedIdx, int32_t((built / 2) * sizeof(T)), n);
            PipeBarrier<PIPE_V>();
            built += n;
        }
        for (uint64_t start = GetBlockIdx() * inputTile; start < length;
             start += GetBlockNum() * inputTile) {
            uint32_t count = length - start < inputTile ? length - start : inputTile;
            DataCopy(in, xg[start], (count + ALIGN - 1) / ALIGN * ALIGN);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            Gather(out, in, idx, 0, count * 2);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            uint16_t rowBlocks = width * 2 / ALIGN;
            DataCopyParams copy{uint16_t(count / width), rowBlocks, 0, rowBlocks};
            DataCopy(yg[start * 4], out, copy);
            DataCopy(yg[start * 4 + width * 2], out, copy);
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }
        return;
    }
    uint32_t rowGroup = ALIGN;
    while (rowGroup > 1 && (width * (rowGroup / 2)) % ALIGN == 0) rowGroup /= 2;
    uint64_t unit = width * rowGroup * 4;
    uint32_t tile = unit <= OUTPUT_TILE ? (OUTPUT_TILE / unit) * unit : OUTPUT_TILE;
    uint64_t cachedPhase = uint64_t(-1);
    uint64_t cachedBasePhase = uint64_t(-1);
    bool cachedSplit = false;
    for (uint64_t start = GetBlockIdx() * tile; start < length * 4;
         start += GetBlockNum() * tile) {
        uint32_t count = length * 4 - start < tile ? length * 4 - start : tile;
        uint32_t padded = (count + ALIGN - 1) / ALIGN * ALIGN;
        uint64_t first = (start / (width * 4)) * width + (start % (width * 2)) / 2;
        uint64_t lastDst = start + count - 1;
        uint64_t last = (lastDst / (width * 4)) * width + (lastDst % (width * 2)) / 2;
        uint64_t low = first < last ? first : last;
        uint64_t high = first > last ? first : last;
        uint64_t firstBoundary = (start / (width * 2) + 1) * (width * 2);
        if (firstBoundary <= lastDst) {
            uint64_t boundaryLow = (firstBoundary / (width * 4)) * width;
            uint64_t lastBoundary = (lastDst / (width * 2)) * (width * 2);
            uint64_t boundaryHigh = ((lastBoundary - 1) / (width * 4)) * width + width - 1;
            low = low < boundaryLow ? low : boundaryLow;
            high = high > boundaryHigh ? high : boundaryHigh;
        }
        uint64_t base = low / ALIGN * ALIGN;
        uint32_t readCount = (high - base + 1 + ALIGN - 1) / ALIGN * ALIGN;
        bool split = readCount > OUTPUT_TILE;
        uint64_t secondBase = 0;
        uint32_t secondCount = 0;
        if (split) {
            base = first / ALIGN * ALIGN;
            uint64_t rowEnd = (first / width + 1) * width;
            readCount = (rowEnd - base + ALIGN - 1) / ALIGN * ALIGN;
            secondBase = (last / width * width) / ALIGN * ALIGN;
            secondCount = (last - secondBase + 1 + ALIGN - 1) / ALIGN * ALIGN;
        }
        if (cachedPhase != start % (width * 4) || cachedBasePhase != first - base || cachedSplit != split) {
            uint32_t seed = unit <= tile && unit < padded ? unit : padded;
            for (uint32_t j = 0; j < seed; ++j) {
                uint64_t dst = start + j;
                uint64_t src = (dst / (width * 4)) * width + (dst % (width * 2)) / 2;
                uint64_t offset = split && dst >= firstBoundary ? readCount + src - secondBase : src - base;
                idx.SetValue(j, offset * sizeof(T));
            }
            SetFlag<HardEvent::S_V>(0);
            WaitFlag<HardEvent::S_V>(0);
            auto signedIdx = offsets.Get<int32_t>();
            for (uint32_t built = seed; built < padded;) {
                uint32_t n = padded - built < built ? padded - built : built;
                Adds(signedIdx[built], signedIdx, int32_t((built / 4) * sizeof(T)), n);
                PipeBarrier<PIPE_V>();
                built += n;
            }
            cachedSplit = split;
            cachedPhase = start % (width * 4);
            cachedBasePhase = first - base;
        }
        DataCopy(in, xg[base], readCount);
        if (split) DataCopy(in[readCount], xg[secondBase], secondCount);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Gather(out, in, idx, 0, padded);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(yg[start], out, padded);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
    }
}
extern "C" __global__ __aicore__ void upsample_fp32(GM_ADDR x, GM_ADDR y, uint64_t length,
                                                 uint64_t height, uint64_t width) {
    upsample_impl<uint32_t>(x, y, length, height, width);
}
extern "C" __global__ __aicore__ void upsample_fp16(GM_ADDR x, GM_ADDR y, uint64_t length,
                                                 uint64_t height, uint64_t width) {
    upsample_impl<uint16_t>(x, y, length, height, width);
}
