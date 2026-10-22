#include "cube_layout.h"
#include "kernel_operator.h"
#include <acl/acl_rt.h>
#include <torch/all.h>
#include <limits>
#include <numeric>
#include <cmath>
#include <torch/library.h>
using namespace AscendC;
namespace thin {
aclrtStream stream();
void sync();
void select(c10::Device);
} // namespace thin
namespace native_flow {
__aicore__ inline int imin(int a, int b) { return a < b ? a : b; }

__aicore__ inline void stage_native(LocalTensor<half> image, GlobalTensor<half> input,
                                    int cb, int bank, int ci, int maxP, int n, int pos,
                                    int count, int plane, int chunk) {
    int channels = imin(chunk, ci - cb);
    if (n - count > 65535) {
        for (int c = 0; c < channels; c += 16)
            DataCopy(image[bank * chunk * maxP + c * plane], input[(cb + c) * n + pos * 16], count * 16);
    } else {
        DataCopy(image[bank * chunk * maxP], input[cb * n + pos * 16],
                 DataCopyParams{(uint16_t)(channels / 16), (uint16_t)count, (uint16_t)(n - count),
                                (uint16_t)(plane - count)});
    }
    SetFlag<HardEvent::MTE2_MTE1>(bank);
}

__aicore__ inline void stage_tile(LocalTensor<half> image, GlobalTensor<half> input,
                                  int tile, int bank, int ci, int h, int width,
                                  int stride, int oh, int ow, int maxP, int TM) {
    int pos = tile * TM, count = imin(TM, oh * ow - pos);
    int row = pos / ow, lastRow = (pos + count - 1) / ow;
    int begin = (row * stride - 1 > 0 ? row * stride - 1 : 0) / stride * stride;
    int end = imin(h, (lastRow * stride + 2 + stride - 1) / stride * stride);
    stage_native(image, input, 0, bank, ci, maxP, h * width, begin * width,
                 (end - begin) * width, (end - begin) * width, ci);
}

template <bool Fused>
__aicore__ inline void begin_epilogue(LocalTensor<float> value, LocalTensor<float> scratch,
                                      LocalTensor<float> bias, LocalTensor<float> accum, int count, int bank,
                                      bool hasBias, int accumStride, bool release, int TN) {
    DataCopyEnhancedParams ep;
    ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(value, accum, DataCopyParams{(uint16_t)(TN / 16), (uint16_t)(count / 16), (uint16_t)((accumStride - count) / 16), 0}, ep);
    PipeBarrier<PIPE_V>();
    if (release) SetFlag<HardEvent::V_M>(bank);
    if (hasBias) {
        image_layout::add_bias(value, bias, count, TN);
    }
    if constexpr (Fused) {
        SetVectorMask<float>(0, uint64_t(-1));
        for (int offset = 0; offset < count * TN; offset += 255 * 64) {
            int repeats = imin(255, (count * TN - offset) / 64);
            Muls<float, false>(scratch[offset], value[offset], -1.0f, 64, repeats, {1, 1, 8, 8});
        }
        PipeBarrier<PIPE_V>();
        for (int offset = 0; offset < count * TN; offset += 255 * 64) {
            int repeats = imin(255, (count * TN - offset) / 64);
            Exp<float, false>(scratch[offset], scratch[offset], 64, repeats, {1, 1, 8, 8});
        }
    }
}

template <bool Fused>
__aicore__ inline void divide_epilogue(LocalTensor<float> value, LocalTensor<float> scratch, int count, int TN) {
    if constexpr (Fused) {
        PipeBarrier<PIPE_V>();
        for (int offset = 0; offset < count * TN; offset += 255 * 64) {
            int repeats = imin(255, (count * TN - offset) / 64);
            Adds<float, false>(scratch[offset], scratch[offset], 1.0f, 64, repeats, {1, 1, 8, 8});
        }
        PipeBarrier<PIPE_V>();
        for (int offset = 0; offset < count * TN; offset += 255 * 64) {
            int repeats = imin(255, (count * TN - offset) / 64);
            Div<float, false>(value[offset], value[offset], scratch[offset], 64, repeats, {1, 1, 1, 8, 8, 8});
        }
    }
}

template <bool Nchw>
__aicore__ inline void end_epilogue(LocalTensor<float> value, LocalTensor<half> halves, int count, int TN) {
    PipeBarrier<PIPE_V>();
    Cast(halves, value, RoundMode::CAST_NONE, count * TN);
    PipeBarrier<PIPE_V>();
    if constexpr (Nchw) {
        image_layout::transpose_half(value.ReinterpretCast<half>(), halves, count, TN);
        PipeBarrier<PIPE_V>();
    }
}

template <bool Nchw>
__aicore__ inline void store(LocalTensor<float> value, LocalTensor<half> halves, GlobalTensor<half> outH,
                             int oc, int outputs, int n, int pos, int count, int bank) {
    SetFlag<HardEvent::V_MTE3>(2 + bank);
    WaitFlag<HardEvent::V_MTE3>(2 + bank);
    if constexpr (Nchw) {
        if ((n - count) / 16 > 65535) {
            for (int channel = 0; channel < outputs; ++channel)
                DataCopy(outH[(oc + channel) * n + pos], value.ReinterpretCast<half>()[channel * count], count);
        } else {
            DataCopy(outH[oc * n + pos], value.ReinterpretCast<half>(),
                     DataCopyParams{(uint16_t)outputs, (uint16_t)(count / 16), 0, (uint16_t)((n - count) / 16)});
        }
    } else {
        if (n - count > 65535) {
            for (int channel = 0; channel < outputs; channel += 16)
                DataCopy(outH[(oc + channel) * n + pos * 16], halves[channel * count], count * 16);
        } else {
            DataCopy(outH[oc * n + pos * 16], halves,
                     DataCopyParams{(uint16_t)(outputs / 16), (uint16_t)count, 0, (uint16_t)(n - count)});
        }
    }
    SetFlag<HardEvent::MTE3_V>(2 + bank);
}

template <int KERNEL, bool Nchw, bool Fused>
__global__ __aicore__ void convolution_native(GM_ADDR x, GM_ADDR pw, GM_ADDR bp, GM_ADDR y, int ci, int co, int h,
                                       int width, int stride, int oh, int ow, int maxP, bool hasBias,
                                       int TM, int imageChannels, int imageBanks, int channelCores, int weightGroups, int dstPlane, int TN, int TK) {
    constexpr int banks = 2;
    int EM = imin(TM, (262144 - weightGroups * TN * 22) / (16 * TN) / 16 * 16);
    bool cachePoint = KERNEL == 1 && ci * h * width * 2 + TN * ci * 2 <= 1048576;
    bool tileBuffered = KERNEL == 3 && imageChannels == ci && imageBanks == 2;
    TPipe pipe;
    TBuf<TPosition::A1> imageBuf;
    TBuf<TPosition::B1> weightBuf;
    TBuf<TPosition::A2> aBuf;
    TBuf<TPosition::B2> bBuf;
    TBuf<TPosition::CO1> cBuf;
    TBuf<TPosition::VECCALC> valueBuf, scratchBuf, biasBuf, outputBuf, broadcastBuf;
    pipe.InitBuffer(imageBuf, cachePoint ? ci * h * width * 2
                                         : imageBanks * imageChannels * maxP * 2);
    pipe.InitBuffer(weightBuf, weightGroups * TN * ci * KERNEL * KERNEL * 2);
    pipe.InitBuffer(aBuf, banks * TM * TK * 2);
    pipe.InitBuffer(bBuf, KERNEL == 1 ? TN * ci * 2 : banks * TN * TK * 2);
    pipe.InitBuffer(cBuf, 2 * TN * TM * 4);
    pipe.InitBuffer(valueBuf, 2 * TN * EM * 4);
    pipe.InitBuffer(scratchBuf, TN * EM * 4);
    pipe.InitBuffer(biasBuf, weightGroups * TN * 6);
    pipe.InitBuffer(broadcastBuf, weightGroups * TN * 16);
    pipe.InitBuffer(outputBuf, 2 * TN * EM * 2);
    GlobalTensor<half> input, weights, biases, outH;
    input.SetGlobalBuffer((__gm__ half *)x);
    weights.SetGlobalBuffer((__gm__ half *)pw);
    biases.SetGlobalBuffer((__gm__ half *)bp);
    outH.SetGlobalBuffer((__gm__ half *)y);
    auto image = imageBuf.Get<half>(), weight = weightBuf.Get<half>();
    auto value = valueBuf.Get<float>(), scratch = scratchBuf.Get<float>();
    auto halves = outputBuf.Get<half>();
    int n = oh * ow, kk = KERNEL * KERNEL;
    int spatialCores = GetBlockNum() / channelCores;
    int tiles = (n + TM - 1) / TM;
    if (cachePoint) {
        DataCopy(image, input, ci * h * width);
        SetFlag<HardEvent::MTE2_MTE1>(3);
        WaitFlag<HardEvent::MTE2_MTE1>(3);
    }
    for (int ocBase = GetBlockIdx() % channelCores * TN; ocBase < co;
         ocBase += channelCores * TN * weightGroups) {
        for (int ng = 0; ng < weightGroups; ++ng) {
            int oc = ocBase + ng * channelCores * TN;
            if (oc >= co) break;
            DataCopy(
                weight[ng * TN * ci * kk], weights[oc * 16],
                DataCopyParams{(uint16_t)(ci / 16 * kk), (uint16_t)TN, (uint16_t)((co + 63) / 64 * 64 - TN), 0});
            SetFlag<HardEvent::MTE2_MTE1>(2);
            WaitFlag<HardEvent::MTE2_MTE1>(2);
            if constexpr (KERNEL == 1) {
                LoadData2DParams ld;
                ld.repeatTimes = ci / 16 * TN / 16;
                ld.srcStride = 1;
                LoadData(bBuf.Get<half>(), weight, ld);
                SetFlag<HardEvent::MTE1_M>(2);
                WaitFlag<HardEvent::MTE1_M>(2);
            }
            if (hasBias) {
                DataCopy(biasBuf.Get<half>()[ng * TN * 3 + TN * 2], biases[oc], TN);
                SetFlag<HardEvent::MTE2_V>(2);
                WaitFlag<HardEvent::MTE2_V>(2);
                Cast(biasBuf.Get<float>()[ng * TN * 3 / 2], biasBuf.Get<half>()[ng * TN * 3 + TN * 2], RoundMode::CAST_NONE, TN);
                PipeBarrier<PIPE_ALL>();
                for (int group = 0; group < TN; group += 16)
                    for (int repeat = 0; repeat < 4; ++repeat)
                        DataCopy(broadcastBuf.Get<float>()[ng * TN * 4 + group * 4 + repeat * 16], biasBuf.Get<float>()[ng * TN * 3 / 2 + group],
                                 16);
                PipeBarrier<PIPE_ALL>();
            }
        }
        for (int bank = 0; bank < banks; ++bank)
            SetFlag<HardEvent::M_MTE1>(bank);
        for (int bank = 0; bank < 2; ++bank)
            SetFlag<HardEvent::V_M>(bank);
        SetFlag<HardEvent::MTE3_V>(2);
        SetFlag<HardEvent::MTE3_V>(3);
        int pendingPos = -1, pendingCount = 0, pendingBank = 0, pendingOc = 0, pendingGroup = 0;
        if (tileBuffered && GetBlockIdx() / channelCores < tiles)
            stage_tile(image, input, GetBlockIdx() / channelCores, 0,
                           ci, h, width, stride, oh, ow, maxP, TM);
        int spatialId = GetBlockIdx() / channelCores;
        int spatialWork = (tiles + spatialCores - 1 - spatialId) / spatialCores;
        for (int work = 0; work < spatialWork * weightGroups; ++work) {
            int ng = work % weightGroups;
            int tile = spatialId + work / weightGroups * spatialCores;
            int oc = ocBase + ng * channelCores * TN;
            if (oc >= co) continue;
            int cBank = work % 2;
            auto accum = cBuf.Get<float>()[cBank * TN * TM];
            int epOffset = 0, epBank = 0, epCount = imin(EM, pendingCount);
            if (pendingPos >= 0) {
                WaitFlag<HardEvent::M_V>(pendingBank);
                WaitFlag<HardEvent::MTE3_V>(2);
                begin_epilogue<Fused>(value, scratch, broadcastBuf.Get<float>()[pendingGroup * TN * 4],
                    cBuf.Get<float>()[pendingBank * TN * TM], epCount, pendingBank,
                    hasBias, pendingCount, epCount == pendingCount, TN);
            }
            WaitFlag<HardEvent::V_M>(cBank);
            int reductionSteps = ci * kk / TK, completedSteps = 0;
            int pos = tile * TM, count = (imin(TM, n - pos) + 15) / 16 * 16;
            int begin = 0, end = 1, length = TM;
            if constexpr (KERNEL == 3) {
                int row = pos / ow, lastRow = (pos + count - 1) / ow;
                int group = stride;
                begin = (row * stride - 1 > 0 ? row * stride - 1 : 0) / group * group;
                end = imin(h, (lastRow * stride + 2 + group - 1) / group * group);
                length = (end - begin) * width;
            }
            int sourcePos = KERNEL == 1 ? pos : begin * width;
            int copied = KERNEL == 1 ? imin(count, n - pos) : length;
            int plane = KERNEL == 1 ? TM : length;
            if (!cachePoint && !tileBuffered && (imageBanks == 2 || ci <= imageChannels) && (ng == 0 || imageChannels < ci))
                stage_native(image, input, 0, 0, ci, maxP, h * width, sourcePos, copied, plane, imageChannels);
            for (int cb = 0; cb < ci; cb += imageChannels) {
                int channels = imin(imageChannels, ci - cb);
                int imageBank = tileBuffered ? (tile / spatialCores) % 2 : cb / imageChannels % imageBanks;
                if (!cachePoint && imageBanks == 1 && ci > imageChannels)
                    stage_native(image, input, cb, 0, ci, maxP, h * width, sourcePos, copied, plane, imageChannels);
                if (!cachePoint && (ng == 0 || imageChannels < ci))
                    WaitFlag<HardEvent::MTE2_MTE1>(imageBank);
                for (int k = 0; k < channels * kk; k += TK) {
                    int bank = (cb / imageChannels * (imageChannels * kk / TK) + k / TK) % banks;
                    auto a = aBuf.Get<half>()[bank * TM * TK];
                    auto b = bBuf.Get<half>()[KERNEL == 1 ? (cb + k) * TN : bank * TN * TK];
                    WaitFlag<HardEvent::M_MTE1>(bank);
                    LoadData3DParamsV2<half> p;
                    p.l1H = KERNEL == 1 ? 1 : end - begin;
                    p.l1W = KERNEL == 1 ? (cachePoint ? n : TM) : width;
                    p.channelSize = channels;
                    p.kExtension = TK;
                    p.kStartPt = k;
                    p.mExtension = count;
                    p.mStartPt =
                        KERNEL == 1 ? (cachePoint ? pos : 0) : (pos / ow - begin / stride) * ow + pos % ow;
                    p.filterW = KERNEL;
                    p.filterH = KERNEL;
                    p.strideW = stride;
                    p.strideH = stride;
                    if constexpr (KERNEL == 3)
                        for (int i = 0; i < 4; ++i)
                            p.padList[i] = 1;
                    LoadData(a, image[cachePoint ? cb * n : imageBank * imageChannels * maxP], p);
                    if constexpr (KERNEL != 1) {
                        LoadData2DParams ld;
                        ld.repeatTimes = TK / 16 * TN / 16;
                        ld.srcStride = 1;
                        LoadData(b, weight[(ng * ci * kk + cb * kk + k) * TN], ld);
                    }
                    SetFlag<HardEvent::MTE1_M>(bank);
                    WaitFlag<HardEvent::MTE1_M>(bank);
                    MmadParams mm;
                    mm.m = count;
                    mm.n = TN;
                    mm.k = TK;
                    mm.isBias = cb != 0 || k != 0;
                    Mmad(accum, a, b, mm);
                    SetFlag<HardEvent::M_MTE1>(bank);
                    ++completedSteps;
                    if (pendingPos >= 0 && epOffset < pendingCount &&
                        completedSteps * ((pendingCount + EM - 1) / EM) >=
                            reductionSteps * (epOffset / EM + 1)) {
                        auto epValue = value[epBank * TN * EM];
                        auto epHalves = halves[epBank * TN * EM];
                        divide_epilogue<Fused>(epValue, scratch, epCount, TN);
                        end_epilogue<Nchw>(epValue, epHalves, epCount, TN);
                        store<Nchw>(epValue, epHalves, outH, pendingOc, imin(TN, co - pendingOc),
                                    dstPlane, pendingPos + epOffset, epCount, epBank);
                        epOffset += epCount;
                        epBank ^= 1;
                        if (epOffset < pendingCount) {
                            epCount = imin(EM, pendingCount - epOffset);
                            WaitFlag<HardEvent::MTE3_V>(2 + epBank);
                            begin_epilogue<Fused>(value[epBank * TN * EM], scratch,
                                broadcastBuf.Get<float>()[pendingGroup * TN * 4],
                                cBuf.Get<float>()[pendingBank * TN * TM + epOffset * 16],
                                epCount, pendingBank, hasBias, pendingCount,
                                epOffset + epCount == pendingCount, TN);
                        }
                    }
                    if (tileBuffered && ng == 0 && k == 0 && tile + spatialCores < tiles)
                        stage_tile(image, input, tile + spatialCores, imageBank ^ 1,
                                   ci, h, width, stride, oh, ow, maxP, TM);
                    if (!cachePoint && imageBanks == 2 && k == 0 && cb + imageChannels < ci) {
                        SetFlag<HardEvent::MTE1_MTE2>(2);
                        WaitFlag<HardEvent::MTE1_MTE2>(2);
                        stage_native(image, input, cb + imageChannels, imageBank ^ 1, ci, maxP, h * width,
                                     sourcePos, copied, plane, imageChannels);
                    }
                }
                if (!cachePoint) {
                    SetFlag<HardEvent::MTE1_MTE2>(2);
                    WaitFlag<HardEvent::MTE1_MTE2>(2);
                }
            }
            SetFlag<HardEvent::M_V>(cBank);
            while (pendingPos >= 0 && epOffset < pendingCount) {
                auto epValue = value[epBank * TN * EM];
                auto epHalves = halves[epBank * TN * EM];
                divide_epilogue<Fused>(epValue, scratch, epCount, TN);
                end_epilogue<Nchw>(epValue, epHalves, epCount, TN);
                store<Nchw>(epValue, epHalves, outH, pendingOc, imin(TN, co - pendingOc),
                            dstPlane, pendingPos + epOffset, epCount, epBank);
                epOffset += epCount;
                epBank ^= 1;
                if (epOffset < pendingCount) {
                    epCount = imin(EM, pendingCount - epOffset);
                    WaitFlag<HardEvent::MTE3_V>(2 + epBank);
                    begin_epilogue<Fused>(value[epBank * TN * EM], scratch,
                        broadcastBuf.Get<float>()[pendingGroup * TN * 4],
                        cBuf.Get<float>()[pendingBank * TN * TM + epOffset * 16],
                        epCount, pendingBank, hasBias, pendingCount,
                        epOffset + epCount == pendingCount, TN);
                }
            }
            pendingPos = pos;
            pendingCount = count;
            pendingBank = cBank;
            pendingOc = oc;
            pendingGroup = ng;
        }
        if (pendingPos >= 0) {
            WaitFlag<HardEvent::M_V>(pendingBank);
            for (int offset = 0; offset < pendingCount; offset += EM) {
                int bank = offset / EM % 2, count = imin(EM, pendingCount - offset);
                auto epValue = value[bank * TN * EM];
                auto epHalves = halves[bank * TN * EM];
                WaitFlag<HardEvent::MTE3_V>(2 + bank);
                begin_epilogue<Fused>(epValue, scratch, broadcastBuf.Get<float>()[pendingGroup * TN * 4],
                    cBuf.Get<float>()[pendingBank * TN * TM + offset * 16], count,
                    pendingBank, hasBias, pendingCount, offset + count == pendingCount, TN);
                divide_epilogue<Fused>(epValue, scratch, count, TN);
                end_epilogue<Nchw>(epValue, epHalves, count, TN);
                store<Nchw>(epValue, epHalves, outH, pendingOc, imin(TN, co - pendingOc), dstPlane,
                            pendingPos + offset, count, bank);
            }
        }
        for (int bank = 0; bank < banks; ++bank)
            WaitFlag<HardEvent::M_MTE1>(bank);
        for (int bank = 0; bank < 2; ++bank)
            WaitFlag<HardEvent::V_M>(bank);
        WaitFlag<HardEvent::MTE3_V>(2);
        WaitFlag<HardEvent::MTE3_V>(3);
        PipeBarrier<PIPE_ALL>();
    }
}


__global__ __aicore__ void trim_output(GM_ADDR src, GM_ADDR dst, int channels, int n, int np, bool nchw) {
    constexpr int tile = 1024;
    TPipe pipe;
    TBuf<TPosition::VECCALC> rawBuf, outBuf;
    pipe.InitBuffer(rawBuf, (tile + 32) * 2);
    pipe.InitBuffer(outBuf, tile * 2);
    auto raw = rawBuf.Get<uint16_t>(), out = outBuf.Get<uint16_t>();
    GlobalTensor<uint16_t> input, output;
    input.SetGlobalBuffer((__gm__ uint16_t*)src);
    output.SetGlobalBuffer((__gm__ uint16_t*)dst);
    int total = channels * n;
    int run = nchw ? n : n * 16, pitch = nchw ? np : np * 16;
    for (int base = GetBlockIdx() * tile; base < total; base += GetBlockNum() * tile) {
        int count = imin(tile, total - base);
        for (int i = 0; i < count;) {
            int index = base + i, pos = index % run;
            int length = imin(run - pos, count - i);
            int source = index / run * pitch + pos, aligned = source / 16 * 16;
            int skew = source - aligned;
            DataCopy(raw, input[aligned], (length + skew + 15) / 16 * 16);
            PipeBarrier<PIPE_ALL>();
            for (int j = 0; j < length; ++j) out.SetValue(i + j, raw.GetValue(skew + j));
            i += length;
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(output[base], out, (count + 15) / 16 * 16);
        PipeBarrier<PIPE_ALL>();
    }
}


struct ConvPlan {
    int m, n = 32, k = 16, channelCores = 1, inputChannels, inputBanks, maxP, epilogueM, weightGroups = 1;
    int64_t ubBytes, l1Bytes;
};

bool fill_plan(ConvPlan &plan, int ci, int co, int h, int w, int kernel, int stride) {
    constexpr int64_t l1Capacity = 1048576, ubCapacity = 262144;
    int oh = (h - 1) / stride + 1, ow = (w - 1) / stride + 1;
    int groups = (co + plan.n - 1) / plan.n;
    int64_t fullInput = int64_t(ci) * h * w * 2;
    int rows = (plan.m + ow - 2) / ow + 1;
    plan.maxP = kernel == 1 ? plan.m : std::min(h, (rows - 1) * stride + 3 + 2 * (stride - 1)) * w;
    int64_t weights = int64_t(plan.n) * ci * kernel * kernel * 2;
    int assignedGroups = (groups + plan.channelCores - 1) / plan.channelCores;
    if (kernel == 3 && plan.channelCores == 1 && fullInput > l1Capacity &&
        assignedGroups * weights + int64_t(ci) * plan.maxP * 2 <= l1Capacity)
        plan.weightGroups = assignedGroups;
    weights *= plan.weightGroups;
    int64_t remaining = l1Capacity - weights;
    if (remaining <= 0) return false;
    if (kernel == 1 && fullInput + weights <= l1Capacity) {
        plan.inputChannels = ci;
        plan.inputBanks = 1;
        plan.l1Bytes = fullInput + weights;
    } else {
        plan.inputBanks = plan.weightGroups > 1
            ? (remaining >= 4 * int64_t(plan.maxP) * ci ? 2 : 1)
            : (remaining >= 4 * int64_t(plan.maxP) * 16 ? 2 : 1);
        int quantum = std::lcm(16, plan.k / std::gcd(plan.k, kernel * kernel));
        int capacityChannels = remaining / (plan.inputBanks * int64_t(plan.maxP) * 2) / quantum * quantum;
        if (capacityChannels < quantum) return false;
        plan.inputChannels = std::min(ci, capacityChannels);
        plan.l1Bytes = weights + int64_t(plan.inputBanks) * plan.inputChannels * plan.maxP * 2;
    }
    plan.epilogueM = std::min<int64_t>(plan.m, (ubCapacity - plan.weightGroups * plan.n * 22) / (16 * plan.n) / 16 * 16);
    plan.ubBytes = int64_t(plan.n) * plan.epilogueM * 16 + plan.weightGroups * plan.n * 22;
    return plan.epilogueM >= 16 && plan.ubBytes <= ubCapacity;
}

double plan_cost(const ConvPlan &p, int ci, int co, int h, int w, int kernel, int stride) {
    int spatial = ((h - 1) / stride + 1) * ((w - 1) / stride + 1);
    int groups = (co + p.n - 1) / p.n, spatialCores = 8 / p.channelCores;
    int tiles = (spatial + p.m - 1) / p.m;
    double rounds = (tiles + spatialCores - 1) / spatialCores;
    double channelRounds = (groups + p.channelCores - 1) / p.channelCores;
    double work = rounds * channelRounds, pixels = std::min(p.m, spatial);
    double reduction = ci * kernel * kernel;
    // Ranking coefficients calibrated on 310P3 in one CANN/container environment;
    // recalibrate for different hardware, drivers or runtime environments.
    constexpr double reductionStepUs = 0.1240, cubeMillionMacUs = 0.1850;
    constexpr double imageLoad100kUs = 0.7718, weightLoad100kUs = 2.1483;
    constexpr double epilogue10kUs = 3.1610, imageDma100kUs = 1.5406, weightDma100kUs = 4.8369;
    return reductionStepUs * work * reduction / p.k
         + cubeMillionMacUs * work * pixels * p.n * reduction / 1e6
         + imageLoad100kUs * work * pixels * reduction / 1e5
         + weightLoad100kUs * work * p.n * reduction / 1e5
         + epilogue10kUs * work * p.n * pixels / 1e4
         + imageDma100kUs * work * ci * p.maxP / 1e5
         + weightDma100kUs * channelRounds * p.n * reduction / 1e5;
}

ConvPlan make_plan(int ci, int co, int h, int w, int kernel, int stride) {
    TORCH_CHECK(ci > 0 && ci % 16 == 0 && co > 0 && h > 0 && w > 0,
                "expected positive shape and C0-aligned input channels");
    TORCH_CHECK((kernel == 1 && stride == 1) || (kernel == 3 && (stride == 1 || stride == 2)),
                "expected point stride1 or spatial stride1/2");
    if (kernel == 1) {
        ConvPlan p;
        TORCH_CHECK(int64_t(ci) * p.n * 2 <= 65536, "pointwise resident weights exceed L0B");
        int groups = (co + p.n - 1) / p.n;
        while (p.channelCores * 2 <= groups && p.channelCores * 2 <= 8) p.channelCores *= 2;
        int perCore = (h * w + 8 / p.channelCores - 1) / (8 / p.channelCores);
        p.m = std::min(1024, (perCore + 15) / 16 * 16);
        TORCH_CHECK(fill_plan(p, ci, co, h, w, kernel, stride), "pointwise strip exceeds hardware buffers");
        return p;
    }
    ConvPlan best;
    double bestCost = std::numeric_limits<double>::infinity();
    for (int tn : {32, 64, 128}) {
        // Bias is padded to 32, not to TN; wider tiles need complete output groups.
        if (tn != 32 && co % tn) continue;
        int groups = (co + tn - 1) / tn, channelCores = 1;
        while (channelCores * 2 <= groups && channelCores * 2 <= 8) channelCores *= 2;
        if (groups <= 2 && int64_t(ci) * h * w * 2 > 1048576) channelCores = 1;
        for (int tk = 16; tk <= 65536 / (4 * tn) && tk <= ci * 9; tk += 16) {
            // LoadData consumes full TK steps; the L1 chunk quantum preserves this divisibility.
            if (ci * 9 % tk) continue;
            int maxM = std::min(65536 / (4 * tk), 262144 / (8 * tn)) / 16 * 16;
            int spatial = ((h - 1) / stride + 1) * ((w - 1) / stride + 1);
            for (int m = 16; m <= maxM && m <= (spatial + 15) / 16 * 16; m += 16) {
                ConvPlan p;
                p.m = m; p.n = tn; p.k = tk; p.channelCores = channelCores;
                if (!fill_plan(p, ci, co, h, w, kernel, stride)) continue;
                double cost = plan_cost(p, ci, co, h, w, kernel, stride);
                if (cost < bestCost) { best = p; bestCost = cost; }
            }
        }
    }
    TORCH_CHECK(std::isfinite(bestCost), "convolution strips and weights exceed hardware buffers");
    return best;
}

std::vector<int64_t> plan_values(int64_t ci, int64_t co, int64_t h, int64_t w, int64_t kernel, int64_t stride) {
    auto p = make_plan(ci, co, h, w, kernel, stride);
    return {p.m, p.n, p.k, p.channelCores, 8 / p.channelCores, p.inputChannels, p.inputBanks,
            1, 2, kernel == 1 ? 1 : 2, 2, 2, 2, p.epilogueM, p.ubBytes, p.l1Bytes, p.maxP, p.weightGroups};
}

at::Tensor conv(const at::Tensor &x, const at::Tensor &packed, const std::optional<at::Tensor> &bias,
                int64_t co, int64_t kernel, int64_t stride, bool fused, bool nchw) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && x.dim() == 5 && x.size(0) == 1 &&
                    x.size(4) == 16 && x.scalar_type() == at::kHalf && x.is_contiguous() &&
                    x.storage_offset() % 16 == 0 && !x.requires_grad(),
                "expected aligned contiguous N=1 FP16 NC1HWC0 input, C0=16");
    TORCH_CHECK(x.size(2) > 0 && x.size(3) > 0 && x.size(3) <= 65535 &&
                    x.numel() <= std::numeric_limits<int>::max(),
                "blocked convolution shape exceeds LoadData/index range: ", x.sizes());
    int ci = x.size(1) * 16, h = x.size(2), w = x.size(3);
    bool point = kernel == 1;
    TORCH_CHECK((point && stride == 1) || (kernel == 3 && (stride == 1 || stride == 2)),
                "expected point stride1 or spatial stride1/2");
    TORCH_CHECK(co > 0 && co <= 65536 && (nchw || co % 16 == 0),
                "blocked outputs require aligned channels and packed-weight DMA strides must fit uint16");
    int64_t outputPixels = int64_t((h - 1) / stride + 1) * ((w - 1) / stride + 1);
    TORCH_CHECK(co * ((outputPixels + 15) / 16 * 16) <= std::numeric_limits<int>::max(),
                "blocked convolution output exceeds index range");
    TORCH_CHECK(packed.device() == x.device() && packed.scalar_type() == at::kHalf &&
                    packed.is_contiguous() && packed.storage_offset() == 0 && !packed.requires_grad() &&
                    packed.sizes() == c10::IntArrayRef({ci / 16, kernel * kernel, (co + 63) / 64 * 64, 16}),
                "expected prepared FRACTAL_Z weights matching the convolution");
    if (bias)
        TORCH_CHECK(bias->device() == x.device() && bias->scalar_type() == at::kHalf &&
                        bias->is_contiguous() && bias->storage_offset() == 0 && !bias->requires_grad() &&
                        bias->dim() == 1 && bias->numel() >= (co + 31) / 32 * 32,
                    "expected same-device padded FP16 bias");
    auto plan = make_plan(ci, co, h, w, kernel, stride);
    thin::select(x.device());
    int oh = (h - 1) / stride + 1, ow = (w - 1) / stride + 1;
    int n = oh * ow, np = (n + 15) / 16 * 16;
    auto y =
        nchw ? at::empty({1, co, oh, ow}, x.options()) : at::empty({1, co / 16, oh, ow, 16}, x.options());
    auto destination = n == np ? y : at::empty({int64_t(co), int64_t(np)}, x.options());
    GM_ADDR bp = bias ? (GM_ADDR)bias->data_ptr() : nullptr;
    auto s = thin::stream();
#define RUN(KERNEL, H, FUSED) \
    convolution_native<KERNEL, H, FUSED><<<8, nullptr, s>>>( \
        (GM_ADDR)x.data_ptr(), (GM_ADDR)packed.data_ptr(), bp, (GM_ADDR)destination.data_ptr(), \
        ci, co, h, w, stride, oh, ow, plan.maxP, bias.has_value(), plan.m, \
        plan.inputChannels, plan.inputBanks, plan.channelCores, plan.weightGroups, np, plan.n, plan.k)
#define OUTPUT(KERNEL) \
    if (nchw) { \
        if (fused) { RUN(KERNEL, true, true); } else { RUN(KERNEL, true, false); } \
    } else { \
        if (fused) { RUN(KERNEL, false, true); } else { RUN(KERNEL, false, false); } \
    }
    if (point) {
        OUTPUT(1);
    } else {
        OUTPUT(3);
    }
#undef OUTPUT
#undef RUN
    if (n != np)
        trim_output<<<8, nullptr, s>>>((GM_ADDR)destination.data_ptr(), (GM_ADDR)y.data_ptr(), co, n, np, nchw);
    thin::sync();
    return y;
}
} // namespace native_flow
namespace image_c4 {
at::Tensor conv_blocked(const at::Tensor &, const at::Tensor &, const std::optional<at::Tensor> &);
}
TORCH_LIBRARY_FRAGMENT(thin_native_flow, m) {
    m.def("plan(int ci, int co, int h, int w, int kernel, int stride) -> int[]", TORCH_FN(native_flow::plan_values));
    m.def("conv(Tensor x, Tensor packed, Tensor? bias, int co, int kernel, int stride, bool fused=True, bool "
          "nchw_output=False) -> Tensor");
    m.def("stem(Tensor x, Tensor w, Tensor? bias) -> Tensor");
}
TORCH_LIBRARY_IMPL(thin_native_flow, PrivateUse1, m) {
    m.impl("conv", TORCH_FN(native_flow::conv));
    m.impl("stem", TORCH_FN(image_c4::conv_blocked));
}
TORCH_LIBRARY_IMPL(thin_native_flow, AutogradPrivateUse1, m) {
    m.impl("conv", torch::CppFunction::makeFallthrough());
    m.impl("stem", torch::CppFunction::makeFallthrough());
}
