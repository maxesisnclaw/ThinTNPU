#include "kernel_operator.h"
#include "graph_capture.h"
#include <acl/acl_rt.h>
#include <torch/all.h>
#include <torch/library.h>

using namespace AscendC;

namespace thin {
aclrtStream stream();
void sync();
void select(c10::Device);
}

namespace detection_selection {
constexpr int TILE = 128;
constexpr int READ_HALF = 1024;
constexpr uint64_t INVALID_KEY = 65536;

struct Layout {
    int64_t batch, anchor, channel, skew;
};

template <HardEvent E>
__aicore__ inline void fence() {
    SetFlag<E>(EVENT_ID0);
    WaitFlag<E>(EVENT_ID0);
}

__aicore__ inline uint64_t score_key(uint16_t bits) {
    uint32_t magnitude = bits & 0x7fff;
    if (magnitude > 0x7c00) return 0;
    if (magnitude == 0) return 0x7fff;
    return bits & 0x8000 ? bits : uint16_t(~(bits ^ 0x8000));
}

__aicore__ inline uint32_t float_bits(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 31;
    uint32_t mantissa = h & 1023;
    if (exponent == 31)
        return sign | 0x7f800000u | (mantissa << 13) | (mantissa ? 0x00400000u : 0);
    if (exponent) return sign | ((exponent + 112) << 23) | (mantissa << 13);
    if (!mantissa) return sign;
    exponent = 113;
    while (!(mantissa & 1024)) {
        mantissa <<= 1;
        --exponent;
    }
    return sign | (exponent << 23) | ((mantissa & 1023) << 13);
}

// Records carry the comparison key, stable source index, original half bits, and label/anchor.
__aicore__ inline bool less(LocalTensor<uint64_t> a, int left_index, LocalTensor<uint64_t> b, int bi) {
    uint64_t ak = a.GetValue(left_index * 4), bk = b.GetValue(bi * 4);
    return ak < bk || (ak == bk && a.GetValue(left_index * 4 + 1) < b.GetValue(bi * 4 + 1));
}

__aicore__ inline void copy_record(LocalTensor<uint64_t> dst, int di,
                                  LocalTensor<uint64_t> src, int si) {
    for (int c = 0; c < 4; ++c) dst.SetValue(di * 4 + c, src.GetValue(si * 4 + c));
}

#include "selection_sort.h"

__aicore__ inline uint16_t read_half(GlobalTensor<uint16_t> input, LocalTensor<uint16_t> cache,
                                    int64_t offset, int64_t& cached) {
    int64_t base = offset / 16 * 16;
    if (base != cached) {
        fence<HardEvent::S_MTE2>();
        DataCopy(cache, input[base], 16);
        fence<HardEvent::MTE2_S>();
        cached = base;
    }
    return cache.GetValue(offset % 16);
}

__aicore__ inline void initialize(LocalTensor<uint64_t> records, int64_t start) {
    for (int i = 0; i < TILE; ++i) {
        records.SetValue(i * 4, INVALID_KEY);
        records.SetValue(i * 4 + 1, start + i);
        records.SetValue(i * 4 + 2, 0);
        records.SetValue(i * 4 + 3, 0);
    }
}

__aicore__ inline void update_max(LocalTensor<uint64_t> records, int i, uint16_t bits, int64_t cls) {
    uint64_t key = score_key(bits);
    if (key < records.GetValue(i * 4)) {
        records.SetValue(i * 4, key);
        records.SetValue(i * 4 + 2, bits);
        records.SetValue(i * 4 + 3, cls);
    }
}

#include "selection_max.h"
#include "selection_reduce.h"

__global__ __aicore__ void anchor_tiles(GM_ADDR xp, GM_ADDR rp, GM_ADDR mp, int64_t batches,
                                       int64_t anchors, int64_t classes, int64_t pitch, Layout layout) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> recordBuf, inputBuf;
    pipe.InitBuffer(recordBuf, TILE * 32);
    pipe.InitBuffer(inputBuf, (READ_HALF + 16) * 2);
    TileSort sorter;
    sorter.init(pipe);
    ClassMaximum maximum;
    maximum.init(pipe);
    ClassReduction reduction;
    reduction.init(pipe);
    auto records = recordBuf.Get<uint64_t>();
    auto input = inputBuf.Get<uint16_t>();
    GlobalTensor<uint16_t> x;
    GlobalTensor<uint64_t> result;
    x.SetGlobalBuffer((__gm__ uint16_t*)xp);
    result.SetGlobalBuffer((__gm__ uint64_t*)rp);
    int64_t tiles = pitch / TILE;
    for (int64_t job = GetBlockIdx(); job < batches * tiles; job += GetBlockNum()) {
        int64_t batch = job / tiles, start = job % tiles * TILE;
        int count = anchors - start < TILE ? anchors - start : TILE;
        int64_t base = layout.skew + batch * layout.batch;
        initialize(records, start);
        if (layout.anchor == 1 && classes <= (1 << 24) && base % 16 == 0
                && layout.channel % 16 == 0 && layout.channel >= (count + 15) / 16 * 16
                && layout.channel / 16 <= 65535) {
            reduction.run(x, records, base, start, count, classes, layout.channel);
        } else if (layout.anchor == 1 && classes <= (1 << 24)) {
            maximum.run(x, input, records, base, start, count, classes, layout.channel);
        } else if (layout.anchor == 1) {
            for (int64_t c = 0; c < classes; ++c) {
                int64_t offset = base + start + (c + 4) * layout.channel;
                int skew = offset % 16;
                fence<HardEvent::S_MTE2>();
                DataCopy(input, x[offset - skew], (skew + count + 15) / 16 * 16);
                fence<HardEvent::MTE2_S>();
                for (int i = 0; i < count; ++i) update_max(records, i, input.GetValue(skew + i), c);
            }
        } else if (layout.channel == 1) {
            for (int i = 0; i < count; ++i) {
                for (int64_t c = 0; c < classes; c += READ_HALF) {
                    int n = classes - c < READ_HALF ? classes - c : READ_HALF;
                    int64_t offset = base + (start + i) * layout.anchor + c + 4;
                    int skew = offset % 16;
                    fence<HardEvent::S_MTE2>();
                    DataCopy(input, x[offset - skew], (skew + n + 15) / 16 * 16);
                    fence<HardEvent::MTE2_S>();
                    for (int j = 0; j < n; ++j) update_max(records, i, input.GetValue(skew + j), c + j);
                }
            }
        } else {
            int64_t cached = -1;
            for (int i = 0; i < count; ++i) {
                for (int64_t c = 0; c < classes; ++c) {
                    auto bits = read_half(x, input, base + (start + i) * layout.anchor
                                                       + (c + 4) * layout.channel, cached);
                    update_max(records, i, bits, c);
                }
            }
        }
        if (mp != nullptr) {
            for (int i = 0; i < TILE; ++i) input.SetValue(i, records.GetValue(i * 4 + 2));
            GlobalTensor<uint16_t> values;
            values.SetGlobalBuffer((__gm__ uint16_t*)mp);
            fence<HardEvent::S_MTE3>();
            DataCopy(values[batch * pitch + start], input, TILE);
            fence<HardEvent::MTE3_S>();
        }
        sorter.sort(records);
        fence<HardEvent::S_MTE3>();
        DataCopy(result[(batch * pitch + start) * 4], records, TILE * 4);
        fence<HardEvent::MTE3_S>();
    }
}

__global__ __aicore__ void merge_tiles(GM_ADDR sp, GM_ADDR dp, int64_t batches,
                                      int64_t pitch, int64_t run) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> leftBuf, rightBuf, outputBuf, probeBuf;
    pipe.InitBuffer(leftBuf, TILE * 32);
    pipe.InitBuffer(rightBuf, TILE * 32);
    pipe.InitBuffer(outputBuf, TILE * 32);
    pipe.InitBuffer(probeBuf, 64);
    auto left = leftBuf.Get<uint64_t>(), right = rightBuf.Get<uint64_t>();
    auto output = outputBuf.Get<uint64_t>(), probe = probeBuf.Get<uint64_t>();
    GlobalTensor<uint64_t> source, target;
    source.SetGlobalBuffer((__gm__ uint64_t*)sp);
    target.SetGlobalBuffer((__gm__ uint64_t*)dp);
    int64_t tiles = pitch / TILE;
    for (int64_t job = GetBlockIdx(); job < batches * tiles; job += GetBlockNum()) {
        int64_t batch = job / tiles, start = job % tiles * TILE;
        int64_t group = start / (2 * run) * (2 * run);
        int64_t ln = pitch - group < run ? pitch - group : run;
        int64_t rn = pitch - group - ln < run ? pitch - group - ln : run;
        int64_t diagonal = start - group;
        int64_t lo = diagonal > rn ? diagonal - rn : 0;
        int64_t hi = diagonal < ln ? diagonal : ln;
        int64_t li = lo, ri = diagonal - lo;
        int64_t lb = (batch * pitch + group) * 4, rb = lb + ln * 4;
        while (lo <= hi) {
            li = lo + (hi - lo) / 2;
            ri = diagonal - li;
            if (li > 0 && ri < rn) {
                fence<HardEvent::S_MTE2>();
                DataCopy(probe, source[lb + (li - 1) * 4], 4);
                DataCopy(probe[4], source[rb + ri * 4], 4);
                fence<HardEvent::MTE2_S>();
                if (less(probe, 1, probe, 0)) {
                    hi = li - 1;
                    continue;
                }
            }
            if (ri > 0 && li < ln) {
                fence<HardEvent::S_MTE2>();
                DataCopy(probe, source[lb + li * 4], 4);
                DataCopy(probe[4], source[rb + (ri - 1) * 4], 4);
                fence<HardEvent::MTE2_S>();
                if (less(probe, 0, probe, 1)) {
                    lo = li + 1;
                    continue;
                }
            }
            break;
        }
        int lc = ln - li < TILE ? ln - li : TILE;
        int rc = rn - ri < TILE ? rn - ri : TILE;
        fence<HardEvent::S_MTE2>();
        if (lc) DataCopy(left, source[lb + li * 4], lc * 4);
        if (rc) DataCopy(right, source[rb + ri * 4], rc * 4);
        fence<HardEvent::MTE2_S>();
        int l = 0, r = 0;
        for (int i = 0; i < TILE; ++i) {
            if (r == rc || (l < lc && less(left, l, right, r))) copy_record(output, i, left, l++);
            else copy_record(output, i, right, r++);
        }
        fence<HardEvent::S_MTE3>();
        DataCopy(target[(batch * pitch + start) * 4], output, TILE * 4);
        fence<HardEvent::MTE3_S>();
    }
}

#include "selection_merge.h"
#include "selection_gather.h"
#include "selection_pairs.h"

__global__ __aicore__ void pair_tiles(GM_ADDR xp, GM_ADDR ap, GM_ADDR rp, GM_ADDR gp,
                                     GM_ADDR cp, int64_t score_pitch, int64_t batches,
                                     int64_t classes, int64_t k, int64_t anchor_pitch,
                                     int64_t pitch, Layout layout) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> recordBuf, inputBuf, anchorBuf;
    pipe.InitBuffer(recordBuf, TILE * 32);
    pipe.InitBuffer(inputBuf, TILE * 2);
    pipe.InitBuffer(anchorBuf, 32);
    TBuf<TPosition::VECCALC> compactBuf, selectedBuf;
    pipe.InitBuffer(compactBuf, cp ? classes * score_pitch * 2 : 32);
    pipe.InitBuffer(selectedBuf, cp ? score_pitch * 32 : 32);
    TileSort sorter;
    sorter.init(pipe);
    PairRecords builder;
    builder.init(pipe);
    auto records = recordBuf.Get<uint64_t>(), anchor = anchorBuf.Get<uint64_t>();
    auto input = inputBuf.Get<uint16_t>();
    auto compact = compactBuf.Get<uint16_t>();
    auto selected = selectedBuf.Get<uint64_t>();
    GlobalTensor<uint16_t> compact_source;
    compact_source.SetGlobalBuffer((__gm__ uint16_t*)cp);
    GlobalTensor<uint16_t> x;
    GlobalTensor<uint64_t> anchors, result;
    x.SetGlobalBuffer((__gm__ uint16_t*)xp);
    anchors.SetGlobalBuffer((__gm__ uint64_t*)ap);
    result.SetGlobalBuffer((__gm__ uint64_t*)rp);
    int64_t tiles = pitch / TILE, n = k * classes;
    int64_t cached_batch = -1;
    for (int64_t job = GetBlockIdx(); job < batches * tiles; job += GetBlockNum()) {
        int64_t batch = job / tiles, start = job % tiles * TILE;
        int count = n - start < TILE ? n - start : TILE;
        if (cp && batch != cached_batch) {
            fence<HardEvent::S_MTE2>();
            DataCopy(compact, compact_source[batch * classes * score_pitch], classes * score_pitch);
            DataCopy(selected, anchors[batch * anchor_pitch * 4], score_pitch * 4);
            fence<HardEvent::MTE2_S>();
            cached_batch = batch;
        }
        auto gathered_input = input;
        if (cp) {
            gathered_input = builder.make(records, compact, selected, start, count, classes, score_pitch);
        } else {
            initialize(records, start);
            int64_t cached = -1, previous_rank = -1, a = 0;
            for (int i = 0; i < count; ++i) {
                int64_t rank = (start + i) / classes, cls = (start + i) % classes;
                if (rank != previous_rank) {
                    fence<HardEvent::S_MTE2>();
                    DataCopy(anchor, anchors[(batch * anchor_pitch + rank) * 4], 4);
                    fence<HardEvent::MTE2_S>();
                    a = anchor.GetValue(1);
                    previous_rank = rank;
                }
                uint16_t bits = read_half(x, input, layout.skew + batch * layout.batch
                    + a * layout.anchor + (cls + 4) * layout.channel, cached);
                records.SetValue(i * 4, score_key(bits));
                records.SetValue(i * 4 + 2, bits);
                records.SetValue(i * 4 + 3, a);
            }
        }
        if (gp != nullptr) {
            if (!cp)
                for (int i = 0; i < TILE; ++i) input.SetValue(i, records.GetValue(i * 4 + 2));
            GlobalTensor<uint16_t> values;
            values.SetGlobalBuffer((__gm__ uint16_t*)gp);
            fence<HardEvent::S_MTE3>();
            DataCopy(values[batch * pitch + start], gathered_input, TILE);
            fence<HardEvent::MTE3_S>();
        }
        sorter.sort(records);
        fence<HardEvent::S_MTE3>();
        DataCopy(result[(batch * pitch + start) * 4], records, TILE * 4);
        fence<HardEvent::MTE3_S>();
    }
}

__global__ __aicore__ void pack(GM_ADDR xp, GM_ADDR rp, GM_ADDR yp, int64_t batches,
                               int64_t classes, int64_t k, int64_t pitch, bool agnostic, Layout layout) {
    constexpr int DETECTIONS = 16;
    TPipe pipe;
    TBuf<TPosition::VECCALC> recordBuf, inputBuf, outputBuf;
    pipe.InitBuffer(recordBuf, 32);
    pipe.InitBuffer(inputBuf, 32);
    pipe.InitBuffer(outputBuf, DETECTIONS * 6 * 4);
    auto record = recordBuf.Get<uint64_t>();
    auto input = inputBuf.Get<uint16_t>();
    auto output = outputBuf.Get<uint32_t>();
    auto floats = output.ReinterpretCast<float>();
    GlobalTensor<uint16_t> x;
    GlobalTensor<uint64_t> records;
    GlobalTensor<uint32_t> result;
    x.SetGlobalBuffer((__gm__ uint16_t*)xp);
    records.SetGlobalBuffer((__gm__ uint64_t*)rp);
    result.SetGlobalBuffer((__gm__ uint32_t*)yp);
    int64_t total = batches * k, tiles = (total + DETECTIONS - 1) / DETECTIONS;
    for (int64_t job = GetBlockIdx(); job < tiles; job += GetBlockNum()) {
        int64_t start = job * DETECTIONS;
        int count = total - start < DETECTIONS ? total - start : DETECTIONS;
        int64_t cached = -1;
        for (int i = 0; i < count; ++i) {
            int64_t batch = (start + i) / k, rank = (start + i) % k;
            fence<HardEvent::S_MTE2>();
            DataCopy(record, records[(batch * pitch + rank) * 4], 4);
            fence<HardEvent::MTE2_S>();
            int64_t a = record.GetValue(agnostic ? 1 : 3);
            int64_t cls = agnostic ? record.GetValue(3) : record.GetValue(1) % classes;
            for (int c = 0; c < 4; ++c) {
                auto bits = read_half(x, input, layout.skew + batch * layout.batch
                    + a * layout.anchor + c * layout.channel, cached);
                output.SetValue(i * 6 + c, float_bits(bits));
            }
            output.SetValue(i * 6 + 4, float_bits(record.GetValue(2)));
            floats.SetValue(i * 6 + 5, static_cast<float>(cls));
        }
        int padded = (count * 6 + 7) / 8 * 8;
        for (int i = count * 6; i < padded; ++i) output.SetValue(i, 0);
        fence<HardEvent::S_MTE3>();
        DataCopy(result[start * 6], output, padded);
        fence<HardEvent::MTE3_S>();
    }
}

at::Tensor sorted_runs(at::Tensor records, int64_t pitch, int64_t k) {
    if (pitch == TILE) return records;
    auto temporary = at::empty_like(records);
    for (int64_t run = TILE; run < pitch; run *= 2) {
        if (k > 0 && k <= MERGE_PREFIX) {
            merge_prefixes<<<8, nullptr, thin::stream()>>>((GM_ADDR)records.data_ptr(),
                (GM_ADDR)temporary.data_ptr(), records.size(0), pitch, run, (k + 15) / 16 * 16);
        } else {
            merge_tiles<<<8, nullptr, thin::stream()>>>((GM_ADDR)records.data_ptr(),
                (GM_ADDR)temporary.data_ptr(), records.size(0), pitch, run);
        }
        std::swap(records, temporary);
    }
    thin::sync();
    return records;
}

__global__ __aicore__ void trace_indices(GM_ADDR ap, GM_ADDR rp, GM_ADDR ip, GM_ADDR vp,
                                        int64_t batches, int64_t k, int64_t classes,
                                        int64_t anchor_pitch, int64_t pitch, int64_t kp, bool agnostic) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> firstBuf, finalBuf, indexBuf, valueBuf;
    pipe.InitBuffer(firstBuf, TILE * 32);
    pipe.InitBuffer(finalBuf, TILE * 32);
    pipe.InitBuffer(indexBuf, TILE * 4 * 8);
    pipe.InitBuffer(valueBuf, TILE * 2);
    auto first = firstBuf.Get<uint64_t>(), final = finalBuf.Get<uint64_t>();
    auto indices = indexBuf.Get<int64_t>();
    auto values = valueBuf.Get<uint16_t>();
    GlobalTensor<uint64_t> anchors, records;
    GlobalTensor<int64_t> index_output;
    GlobalTensor<uint16_t> value_output;
    anchors.SetGlobalBuffer((__gm__ uint64_t*)ap);
    records.SetGlobalBuffer((__gm__ uint64_t*)rp);
    index_output.SetGlobalBuffer((__gm__ int64_t*)ip);
    value_output.SetGlobalBuffer((__gm__ uint16_t*)vp);
    int64_t tiles = kp / TILE;
    for (int64_t job = GetBlockIdx(); job < batches * tiles; job += GetBlockNum()) {
        int64_t batch = job / tiles, start = job % tiles * TILE;
        int count = k - start < TILE ? k - start : TILE;
        fence<HardEvent::S_MTE2>();
        DataCopy(first, anchors[(batch * anchor_pitch + start) * 4], TILE * 4);
        DataCopy(final, records[(batch * pitch + start) * 4], TILE * 4);
        fence<HardEvent::MTE2_S>();
        for (int i = 0; i < TILE; ++i) {
            bool valid = i < count;
            indices.SetValue(i, valid ? first.GetValue(i * 4 + 1) : 0);
            indices.SetValue(TILE + i, valid ? final.GetValue(i * 4 + 1) : 0);
            indices.SetValue(2 * TILE + i, valid ? final.GetValue(i * 4 + (agnostic ? 1 : 3)) : 0);
            indices.SetValue(3 * TILE + i, valid ? (agnostic ? final.GetValue(i * 4 + 3)
                                                         : final.GetValue(i * 4 + 1) % classes) : 0);
            values.SetValue(i, valid ? final.GetValue(i * 4 + 2) : 0);
        }
        fence<HardEvent::S_MTE3>();
        for (int c = 0; c < 4; ++c)
            DataCopy(index_output[(batch * 4 + c) * kp + start], indices[c * TILE], TILE);
        DataCopy(value_output[batch * kp + start], values, TILE);
        fence<HardEvent::MTE3_S>();
    }
}

std::vector<at::Tensor> run(const at::Tensor& decoded, int64_t max_det, bool agnostic, bool trace) {
    TORCH_CHECK(decoded.device().type() == c10::DeviceType::PrivateUse1
        && decoded.scalar_type() == at::kHalf && decoded.dim() == 3 && !decoded.requires_grad(),
        "detection_select expects inference-only device FP16 [B,A,4+nc]");
    TORCH_CHECK(decoded.size(2) > 4 && max_det >= 0,
        "detection_select requires nc > 0 and max_det >= 0");
    int64_t batches = decoded.size(0), anchors = decoded.size(1), classes = decoded.size(2) - 4;
    int64_t k = std::min(max_det, anchors);
    thin::select(decoded.device());
    auto output = at::empty({batches, k, 6}, decoded.options().dtype(at::kFloat));
    if (!trace && (!batches || !k)) return {output};
    int64_t anchor_pitch = (anchors + TILE - 1) / TILE * TILE;
    int64_t pitch = agnostic ? anchor_pitch : (k * classes + TILE - 1) / TILE * TILE;
    int64_t kp = (k + TILE - 1) / TILE * TILE;
    at::Tensor maxima, gathered, indices, values;
    if (trace) {
        maxima = at::empty({batches, anchor_pitch}, decoded.options());
        gathered = at::empty({batches, agnostic ? 0 : pitch}, decoded.options());
        indices = at::empty({batches, 4, kp}, decoded.options().dtype(at::kLong));
        values = at::empty({batches, kp}, decoded.options());
    }
    if (batches && anchors) {
        Layout layout{decoded.stride(0), decoded.stride(1), decoded.stride(2),
            int64_t(reinterpret_cast<uintptr_t>(decoded.const_data_ptr()) % 32 / 2)};
        auto input = reinterpret_cast<GM_ADDR>(reinterpret_cast<uintptr_t>(decoded.const_data_ptr()) / 32 * 32);
        auto records = at::empty({batches, anchor_pitch, 4}, decoded.options().dtype(at::kLong));
        anchor_tiles<<<8, nullptr, thin::stream()>>>(input, (GM_ADDR)records.data_ptr(),
            trace ? (GM_ADDR)maxima.data_ptr() : nullptr, batches, anchors, classes, anchor_pitch, layout);
        records = sorted_runs(records, anchor_pitch, k);
        auto anchor_records = records;
        if (k) {
            if (!agnostic) {
                auto pairs = at::empty({batches, pitch, 4}, decoded.options().dtype(at::kLong));
                int64_t score_pitch = (k + 15) / 16 * 16;
                at::Tensor compact;
                if (layout.anchor == 1 && anchors <= GATHER_ANCHORS && k <= MERGE_PREFIX
                        && classes * score_pitch <= GATHER_PAIRS) {
                    compact = at::empty({batches, classes, score_pitch}, decoded.options());
                    gather_scores<<<8, nullptr, thin::stream()>>>(input, (GM_ADDR)records.data_ptr(),
                        (GM_ADDR)compact.data_ptr(), batches, anchors, classes, anchor_pitch, score_pitch, layout);
                }
                pair_tiles<<<8, nullptr, thin::stream()>>>(input, (GM_ADDR)records.data_ptr(),
                    (GM_ADDR)pairs.data_ptr(), trace ? (GM_ADDR)gathered.data_ptr() : nullptr,
                    compact.defined() ? (GM_ADDR)compact.data_ptr() : nullptr, score_pitch,
                    batches, classes, k, anchor_pitch, pitch, layout);
                thin::sync();
                records = sorted_runs(pairs, pitch, k);
            }
            pack<<<8, nullptr, thin::stream()>>>(input, (GM_ADDR)records.data_ptr(), (GM_ADDR)output.data_ptr(),
                batches, classes, k, pitch, agnostic, layout);
            if (trace)
                trace_indices<<<8, nullptr, thin::stream()>>>((GM_ADDR)anchor_records.data_ptr(),
                    (GM_ADDR)records.data_ptr(), (GM_ADDR)indices.data_ptr(), (GM_ADDR)values.data_ptr(),
                    batches, k, classes, anchor_pitch, pitch, kp, agnostic);
        }
        thin::sync();
    }
    if (!trace) return {output};
    return {output, maxima.slice(1, 0, anchors), indices.select(1, 0).slice(1, 0, k),
        gathered.as_strided({batches, agnostic ? 0 : k, classes}, {agnostic ? 0 : pitch, classes, 1}),
        values.slice(1, 0, k), indices.select(1, 1).slice(1, 0, agnostic ? 0 : k),
        indices.select(1, 2).slice(1, 0, k), indices.select(1, 3).slice(1, 0, k)};
}

at::Tensor detection_select(const at::Tensor& decoded, int64_t max_det, bool agnostic) {
    return run(decoded, max_det, agnostic, false)[0];
}

std::vector<at::Tensor> detection_select_trace(const at::Tensor& decoded, int64_t max_det, bool agnostic) {
    return run(decoded, max_det, agnostic, true);
}

}

TORCH_LIBRARY_FRAGMENT(thin_native_flow, m) {
    m.def("detection_select(Tensor decoded, int max_det, bool agnostic=False) -> Tensor");
    m.def("detection_select_trace(Tensor decoded, int max_det, bool agnostic=False) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(thin_native_flow, PrivateUse1, m) {
    m.impl("detection_select", TORCH_FN(detection_selection::detection_select));
    m.impl("detection_select_trace", TORCH_FN(detection_selection::detection_select_trace));
}
TORCH_LIBRARY_IMPL(thin_native_flow, AutogradPrivateUse1, m) {
    m.impl("detection_select", torch::CppFunction::makeFallthrough());
    m.impl("detection_select_trace", torch::CppFunction::makeFallthrough());
}
