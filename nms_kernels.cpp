#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t TILE = 1024;
constexpr float NEG_INF = -__builtin_inff();

template<HardEvent E> __aicore__ inline void fence() {
    SetFlag<E>(EVENT_ID0);
    WaitFlag<E>(EVENT_ID0);
}

extern "C" __global__ __aicore__ void nms_fp32(
    GM_ADDR boxes, GM_ADDR scores, GM_ADDR scratch, GM_ADDR indices, GM_ADDR count,
    uint64_t length, uint64_t max_out, float iou_threshold, float score_threshold) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> data_buf, aos_buf, offset_buf, temp_buf, mask_buf, reduce_buf, out_buf;
    pipe.InitBuffer(data_buf, 6 * TILE * sizeof(float));
    pipe.InitBuffer(aos_buf, 4 * TILE * sizeof(float));
    pipe.InitBuffer(offset_buf, TILE * sizeof(uint32_t));
    pipe.InitBuffer(temp_buf, 4 * TILE * sizeof(float));
    pipe.InitBuffer(mask_buf, TILE / 8);
    pipe.InitBuffer(reduce_buf, TILE / 64 * 2 * sizeof(float));
    pipe.InitBuffer(out_buf, TILE * sizeof(int32_t));
    auto data = data_buf.Get<float>();
    auto aos = aos_buf.Get<float>();
    auto offsets = offset_buf.Get<uint32_t>();
    auto t0 = temp_buf.Get<float>();
    auto t1 = t0[TILE];
    auto t2 = t0[2 * TILE];
    auto t3 = t0[3 * TILE];
    auto mask = mask_buf.Get<uint8_t>();
    auto reduced = reduce_buf.Get<float>();
    auto out = out_buf.Get<int32_t>();
    auto x1 = data; auto y1 = data[TILE]; auto x2 = data[2*TILE];
    auto y2 = data[3*TILE]; auto area = data[4*TILE]; auto score = data[5*TILE];
    GlobalTensor<float> bg, sg, work;
    GlobalTensor<int32_t> ig, cg;
    bg.SetGlobalBuffer((__gm__ float*)boxes);
    sg.SetGlobalBuffer((__gm__ float*)scores);
    work.SetGlobalBuffer((__gm__ float*)scratch);
    ig.SetGlobalBuffer((__gm__ int32_t*)indices);
    cg.SetGlobalBuffer((__gm__ int32_t*)count);
    Duplicate(out, int32_t(-1), TILE);
    fence<HardEvent::V_MTE3>();
    for (uint64_t k = 0; k < max_out; k += TILE) {
        uint32_t n = max_out-k < TILE ? max_out-k : TILE;
        DataCopy(ig[k], out, (n+7)/8*8);
    }
    fence<HardEvent::MTE3_S>();
    for (uint32_t i = 0; i < TILE; ++i) offsets.SetValue(i, i*16);
    fence<HardEvent::S_V>();
    const uint64_t tiles = (length+TILE-1)/TILE;
    for (uint64_t k = 0; k < tiles; ++k) {
        uint64_t start = k*TILE;
        uint32_t n = length-start < TILE ? length-start : TILE;
        DataCopy(aos, bg[start*4], (n*4+7)/8*8);
        DataCopy(score, sg[start], (n+7)/8*8);
        fence<HardEvent::MTE2_V>();
        for (uint32_t j = 0; j < 4; ++j) Gather(data[j*TILE], aos, offsets, j*4, TILE);
        PipeBarrier<PIPE_V>();
        Sub(t0, x2, x1, TILE); Sub(t1, y2, y1, TILE);
        PipeBarrier<PIPE_V>();
        Mul(area, t0, t1, TILE);
        fence<HardEvent::V_S>();
        for (uint32_t i = n; i < TILE; ++i) score.SetValue(i, NEG_INF);
        fence<HardEvent::S_MTE3>();
        DataCopy(work[k*6*TILE], data, 6*TILE);
        fence<HardEvent::MTE3_MTE2>();
    }
    float px1=0, py1=0, px2=0, py2=0, pa=0;
    int64_t previous=-1;
    uint32_t kept=0;
    while (kept < max_out) {
        float best = NEG_INF;
        int64_t best_index=-1;
        float bx1=0, by1=0, bx2=0, by2=0, ba=0;
        for (uint64_t k = 0; k < tiles; ++k) {
            DataCopy(data, work[k*6*TILE], 6*TILE);
            fence<HardEvent::MTE2_V>();
            if (previous >= 0) {
                Mins(t0, x2, px2, TILE); Maxs(t1, x1, px1, TILE);
                Mins(t2, y2, py2, TILE); Maxs(t3, y1, py1, TILE);
                PipeBarrier<PIPE_V>();
                Sub(t0, t0, t1, TILE); Sub(t2, t2, t3, TILE);
                PipeBarrier<PIPE_V>();
                Maxs(t0, t0, 0.0f, TILE); Maxs(t2, t2, 0.0f, TILE);
                Adds(t1, area, pa, TILE);
                PipeBarrier<PIPE_V>();
                Mul(t0, t0, t2, TILE);
                PipeBarrier<PIPE_V>();
                Sub(t1, t1, t0, TILE);
                PipeBarrier<PIPE_V>();
                Div(t0, t0, t1, TILE);
                PipeBarrier<PIPE_V>();
                CompareScalar(mask, t0, iou_threshold, CMPMODE::GT, TILE);
                Duplicate(t1, NEG_INF, TILE);
                PipeBarrier<PIPE_V>();
                Select(score, mask, t1, score, SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE);
                fence<HardEvent::V_S>();
                if (uint64_t(previous)/TILE == k) score.SetValue(previous%TILE, NEG_INF);
                fence<HardEvent::S_V>();
            }
            WholeReduceMax(reduced, score, 64, TILE/64, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
            fence<HardEvent::V_S>();
            auto ri = reduced.ReinterpretCast<uint32_t>();
            for (uint32_t r = 0; r < TILE/64; ++r) {
                float value = reduced.GetValue(r*2);
                if (value > best) {
                    uint32_t i = r*64 + ri.GetValue(r*2+1);
                    best=value; best_index=k*TILE+i;
                    bx1=x1.GetValue(i); by1=y1.GetValue(i);
                    bx2=x2.GetValue(i); by2=y2.GetValue(i); ba=area.GetValue(i);
                }
            }
            fence<HardEvent::V_MTE3>();
            DataCopy(work[k*6*TILE+5*TILE], score, TILE);
            fence<HardEvent::MTE3_MTE2>();
        }
        if (best_index < 0 || !(best > score_threshold)) break;
        previous=best_index; px1=bx1; py1=by1; px2=bx2; py2=by2; pa=ba;
        out.SetValue(kept%8, int32_t(best_index));
        ++kept;
        if (kept%8 == 0) {
            fence<HardEvent::S_MTE3>();
            DataCopy(ig[kept-8], out, 8);
            fence<HardEvent::MTE3_S>();
        }
    }
    if (kept%8) {
        for (uint32_t i=kept%8; i<8; ++i) out.SetValue(i, -1);
        fence<HardEvent::S_MTE3>();
        DataCopy(ig[kept/8*8], out, 8);
        fence<HardEvent::MTE3_S>();
    }
    out.SetValue(0, int32_t(kept));
    fence<HardEvent::S_MTE3>();
    DataCopy(cg, out, 8);
}
