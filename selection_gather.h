constexpr int GATHER_ANCHORS = 32768;
constexpr int GATHER_PAIRS = 65536;

__global__ __aicore__ void gather_scores(GM_ADDR xp, GM_ADDR ap, GM_ADDR gp,
                                        int64_t batches, int64_t anchors, int64_t classes,
                                        int64_t anchor_pitch, int64_t kp, Layout layout) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> scoreBuf, anchorBuf, offsetBuf, indexBuf, outputBuf;
    pipe.InitBuffer(scoreBuf, (GATHER_ANCHORS + 16) * 2);
    pipe.InitBuffer(anchorBuf, MERGE_PREFIX * 32);
    pipe.InitBuffer(offsetBuf, MERGE_PREFIX * 4);
    pipe.InitBuffer(indexBuf, MERGE_PREFIX * 4);
    pipe.InitBuffer(outputBuf, MERGE_PREFIX * 2);
    auto scores = scoreBuf.Get<uint16_t>(), output = outputBuf.Get<uint16_t>();
    auto records = anchorBuf.Get<int32_t>(), offsets = offsetBuf.Get<int32_t>();
    auto index = indexBuf.Get<float>();
    CreateVecIndex(index, 0.0f, MERGE_PREFIX);
    PipeBarrier<PIPE_V>();
    Muls(index, index, 32.0f, MERGE_PREFIX);
    PipeBarrier<PIPE_V>();
    Adds(index, index, 8.0f, MERGE_PREFIX);
    PipeBarrier<PIPE_V>();
    Cast(offsets, index, RoundMode::CAST_RINT, MERGE_PREFIX);
    GlobalTensor<uint16_t> input, result;
    GlobalTensor<int32_t> first;
    input.SetGlobalBuffer((__gm__ uint16_t*)xp);
    first.SetGlobalBuffer((__gm__ int32_t*)ap);
    result.SetGlobalBuffer((__gm__ uint16_t*)gp);
    for (int64_t job = GetBlockIdx(); job < batches * classes; job += GetBlockNum()) {
        int64_t batch = job / classes, cls = job % classes;
        int64_t address = layout.skew + batch * layout.batch + (cls + 4) * layout.channel;
        int skew = address % 16;
        fence<HardEvent::V_MTE2>();
        DataCopy(scores, input[address - skew], (anchors + skew + 15) / 16 * 16);
        DataCopy(records, first[batch * anchor_pitch * 8], kp * 8);
        fence<HardEvent::MTE2_V>();
        Gather(index.ReinterpretCast<int32_t>(), records, offsets.ReinterpretCast<uint32_t>(), 0, kp);
        PipeBarrier<PIPE_V>();
        Cast(index, index.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, kp);
        PipeBarrier<PIPE_V>();
        Muls(index, index, 2.0f, kp);
        PipeBarrier<PIPE_V>();
        Adds(index, index, float(skew * 2), kp);
        PipeBarrier<PIPE_V>();
        Cast(index.ReinterpretCast<int32_t>(), index, RoundMode::CAST_RINT, kp);
        PipeBarrier<PIPE_V>();
        Gather(output, scores, index.ReinterpretCast<uint32_t>(), 0, kp);
        fence<HardEvent::V_MTE3>();
        DataCopy(result[(batch * classes + cls) * kp], output, kp);
        fence<HardEvent::MTE3_V>();
    }
}
