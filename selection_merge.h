constexpr int MERGE_PREFIX = 512;

__global__ __aicore__ void merge_prefixes(GM_ADDR sp, GM_ADDR dp, int64_t batches,
                                         int64_t pitch, int64_t run, int64_t keep) {
    constexpr int CAPACITY = 2 * MERGE_PREFIX;
    TPipe pipe;
    TBuf<TPosition::VECCALC> sourceBuf, proposalBuf, resultBuf, sideBuf;
    pipe.InitBuffer(sourceBuf, CAPACITY * 32);
    pipe.InitBuffer(proposalBuf, CAPACITY * 32);
    pipe.InitBuffer(resultBuf, CAPACITY * 32);
    pipe.InitBuffer(sideBuf, CAPACITY * 4);
    RecordPermutation<CAPACITY> permutation;
    permutation.init(pipe);
    auto source = sourceBuf.Get<uint64_t>();
    auto proposal = proposalBuf.Get<float>(), result = resultBuf.Get<float>();
    auto keys = permutation.keyBuf.Get<float>(), sides = sideBuf.Get<float>();
    auto index = permutation.indexBuf.Get<uint32_t>();
    GlobalTensor<uint64_t> input, output;
    input.SetGlobalBuffer((__gm__ uint64_t*)sp);
    output.SetGlobalBuffer((__gm__ uint64_t*)dp);
    int64_t groups = (pitch + 2 * run - 1) / (2 * run);
    for (int64_t job = GetBlockIdx(); job < batches * groups; job += GetBlockNum()) {
        int64_t batch = job / groups, start = job % groups * 2 * run;
        int left = int(pitch - start < run ? pitch - start : run);
        int right = int(pitch - start - left < run ? pitch - start - left : run);
        left = left < keep ? left : keep;
        right = right < keep ? right : keep;
        int count = left + right;
        int written = count < keep ? count : keep;
        fence<HardEvent::V_MTE2>();
        DataCopy(source, input[(batch * pitch + start) * 4], left * 4);
        if (right) DataCopy(source[left * 4], input[(batch * pitch + start + run) * 4], right * 4);
        fence<HardEvent::MTE2_V>();
        if (!right) {
            fence<HardEvent::V_MTE3>();
            DataCopy(output[(batch * pitch + start) * 4], source, written * 4);
            fence<HardEvent::MTE3_V>();
            continue;
        }
        permutation.keys(source, 2.0f, count);
        Duplicate(sides, 0.0f, left);
        Duplicate(sides[left], 1.0f, right);
        Cast(index.ReinterpretCast<int32_t>(), permutation.rankBuf.Get<float>(), RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        // Left runs cover lower original indices; a merge preserves order within each input stream.
        Add(keys, keys, sides, count);
        PipeBarrier<PIPE_V>();
        Muls(keys, keys, -1.0f, count);
        PipeBarrier<PIPE_V>();
        ProposalConcat(proposal, keys, count / 16, 4);
        ProposalConcat(proposal, index.ReinterpretCast<float>(), count / 16, 5);
        PipeBarrier<PIPE_V>();
        uint16_t lengths[4] = {uint16_t(left), uint16_t(right), 0, 0};
        uint32_t consumed[4];
        MrgSort(result, MrgSortSrcList<float>(proposal, proposal[left * 8], proposal, proposal),
                lengths, consumed, 3, 1);
        PipeBarrier<PIPE_V>();
        Extract(keys, index, result, written / 16);
        PipeBarrier<PIPE_V>();
        permutation.gather(proposal.ReinterpretCast<uint64_t>(), source, written);
        fence<HardEvent::V_MTE3>();
        DataCopy(output[(batch * pitch + start) * 4], proposal.ReinterpretCast<uint64_t>(), written * 4);
        fence<HardEvent::MTE3_V>();
    }
}
