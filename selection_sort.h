template<int RECORDS>
struct RecordPermutation {
    TBuf<TPosition::VECCALC> keyBuf, rankBuf, keyOffsetBuf, repeatOffsetBuf, wordOffsetBuf, workBuf, indexBuf;

    __aicore__ inline void init(TPipe& pipe) {
        pipe.InitBuffer(keyBuf, RECORDS * 4);
        pipe.InitBuffer(rankBuf, RECORDS * 4);
        pipe.InitBuffer(keyOffsetBuf, RECORDS * 4);
        pipe.InitBuffer(repeatOffsetBuf, RECORDS * 8 * 4);
        pipe.InitBuffer(wordOffsetBuf, RECORDS * 8 * 4);
        pipe.InitBuffer(workBuf, RECORDS * 8 * 4);
        pipe.InitBuffer(indexBuf, RECORDS * 4);
        auto work = workBuf.Get<float>(), word = wordOffsetBuf.Get<float>();
        auto repeat = repeatOffsetBuf.Get<int32_t>();
        auto ranks = rankBuf.Get<float>(), keys = keyBuf.Get<float>();
        CreateVecIndex(work, 0.0f, RECORDS * 8);
        PipeBarrier<PIPE_V>();
        Muls(word, work, 4.0f, RECORDS * 8);
        Muls(work, work, 0.125f, RECORDS * 8);
        PipeBarrier<PIPE_V>();
        Cast(repeat, work, RoundMode::CAST_FLOOR, RECORDS * 8);
        PipeBarrier<PIPE_V>();
        Cast(work, repeat, RoundMode::CAST_NONE, RECORDS * 8);
        PipeBarrier<PIPE_V>();
        Muls(work, work, 32.0f, RECORDS * 8);
        PipeBarrier<PIPE_V>();
        Sub(word, word, work, RECORDS * 8);
        Muls(work, work, 0.125f, RECORDS * 8);
        PipeBarrier<PIPE_V>();
        Cast(repeat, work, RoundMode::CAST_RINT, RECORDS * 8);
        CreateVecIndex(ranks, 0.0f, RECORDS);
        PipeBarrier<PIPE_V>();
        Muls(keys, ranks, 32.0f, RECORDS);
        Cast(indexBuf.Get<int32_t>(), ranks, RoundMode::CAST_RINT, RECORDS);
        PipeBarrier<PIPE_V>();
        Cast(keyOffsetBuf.Get<int32_t>(), keys, RoundMode::CAST_RINT, RECORDS);
        fence<HardEvent::V_S>();
    }

    __aicore__ inline void keys(LocalTensor<uint64_t> source, float scale, int count) {
        auto key = keyBuf.Get<float>();
        Gather(key.ReinterpretCast<int32_t>(), source.ReinterpretCast<int32_t>(), keyOffsetBuf.Get<uint32_t>(), 0, count);
        PipeBarrier<PIPE_V>();
        Cast(key, key.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Muls(key, key, scale, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void gather(LocalTensor<uint64_t> target, LocalTensor<uint64_t> source, int count) {
        auto work = workBuf.Get<float>();
        Gather(work.ReinterpretCast<uint32_t>(), indexBuf.Get<uint32_t>(), repeatOffsetBuf.Get<uint32_t>(), 0, count * 8);
        PipeBarrier<PIPE_V>();
        Cast(work, work.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, count * 8);
        PipeBarrier<PIPE_V>();
        Muls(work, work, 32.0f, count * 8);
        PipeBarrier<PIPE_V>();
        Add(work, work, wordOffsetBuf.Get<float>(), count * 8);
        PipeBarrier<PIPE_V>();
        Cast(work.ReinterpretCast<int32_t>(), work, RoundMode::CAST_RINT, count * 8);
        PipeBarrier<PIPE_V>();
        Gather(target.ReinterpretCast<uint32_t>(), source.ReinterpretCast<uint32_t>(), work.ReinterpretCast<uint32_t>(), 0, count * 8);
        PipeBarrier<PIPE_V>();
    }
};

struct TileSort {
    RecordPermutation<TILE> permutation;
    TBuf<TPosition::VECCALC> proposalBuf, sortedBuf, temporaryBuf;

    __aicore__ inline void init(TPipe& pipe) {
        permutation.init(pipe);
        pipe.InitBuffer(proposalBuf, TILE * 32);
        pipe.InitBuffer(sortedBuf, TILE * 32);
        pipe.InitBuffer(temporaryBuf, TILE * 32);
    }

    __aicore__ inline void sort(LocalTensor<uint64_t> records) {
        auto keys = permutation.keyBuf.Get<float>();
        auto index = permutation.indexBuf.Get<uint32_t>();
        auto proposal = proposalBuf.Get<float>();
        auto sorted = sortedBuf.Get<float>();
        auto temporary = temporaryBuf.Get<float>();
        fence<HardEvent::S_V>();
        permutation.keys(records, 128.0f, TILE);
        // Valid half keys and local ranks form exact, unique FP32 integer keys.
        Add(keys, keys, permutation.rankBuf.Get<float>(), TILE);
        PipeBarrier<PIPE_V>();
        Muls(keys, keys, -1.0f, TILE);
        PipeBarrier<PIPE_V>();
        Cast(index.ReinterpretCast<int32_t>(), permutation.rankBuf.Get<float>(), RoundMode::CAST_RINT, TILE);
        ProposalConcat(proposal, keys, TILE / 16, 4);
        PipeBarrier<PIPE_V>();
        Sort<float, false>(sorted, proposal, index, temporary, TILE / 16);
        PipeBarrier<PIPE_V>();
        uint32_t consumed[4];
        const uint16_t small[4] = {16, 16, 16, 16};
        MrgSort(temporary, MrgSortSrcList<float>(sorted, sorted[16 * 8], sorted[32 * 8], sorted[48 * 8]),
                small, consumed, 15, 1);
        MrgSort(temporary[64 * 8], MrgSortSrcList<float>(sorted[64 * 8], sorted[80 * 8], sorted[96 * 8], sorted[112 * 8]),
                small, consumed, 15, 1);
        PipeBarrier<PIPE_V>();
        const uint16_t large[4] = {64, 64, 0, 0};
        MrgSort(sorted, MrgSortSrcList<float>(temporary, temporary[64 * 8], temporary, temporary),
                large, consumed, 3, 1);
        PipeBarrier<PIPE_V>();
        Extract(keys, index, sorted, TILE / 16);
        PipeBarrier<PIPE_V>();
        permutation.gather(temporary.ReinterpretCast<uint64_t>(), records, TILE);
        DataCopy(records, temporary.ReinterpretCast<uint64_t>(), TILE * 4);
        fence<HardEvent::V_S>();
    }
};
