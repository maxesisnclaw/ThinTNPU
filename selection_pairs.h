struct PairRecords {
    TBuf<TPosition::VECCALC> sourceBuf, scoreOffsetBuf, anchorOffsetBuf, recordOffsetBuf,
        wideOffsetBuf, inputBuf, wideBuf, rawBuf, keyBuf, tempBuf, magnitudeBuf, rankBuf, maskBuf;

    __aicore__ inline void init(TPipe& pipe) {
        pipe.InitBuffer(sourceBuf, TILE * 5 * 4);
        pipe.InitBuffer(scoreOffsetBuf, TILE * 4);
        pipe.InitBuffer(anchorOffsetBuf, TILE * 4);
        pipe.InitBuffer(recordOffsetBuf, TILE * 8 * 4);
        pipe.InitBuffer(wideOffsetBuf, TILE * 2 * 4);
        pipe.InitBuffer(inputBuf, (TILE + 16) * 2);
        pipe.InitBuffer(wideBuf, TILE * 4);
        pipe.InitBuffer(rawBuf, TILE * 4);
        pipe.InitBuffer(keyBuf, TILE * 4);
        pipe.InitBuffer(tempBuf, TILE * 4);
        pipe.InitBuffer(magnitudeBuf, TILE * 4);
        pipe.InitBuffer(rankBuf, TILE * 4);
        pipe.InitBuffer(maskBuf, 32);
        auto wide = wideOffsetBuf.Get<uint32_t>(), record = recordOffsetBuf.Get<uint32_t>();
        for (int i = 0; i < TILE; ++i) {
            wide.SetValue(i * 2, i * 2);
            wide.SetValue(i * 2 + 1, TILE * 2);
            for (int word = 0; word < 8; ++word)
                record.SetValue(i * 8 + word, (word % 2 ? 4 * TILE : word / 2 * TILE + i) * 4);
        }
        fence<HardEvent::S_V>();
        Duplicate(sourceBuf.Get<int32_t>()[4 * TILE], int32_t(0), TILE);
        Duplicate(inputBuf.Get<uint16_t>()[TILE], uint16_t(0), 16);
        CreateVecIndex(rankBuf.Get<float>(), 0.0f, TILE);
        fence<HardEvent::V_S>();
    }

    __aicore__ inline LocalTensor<uint16_t> make(LocalTensor<uint64_t> records,
            LocalTensor<uint16_t> compact, LocalTensor<uint64_t> selected,
            int64_t start, int count, int64_t classes, int64_t pitch) {
        auto scoreOffset = scoreOffsetBuf.Get<uint32_t>(), anchorOffset = anchorOffsetBuf.Get<uint32_t>();
        for (int i = 0; i < TILE; ++i) {
            int64_t rank = i < count ? (start + i) / classes : 0;
            int64_t cls = i < count ? (start + i) % classes : 0;
            scoreOffset.SetValue(i, (cls * pitch + rank) * 2);
            anchorOffset.SetValue(i, rank * 32 + 8);
        }
        auto source = sourceBuf.Get<int32_t>();
        auto input = inputBuf.Get<uint16_t>();
        auto wide = wideBuf.Get<int32_t>();
        auto raw = rawBuf.Get<float>(), key = keyBuf.Get<float>();
        auto temp = tempBuf.Get<float>(), magnitude = magnitudeBuf.Get<float>();
        auto mask = maskBuf.Get<uint8_t>();
        fence<HardEvent::S_V>();
        Gather(input, compact, scoreOffset, 0, TILE);
        Gather(source[3 * TILE], selected.ReinterpretCast<int32_t>(), anchorOffset, 0, TILE);
        PipeBarrier<PIPE_V>();
        Gather(wide.ReinterpretCast<uint16_t>(), input, wideOffsetBuf.Get<uint32_t>(), 0, TILE * 2);
        PipeBarrier<PIPE_V>();
        Cast(raw, wide, RoundMode::CAST_NONE, TILE);
        PipeBarrier<PIPE_V>();
        half_score_keys(key, raw, temp, magnitude, mask);
        CompareScalar(mask, rankBuf.Get<float>(), float(count), CMPMODE::LT, TILE);
        PipeBarrier<PIPE_V>();
        Select(key, mask, key, 65536.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE);
        Adds(temp, rankBuf.Get<float>(), float(int32_t(start)), TILE);
        PipeBarrier<PIPE_V>();
        Cast(source, key, RoundMode::CAST_RINT, TILE);
        Cast(source[TILE], temp, RoundMode::CAST_RINT, TILE);
        Cast(source[2 * TILE], raw, RoundMode::CAST_RINT, TILE);
        PipeBarrier<PIPE_V>();
        Gather(records.ReinterpretCast<uint32_t>(), source.ReinterpretCast<uint32_t>(),
               recordOffsetBuf.Get<uint32_t>(), 0, TILE * 8);
        fence<HardEvent::V_S>();
        return input;
    }
};
