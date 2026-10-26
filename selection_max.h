__aicore__ inline void half_score_keys(LocalTensor<float> key, LocalTensor<float> raw,
                                      LocalTensor<float> temp, LocalTensor<float> magnitude,
                                      LocalTensor<uint8_t> mask, int count = TILE) {
    Muls(temp, raw, -1.0f, count);
    PipeBarrier<PIPE_V>();
    Adds(key, temp, 32767.0f, count);
    Adds(temp, raw, -32768.0f, count);
    CompareScalar(mask, raw, 32768.0f, CMPMODE::GE, count);
    PipeBarrier<PIPE_V>();
    Select(key, mask, raw, key, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    Select(magnitude, mask, temp, raw, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask, magnitude, 0.0f, CMPMODE::NE, count);
    PipeBarrier<PIPE_V>();
    Select(key, mask, key, 32767.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask, magnitude, 31744.0f, CMPMODE::LE, count);
    PipeBarrier<PIPE_V>();
    Select(key, mask, key, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    PipeBarrier<PIPE_V>();
}

struct ClassMaximum {
    TBuf<TPosition::VECCALC> offsetBuf, wideBuf, rawBuf, keyBuf, tempBuf, magnitudeBuf,
        bestBuf, bitsBuf, labelBuf, classBuf, maskBuf;

    __aicore__ inline void init(TPipe& pipe) {
        pipe.InitBuffer(offsetBuf, TILE * 8);
        pipe.InitBuffer(wideBuf, TILE * 4);
        pipe.InitBuffer(rawBuf, TILE * 4);
        pipe.InitBuffer(keyBuf, TILE * 4);
        pipe.InitBuffer(tempBuf, TILE * 4);
        pipe.InitBuffer(magnitudeBuf, TILE * 4);
        pipe.InitBuffer(bestBuf, TILE * 4);
        pipe.InitBuffer(bitsBuf, TILE * 4);
        pipe.InitBuffer(labelBuf, TILE * 4);
        pipe.InitBuffer(classBuf, TILE * 4);
        pipe.InitBuffer(maskBuf, 32);
        auto offset = offsetBuf.Get<uint32_t>();
        for (int i = 0; i < TILE; ++i) {
            offset.SetValue(2 * i, i * 2);
            offset.SetValue(2 * i + 1, READ_HALF * 2);
        }
    }

    __aicore__ inline void run(GlobalTensor<uint16_t> x, LocalTensor<uint16_t> input,
                               LocalTensor<uint64_t> records, int64_t base,
                               int64_t start, int count, int64_t classes, int64_t channel) {
        auto offset = offsetBuf.Get<uint32_t>();
        auto wide = wideBuf.Get<int32_t>();
        auto raw = rawBuf.Get<float>(), key = keyBuf.Get<float>(), temp = tempBuf.Get<float>();
        auto magnitude = magnitudeBuf.Get<float>(), best = bestBuf.Get<float>();
        auto bits = bitsBuf.Get<float>(), labels = labelBuf.Get<float>(), cls = classBuf.Get<float>();
        auto mask = maskBuf.Get<uint8_t>();
        fence<HardEvent::S_V>();
        Duplicate(best, 65536.0f, TILE);
        Duplicate(bits, 0.0f, TILE);
        Duplicate(labels, 0.0f, TILE);
        Duplicate(input[READ_HALF], uint16_t(0), 16);
        for (int64_t c = 0; c < classes; ++c) {
            int64_t address = base + start + (c + 4) * channel;
            int skew = address % 16;
            fence<HardEvent::V_MTE2>();
            DataCopy(input, x[address - skew], (skew + count + 15) / 16 * 16);
            fence<HardEvent::MTE2_V>();
            // Zero-interleaving widens raw half bits without floating-point half conversion.
            Gather(wide.ReinterpretCast<uint16_t>(), input[skew], offset, 0, TILE * 2);
            PipeBarrier<PIPE_V>();
            Cast(raw, wide, RoundMode::CAST_NONE, TILE);
            PipeBarrier<PIPE_V>();
            half_score_keys(key, raw, temp, magnitude, mask);
            Duplicate(cls, float(int32_t(c)), TILE);
            PipeBarrier<PIPE_V>();
            Compare(mask, key, best, CMPMODE::LT, TILE);
            PipeBarrier<PIPE_V>();
            Select(best, mask, key, best, SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE);
            Select(bits, mask, raw, bits, SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE);
            Select(labels, mask, cls, labels, SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE);
            PipeBarrier<PIPE_V>();
        }
        fence<HardEvent::V_S>();
        for (int i = 0; i < count; ++i) {
            records.SetValue(i * 4, int32_t(best.GetValue(i)));
            records.SetValue(i * 4 + 2, int32_t(bits.GetValue(i)));
            records.SetValue(i * 4 + 3, int32_t(labels.GetValue(i)));
        }
    }
};
