struct ClassReduction {
    static constexpr int CLASSES = 16;
    static constexpr int ELEMENTS = CLASSES * TILE;
    TBuf<TPosition::VECCALC> inputBuf, offsetBuf, wideBuf, rawBuf, keyBuf, tempBuf,
        magnitudeBuf, classBuf, maskBuf, bestBuf, bitsBuf, labelBuf;

    __aicore__ inline void init(TPipe& pipe) {
        pipe.InitBuffer(inputBuf, (ELEMENTS + 16) * 2);
        pipe.InitBuffer(offsetBuf, ELEMENTS * 2 * 4);
        pipe.InitBuffer(wideBuf, ELEMENTS * 4);
        pipe.InitBuffer(rawBuf, ELEMENTS * 4);
        pipe.InitBuffer(keyBuf, ELEMENTS * 4);
        pipe.InitBuffer(tempBuf, ELEMENTS * 4);
        pipe.InitBuffer(magnitudeBuf, ELEMENTS * 4);
        pipe.InitBuffer(classBuf, ELEMENTS * 4);
        pipe.InitBuffer(maskBuf, ELEMENTS / 8);
        pipe.InitBuffer(bestBuf, TILE * 4);
        pipe.InitBuffer(bitsBuf, TILE * 4);
        pipe.InitBuffer(labelBuf, TILE * 4);
        auto offsets = offsetBuf.Get<uint32_t>();
        for (int i = 0; i < ELEMENTS; ++i) {
            offsets.SetValue(i * 2, i * 2);
            offsets.SetValue(i * 2 + 1, ELEMENTS * 2);
        }
        auto cls = classBuf.Get<float>(), temp = tempBuf.Get<float>();
        fence<HardEvent::S_V>();
        CreateVecIndex(temp, 0.0f, ELEMENTS);
        PipeBarrier<PIPE_V>();
        Muls(temp, temp, 1.0f / TILE, ELEMENTS);
        PipeBarrier<PIPE_V>();
        Cast(wideBuf.Get<int32_t>(), temp, RoundMode::CAST_FLOOR, ELEMENTS);
        PipeBarrier<PIPE_V>();
        Cast(cls, wideBuf.Get<int32_t>(), RoundMode::CAST_NONE, ELEMENTS);
        fence<HardEvent::V_S>();
    }

    __aicore__ inline void run(GlobalTensor<uint16_t> x, LocalTensor<uint64_t> records,
                               int64_t base, int64_t start, int count, int64_t classes, int64_t channel) {
        auto input = inputBuf.Get<uint16_t>();
        auto wide = wideBuf.Get<int32_t>();
        auto raw = rawBuf.Get<float>(), key = keyBuf.Get<float>(), temp = tempBuf.Get<float>();
        auto magnitude = magnitudeBuf.Get<float>(), cls = classBuf.Get<float>();
        auto best = bestBuf.Get<float>(), bits = bitsBuf.Get<float>(), labels = labelBuf.Get<float>();
        auto mask = maskBuf.Get<uint8_t>();
        fence<HardEvent::S_V>();
        Duplicate(best, 65536.0f, TILE);
        Duplicate(bits, 0.0f, TILE);
        Duplicate(labels, 0.0f, TILE);
        Duplicate(input[ELEMENTS], uint16_t(0), 16);
        int padded = (count + 15) / 16 * 16;
        for (int64_t c = 0; c < classes; c += CLASSES) {
            int active = classes - c < CLASSES ? classes - c : CLASSES;
            fence<HardEvent::V_MTE2>();
            DataCopyParams copy{uint16_t(active), uint16_t(padded / 16),
                uint16_t((channel - padded) / 16), uint16_t((TILE - padded) / 16)};
            DataCopy(input, x[base + start + (c + 4) * channel], copy);
            fence<HardEvent::MTE2_V>();
            Gather(wide.ReinterpretCast<uint16_t>(), input, offsetBuf.Get<uint32_t>(), 0, ELEMENTS * 2);
            PipeBarrier<PIPE_V>();
            Cast(raw, wide, RoundMode::CAST_NONE, ELEMENTS);
            PipeBarrier<PIPE_V>();
            half_score_keys(key, raw, temp, magnitude, mask, ELEMENTS);
            Muls(key, key, float(CLASSES), ELEMENTS);
            PipeBarrier<PIPE_V>();
            Add(key, key, cls, ELEMENTS);
            CompareScalar(mask, cls, float(active), CMPMODE::LT, ELEMENTS);
            PipeBarrier<PIPE_V>();
            Select(key, mask, key, float(65536 * CLASSES), SELMODE::VSEL_TENSOR_SCALAR_MODE, ELEMENTS);
            PipeBarrier<PIPE_V>();
            for (int width = CLASSES / 2; width; width /= 2) {
                Compare(mask, key[width * TILE], key, CMPMODE::LT, width * TILE);
                PipeBarrier<PIPE_V>();
                Select(key, mask, key[width * TILE], key, SELMODE::VSEL_TENSOR_TENSOR_MODE, width * TILE);
                Select(raw, mask, raw[width * TILE], raw, SELMODE::VSEL_TENSOR_TENSOR_MODE, width * TILE);
                PipeBarrier<PIPE_V>();
            }
            Muls(temp, key, 1.0f / CLASSES, TILE);
            PipeBarrier<PIPE_V>();
            Cast(wide, temp, RoundMode::CAST_FLOOR, TILE);
            PipeBarrier<PIPE_V>();
            Cast(temp, wide, RoundMode::CAST_NONE, TILE);
            PipeBarrier<PIPE_V>();
            Muls(magnitude, temp, float(CLASSES), TILE);
            PipeBarrier<PIPE_V>();
            Sub(magnitude, key, magnitude, TILE);
            PipeBarrier<PIPE_V>();
            Adds(magnitude, magnitude, float(int32_t(c)), TILE);
            Compare(mask, temp, best, CMPMODE::LT, TILE);
            PipeBarrier<PIPE_V>();
            Select(best, mask, temp, best, SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE);
            Select(bits, mask, raw, bits, SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE);
            Select(labels, mask, magnitude, labels, SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE);
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
