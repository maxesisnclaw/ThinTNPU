__aicore__ inline void point_stage_begin() {
    for (int bank = 0; bank < 2; ++bank) {
        SetFlag<HardEvent::V_MTE2>(bank); SetFlag<HardEvent::MTE3_V>(bank);
    }
}

__aicore__ inline void point_stage_end() {
    for (int bank = 0; bank < 2; ++bank) {
        WaitFlag<HardEvent::V_MTE2>(bank); WaitFlag<HardEvent::MTE3_V>(bank);
    }
}

template<int TM, int TK>
__aicore__ inline void point_stage_chunk(LocalTensor<half> raw, LocalTensor<half> packed,
                                         LocalTensor<half> image, GlobalTensor<half> input,
                                         int pos, int ci, int n, int imageBank, int cb) {
    int count = imin(TM, n - pos), bank = (cb / TK) % 2, channels = imin(TK, ci - cb);
    auto r = raw[bank * TK * TM], p = packed[bank * TK * TM];
    WaitFlag<HardEvent::V_MTE2>(bank);
    DataCopy(r, input[cb * n + pos],
             DataCopyParams{(uint16_t)channels, (uint16_t)(count / 16),
                            (uint16_t)((n - count) / 16), (uint16_t)((TM - count) / 16)});
    SetFlag<HardEvent::MTE2_V>(bank); WaitFlag<HardEvent::MTE2_V>(bank);
    WaitFlag<HardEvent::MTE3_V>(bank);
    for (int c = 0; c < channels; c += 16)
        transpose_image(p[c * TM], r[c * TM], TM, TM);
    SetFlag<HardEvent::V_MTE2>(bank);
    SetFlag<HardEvent::V_MTE3>(bank); WaitFlag<HardEvent::V_MTE3>(bank);
    DataCopy(image[imageBank * ci * TM + cb * TM], p, channels * TM);
    SetFlag<HardEvent::MTE3_V>(bank);
}

template<int TM, int TK>
__aicore__ inline void point_stage(LocalTensor<half> raw, LocalTensor<half> packed,
                                   LocalTensor<half> image, GlobalTensor<half> input,
                                   int pos, int ci, int n, int imageBank) {
    point_stage_begin();
    for (int cb = 0; cb < ci; cb += TK)
        point_stage_chunk<TM, TK>(raw, packed, image, input, pos, ci, n, imageBank, cb);
    point_stage_end();
}

template <int TN, bool HalfOut, bool Fused, bool Wide>
__global__ __aicore__ void cube_point(GM_ADDR x, GM_ADDR packedWeight, GM_ADDR bias, GM_ADDR y,
                                     int ci, int co, int n, int weightCo, bool hasBias) {
    constexpr int TM = Wide ? 256 : 128, TK = Wide ? 64 : 128;
    TPipe pipe;
    TBuf<TPosition::VECCALC> rawBuf, packBuf, biasBuf, broadcastBuf;
    TBuf<TPosition::A1> imageBuf;
    TBuf<TPosition::B1> weightBuf;
    TBuf<TPosition::A2> aBuf;
    TBuf<TPosition::B2> bBuf;
    TBuf<TPosition::CO1> cBuf;
    pipe.InitBuffer(rawBuf, imax(2 * TK * TM * 2, TN * TM * 4));
    pipe.InitBuffer(packBuf, imax(2 * TK * TM * 2, TN * TM * 4));
    pipe.InitBuffer(biasBuf, TN * 6);
    pipe.InitBuffer(broadcastBuf, TN * TM * 4);
    pipe.InitBuffer(imageBuf, 2 * ci * TM * 2);
    pipe.InitBuffer(weightBuf, weightCo * ci * 2);
    pipe.InitBuffer(aBuf, 2 * TM * TK * 2);
    pipe.InitBuffer(bBuf, 2 * TN * TK * 2);
    pipe.InitBuffer(cBuf, TN * TM * 4);
    GlobalTensor<half> input, weights, biases, outH;
    GlobalTensor<float> outF;
    input.SetGlobalBuffer((__gm__ half *)x);
    weights.SetGlobalBuffer((__gm__ half *)packedWeight);
    biases.SetGlobalBuffer((__gm__ half *)bias);
    outH.SetGlobalBuffer((__gm__ half *)y);
    outF.SetGlobalBuffer((__gm__ float *)y);
    auto raw = rawBuf.Get<half>(), packed = packBuf.Get<half>();
    auto image = imageBuf.Get<half>(), weight = weightBuf.Get<half>();
    Duplicate(raw, (half)0, 2 * TK * TM);
    PipeBarrier<PIPE_V>();
    int tiles = (n + TM - 1) / TM;
    int groups = (co + weightCo - 1) / weightCo, cachedOc = -1, imageBank = 0;
    int channelCores = 1;
    while (channelCores * 2 <= groups && channelCores * 2 <= GetBlockNum()) channelCores *= 2;
    int spatialCores = GetBlockNum() / channelCores;
    for (int group = GetBlockIdx() % channelCores; group < groups; group += channelCores) {
        int task = group * tiles + GetBlockIdx() / channelCores;
        imageBank = 0;
        if (task >= (group + 1) * tiles) continue;
        point_stage<TM, TK>(raw, packed, image, input, task % tiles * TM, ci, n, imageBank);
        for (; task < (group + 1) * tiles; task += spatialCores, imageBank ^= 1) {
            int tile = task % tiles, ocBase = task / tiles * weightCo;
            int outputs = imin(weightCo, co - ocBase);
            int pos = tile * TM, count = imin(TM, n - pos);
            if (cachedOc != ocBase) {
                for (int oc = 0; oc < outputs; oc += TN)
                    DataCopy(weight[oc * ci], weights[(ocBase + oc) * 16],
                             DataCopyParams{(uint16_t)(ci / 16), (uint16_t)TN,
                                            (uint16_t)(((co + 63) / 64 * 64) - TN), 0});
                SetFlag<HardEvent::MTE2_MTE1>(2); WaitFlag<HardEvent::MTE2_MTE1>(2);
                cachedOc = ocBase;
            }
            SetFlag<HardEvent::MTE3_MTE1>(2);
            WaitFlag<HardEvent::MTE3_MTE1>(2);
            bool prefetch = task + spatialCores < (group + 1) * tiles;
            for (int oc = 0; oc < outputs; oc += TN) {
                if (prefetch && oc == 0) point_stage_begin();
                for (int bank = 0; bank < 2; ++bank) SetFlag<HardEvent::M_MTE1>(bank);
                for (int cb = 0; cb < ci; cb += TK) {
                    int bank = (cb / TK) % 2, channels = imin(TK, ci - cb);
                    auto a = aBuf.Get<half>()[bank * TM * TK];
                    auto b = bBuf.Get<half>()[bank * TN * TK];
                    WaitFlag<HardEvent::M_MTE1>(bank);
                    LoadData3DParamsV2<half> p;
                    p.l1H = 1; p.l1W = TM; p.channelSize = channels;
                    p.kExtension = channels; p.mExtension = TM;
                    p.filterW = 1; p.filterH = 1; p.strideW = 1; p.strideH = 1;
                    LoadData(a, image[imageBank * ci * TM + cb * TM], p);
                    LoadData2DParams ld;
                    ld.repeatTimes = channels / 16 * TN / 16;
                    ld.srcStride = 1;
                    LoadData(b, weight[oc * ci + cb * TN], ld);
                    SetFlag<HardEvent::MTE1_M>(bank);
                    WaitFlag<HardEvent::MTE1_M>(bank);
                    MmadParams mm;
                    mm.m = TM; mm.n = TN; mm.k = channels; mm.isBias = cb != 0;
                    Mmad(cBuf.Get<float>(), a, b, mm);
                    SetFlag<HardEvent::M_MTE1>(bank);
                    if (prefetch && oc == 0)
                        point_stage_chunk<TM, TK>(raw, packed, image, input, ((task + spatialCores) % tiles) * TM,
                                          ci, n, imageBank ^ 1, cb);
                }
                if (prefetch && oc == 0) point_stage_end();
                for (int bank = 0; bank < 2; ++bank) WaitFlag<HardEvent::M_MTE1>(bank);
                SetFlag<HardEvent::M_V>(2);
                WaitFlag<HardEvent::M_V>(2);
                DataCopyEnhancedParams ep;
                ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
                DataCopy(rawBuf.Get<float>(), cBuf.Get<float>(),
                         DataCopyParams{(uint16_t)(TN / 16), (uint16_t)(TM / 16), 0, 0}, ep);
                PipeBarrier<PIPE_V>();
                if constexpr (HalfOut) {
                    if (hasBias) {
                        DataCopy(biasBuf.Get<half>()[TN * 2], biases[ocBase + oc], TN);
                        SetFlag<HardEvent::MTE2_V>(2); WaitFlag<HardEvent::MTE2_V>(2);
                        Cast(biasBuf.Get<float>(), biasBuf.Get<half>()[TN * 2], RoundMode::CAST_NONE, TN);
                        PipeBarrier<PIPE_V>();
                    }
                    image_layout::finish_half<Fused>(rawBuf.Get<half>(), packBuf.Get<half>(),
                        rawBuf.Get<float>(), broadcastBuf.Get<float>(), biasBuf.Get<float>(), TM, TN, hasBias);
                    SetFlag<HardEvent::V_MTE3>(2); WaitFlag<HardEvent::V_MTE3>(2);
                    DataCopy(outH[(ocBase + oc) * n + pos], rawBuf.Get<half>(),
                             DataCopyParams{(uint16_t)TN, (uint16_t)(count / 16),
                                            (uint16_t)((TM - count) / 16), (uint16_t)((n - count) / 16)});
                } else {
                    auto trans = packBuf.Get<float>();
                    image_layout::transpose(trans, rawBuf.Get<float>(), TM, TN);
                    PipeBarrier<PIPE_V>();
                    if (hasBias) {
                        DataCopy(biasBuf.Get<half>()[TN * 2], biases[ocBase + oc], TN);
                        SetFlag<HardEvent::MTE2_V>(2);
                        WaitFlag<HardEvent::MTE2_V>(2);
                        Cast(biasBuf.Get<float>(), biasBuf.Get<half>()[TN * 2], RoundMode::CAST_NONE, TN);
                        PipeBarrier<PIPE_V>();
                        image_layout::broadcast(broadcastBuf.Get<float>(), biasBuf.Get<float>(), TM, TN);
                        PipeBarrier<PIPE_V>();
                        Add(trans, trans, broadcastBuf.Get<float>(), TN * TM);
                        PipeBarrier<PIPE_V>();
                    }
                    if constexpr (Fused) {
                        auto tmp = broadcastBuf.Get<float>();
                        Muls(tmp, trans, -1.0f, TN * TM); PipeBarrier<PIPE_V>();
                        Exp(tmp, tmp, TN * TM); PipeBarrier<PIPE_V>();
                        Adds(tmp, tmp, 1.0f, TN * TM); PipeBarrier<PIPE_V>();
                        Div(trans, trans, tmp, TN * TM); PipeBarrier<PIPE_V>();
                    }
                    SetFlag<HardEvent::V_MTE3>(2); WaitFlag<HardEvent::V_MTE3>(2);
                    DataCopy(outF[(ocBase + oc) * n + pos], trans,
                             DataCopyParams{(uint16_t)TN, (uint16_t)(count / 8),
                                            (uint16_t)((TM - count) / 8), (uint16_t)((n - count) / 8)});
                }
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
}
