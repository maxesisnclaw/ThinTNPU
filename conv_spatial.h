__aicore__ inline void spatial_cache(LocalTensor<half> raw, LocalTensor<half> packed,
                                     LocalTensor<half> image, GlobalTensor<half> input,
                                     int ci, int spatial, int chunk) {
    for (int bank = 0; bank < 2; ++bank) {
        SetFlag<HardEvent::V_MTE2>(bank); SetFlag<HardEvent::MTE3_V>(bank);
    }
    int sequence = 0;
    for (int cb = 0; cb < ci; cb += 16) {
        for (int off = 0; off < spatial; off += chunk, ++sequence) {
            int bank = sequence % 2, count = imin(chunk, spatial - off);
            auto r = raw[bank * 16 * chunk], p = packed[bank * 16 * chunk];
            WaitFlag<HardEvent::V_MTE2>(bank);
            DataCopy(r, input[cb * spatial + off],
                     DataCopyParams{16, (uint16_t)(count / 16), (uint16_t)((spatial - count) / 16),
                                    (uint16_t)((chunk - count) / 16)});
            SetFlag<HardEvent::MTE2_V>(bank); WaitFlag<HardEvent::MTE2_V>(bank);
            WaitFlag<HardEvent::MTE3_V>(bank);
            transpose_image(p, r, count, chunk);
            SetFlag<HardEvent::V_MTE2>(bank);
            SetFlag<HardEvent::V_MTE3>(bank); WaitFlag<HardEvent::V_MTE3>(bank);
            DataCopy(image[cb * spatial + off * 16], p, count * 16);
            SetFlag<HardEvent::MTE3_V>(bank);
        }
    }
    for (int bank = 0; bank < 2; ++bank) {
        WaitFlag<HardEvent::V_MTE2>(bank); WaitFlag<HardEvent::MTE3_V>(bank);
    }
    SetFlag<HardEvent::MTE3_MTE1>(2); WaitFlag<HardEvent::MTE3_MTE1>(2);
}

template<int TM = 112>
__aicore__ inline void spatial_stage(LocalTensor<half> raw, LocalTensor<half> packed,
                                     LocalTensor<half> image, GlobalTensor<half> input,
                                     int pos, int ci, int h, int width, int stride, int oh, int ow,
                                     int maxP, int imageBank) {
    int buffers = maxP <= 1536 ? 2 : 1;
    int count = imin(TM, oh * ow - pos), row = pos / ow, lastRow = (pos + count - 1) / ow;
    int group = 1; while (width * group % 16 != 0 || group % stride != 0) ++group;
    int begin = imax(0, row * stride - 1) / group * group;
    int end = imin(h, (lastRow * stride + 2 + group - 1) / group * group);
    int length = (end - begin) * width;
    for (int bank = 0; bank < buffers; ++bank) {
        SetFlag<HardEvent::V_MTE2>(bank);
        SetFlag<HardEvent::MTE3_V>(bank);
    }
    for (int cb = 0; cb < ci; cb += 16) {
        int bank = (cb / 16) % buffers;
        auto r = raw[bank * 16 * maxP], p = packed[bank * 16 * maxP];
        WaitFlag<HardEvent::V_MTE2>(bank);
        DataCopy(r, input[cb * h * width + begin * width],
                 DataCopyParams{16, (uint16_t)(length / 16),
                                (uint16_t)((h * width - length) / 16),
                                (uint16_t)((maxP - length) / 16)});
        SetFlag<HardEvent::MTE2_V>(bank);
        WaitFlag<HardEvent::MTE2_V>(bank);
        WaitFlag<HardEvent::MTE3_V>(bank);
        transpose_image(p, r, length, maxP);
        SetFlag<HardEvent::V_MTE2>(bank);
        SetFlag<HardEvent::V_MTE3>(bank);
        WaitFlag<HardEvent::V_MTE3>(bank);
        DataCopy(image[imageBank * ci * maxP + cb * length], p, 16 * length);
        SetFlag<HardEvent::MTE3_V>(bank);
    }
    for (int bank = 0; bank < buffers; ++bank) {
        WaitFlag<HardEvent::V_MTE2>(bank);
        WaitFlag<HardEvent::MTE3_V>(bank);
    }
}

template <int TN, bool HalfOut, bool Fused>
__global__ __aicore__ void cube_spatial(GM_ADDR x, GM_ADDR packedWeight, GM_ADDR bias, GM_ADDR y,
                                     int ci, int co, int h, int width, int stride, int oh, int ow,
                                     int maxP, int weightCo, bool hasBias, bool cacheInput) {
    // Two 112x144 FP16 operands fit the 64 KiB L0A; 128x144 does not.
    constexpr int TM = 112, TK = 144;
    const int n = oh * ow, buffers = cacheInput || maxP <= 1536 ? 2 : 1;
    TPipe pipe;
    TBuf<TPosition::VECCALC> rawBuf, packBuf, biasBuf, broadcastBuf;
    TBuf<TPosition::A1> imageBuf;
    TBuf<TPosition::B1> weightBuf;
    TBuf<TPosition::A2> aBuf;
    TBuf<TPosition::B2> bBuf;
    TBuf<TPosition::CO1> cBuf;
    pipe.InitBuffer(rawBuf, imax(buffers * 16 * maxP * 2, TN * TM * 4));
    pipe.InitBuffer(packBuf, imax(buffers * 16 * maxP * 2, TN * TM * 4));
    pipe.InitBuffer(biasBuf, TN * 6);
    pipe.InitBuffer(broadcastBuf, TN * TM * 4);
    pipe.InitBuffer(imageBuf, (cacheInput ? ci * h * width : 2 * ci * maxP) * 2);
    pipe.InitBuffer(weightBuf, weightCo * ci * 9 * 2);
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
    Duplicate(raw, (half)0, buffers * 16 * maxP);
    PipeBarrier<PIPE_V>();
    if (cacheInput) spatial_cache(raw, packed, image, input, ci, h * width, maxP);
    for (int ocBase = 0; ocBase < co; ocBase += weightCo) {
        int outputs = imin(weightCo, co - ocBase);
        for (int oc = 0; oc < outputs; oc += TN)
            DataCopy(weight[oc * ci * 9], weights[ocBase * 16 + oc * 16],
                     DataCopyParams{(uint16_t)(ci / 16 * 9), (uint16_t)TN,
                                    (uint16_t)(((co + 63) / 64 * 64) - TN), 0});
        SetFlag<HardEvent::MTE2_MTE1>(2);
        WaitFlag<HardEvent::MTE2_MTE1>(2);
        int tiles = (n + TM - 1) / TM;
        int tile = GetBlockIdx(), imageBank = 0;
        if (tile >= tiles) continue;
        if (!cacheInput) spatial_stage(raw, packed, image, input, tile * TM, ci, h, width, stride, oh, ow, maxP, imageBank);
        for (; tile < tiles; tile += GetBlockNum(), imageBank ^= 1) {
            int pos = tile * TM, count = imin(TM, n - pos);
            int row = pos / ow, lastRow = (pos + count - 1) / ow;
            int group = 1; while (width * group % 16 != 0 || group % stride != 0) ++group;
            int begin = imax(0, row * stride - 1) / group * group;
            int end = imin(h, (lastRow * stride + 2 + group - 1) / group * group);
            int plane = (end - begin) * width;
            SetFlag<HardEvent::MTE3_MTE1>(2);
            WaitFlag<HardEvent::MTE3_MTE1>(2);
            for (int oc = 0; oc < outputs; oc += TN) {
                for (int bank = 0; bank < 2; ++bank) SetFlag<HardEvent::M_MTE1>(bank);
                for (int cb = 0; cb < ci; cb += 16) {
                    int bank = (cb / 16) % 2;
                    auto a = aBuf.Get<half>()[bank * TM * TK];
                    auto b = bBuf.Get<half>()[bank * TN * TK];
                    WaitFlag<HardEvent::M_MTE1>(bank);
                    LoadData3DParamsV2<half> p;
                    p.l1H = cacheInput ? h : end - begin; p.l1W = width; p.channelSize = 16;
                    p.kExtension = TK; p.mExtension = TM;
                    p.mStartPt = cacheInput ? pos : (row - begin / stride) * ow + pos % ow;
                    p.filterW = 3; p.filterH = 3; p.strideW = stride; p.strideH = stride;
                    p.padList[0] = 1; p.padList[1] = 1; p.padList[2] = 1; p.padList[3] = 1;
                    LoadData(a, image[cacheInput ? cb * h * width : imageBank * ci * maxP + cb * plane], p);
                    LoadData2DParams ld;
                    ld.repeatTimes = 9 * TN / 16;
                    ld.srcStride = 1;
                    LoadData(b, weight[oc * ci * 9 + cb * TN * 9], ld);
                    SetFlag<HardEvent::MTE1_M>(bank);
                    WaitFlag<HardEvent::MTE1_M>(bank);
                    MmadParams mm;
                    mm.m = TM; mm.n = TN; mm.k = TK; mm.isBias = cb != 0;
                    Mmad(cBuf.Get<float>(), a, b, mm);
                    SetFlag<HardEvent::M_MTE1>(bank);
                }
                if (!cacheInput && oc + TN == outputs && tile + GetBlockNum() < tiles)
                    spatial_stage(raw, packed, image, input, (tile + GetBlockNum()) * TM, ci, h, width, stride, oh, ow, maxP, imageBank ^ 1);
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
