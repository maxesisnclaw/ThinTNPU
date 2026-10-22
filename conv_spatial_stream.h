template<int TM, int TN, int CI, int TK, bool HalfOut, bool Fused>
__global__ __aicore__ void cube_spatial_stream(GM_ADDR x, GM_ADDR packedWeight, GM_ADDR bias, GM_ADDR y,
                                               int ci, int co, int h, int width, int stride,
                                               int oh, int ow, int maxP, bool hasBias) {
    int n = oh * ow;
    TPipe pipe;
    TBuf<TPosition::VECCALC> rawBuf, packBuf, biasBuf;
    TBuf<TPosition::A1> imageBuf;
    TBuf<TPosition::B1> weightBuf;
    TBuf<TPosition::A2> aBuf;
    TBuf<TPosition::B2> bBuf;
    TBuf<TPosition::CO1> cBuf;
    pipe.InitBuffer(rawBuf, imax(2 * 16 * maxP * 2, TN * TM * 4));
    pipe.InitBuffer(packBuf, imax(2 * 16 * maxP * 2, TN * TM * 4));
    pipe.InitBuffer(biasBuf, TN * 6);
    pipe.InitBuffer(imageBuf, (ci == CI ? 1 : 2) * CI * maxP * 2);
    pipe.InitBuffer(weightBuf, TN * ci * 9 * 2);
    pipe.InitBuffer(aBuf, 2 * TM * TK * 2);
    pipe.InitBuffer(bBuf, 2 * TN * TK * 2);
    pipe.InitBuffer(cBuf, TN * TM * 4);
    GlobalTensor<half> input, weights, biases, outH;
    GlobalTensor<float> outF;
    input.SetGlobalBuffer((__gm__ half*)x);
    weights.SetGlobalBuffer((__gm__ half*)packedWeight);
    biases.SetGlobalBuffer((__gm__ half*)bias);
    outH.SetGlobalBuffer((__gm__ half*)y);
    outF.SetGlobalBuffer((__gm__ float*)y);
    auto raw = rawBuf.Get<half>(), packed = packBuf.Get<half>();
    auto image = imageBuf.Get<half>(), weight = weightBuf.Get<half>();
    int groups = co / TN, channelCores = 1;
    while (channelCores * 2 <= groups && channelCores * 2 <= GetBlockNum()) channelCores *= 2;
    int spatialCores = GetBlockNum() / channelCores;
    int tiles = (n + TM - 1) / TM;
    for (int oc = GetBlockIdx() % channelCores * TN; oc < co; oc += channelCores * TN) {
        DataCopy(weight, weights[oc * 16],
                 DataCopyParams{(uint16_t)(ci / 16 * 9), (uint16_t)TN,
                                (uint16_t)((co + 63) / 64 * 64 - TN), 0});
        SetFlag<HardEvent::MTE2_MTE1>(2); WaitFlag<HardEvent::MTE2_MTE1>(2);
        if (hasBias) {
            DataCopy(biasBuf.Get<half>()[TN * 2], biases[oc], TN);
            SetFlag<HardEvent::MTE2_V>(2); WaitFlag<HardEvent::MTE2_V>(2);
            Cast(biasBuf.Get<float>(), biasBuf.Get<half>()[TN * 2], RoundMode::CAST_NONE, TN);
            PipeBarrier<PIPE_V>();
        }
        for (int tile = GetBlockIdx() / channelCores; tile < tiles; tile += spatialCores) {
            int pos = tile * TM, count = imin(TM, n - pos);
            int row = pos / ow, lastRow = (pos + count - 1) / ow;
            int group = 1; while (width * group % 16 != 0 || group % stride != 0) ++group;
            int begin = imax(0, row * stride - 1) / group * group;
            int end = imin(h, (lastRow * stride + 2 + group - 1) / group * group);
            SetFlag<HardEvent::MTE1_MTE3>(2); WaitFlag<HardEvent::MTE1_MTE3>(2);
            spatial_stage<TM>(raw, packed, image, input, pos, CI, h, width, stride, oh, ow, maxP, 0);
            for (int bank = 0; bank < 2; ++bank) SetFlag<HardEvent::M_MTE1>(bank);
            for (int cb = 0; cb < ci; cb += CI) {
                int imageBank = cb / CI % 2;
                SetFlag<HardEvent::MTE3_MTE1>(2); WaitFlag<HardEvent::MTE3_MTE1>(2);
                for (int k = 0; k < CI * 9; k += TK) {
                    int bank = (cb / CI * (CI * 9 / TK) + k / TK) % 2;
                    auto a = aBuf.Get<half>()[bank * TM * TK];
                    auto b = bBuf.Get<half>()[bank * TN * TK];
                    WaitFlag<HardEvent::M_MTE1>(bank);
                    LoadData3DParamsV2<half> p;
                    p.l1H = end - begin; p.l1W = width; p.channelSize = CI;
                    p.kExtension = TK; p.kStartPt = k; p.mExtension = TM;
                    p.mStartPt = (row - begin / stride) * ow + pos % ow;
                    p.filterW = 3; p.filterH = 3; p.strideW = stride; p.strideH = stride;
                    p.padList[0] = 1; p.padList[1] = 1; p.padList[2] = 1; p.padList[3] = 1;
                    LoadData(a, image[imageBank * CI * maxP], p);
                    LoadData2DParams ld; ld.repeatTimes = TK / 16 * TN / 16; ld.srcStride = 1;
                    LoadData(b, weight[(cb * 9 + k) * TN], ld);
                    SetFlag<HardEvent::MTE1_M>(bank); WaitFlag<HardEvent::MTE1_M>(bank);
                    MmadParams mm; mm.m = TM; mm.n = TN; mm.k = TK;
                    mm.isBias = cb != 0 || k != 0;
                    Mmad(cBuf.Get<float>(), a, b, mm);
                    SetFlag<HardEvent::M_MTE1>(bank);
                    if (k == 0 && cb + CI < ci) {
                        SetFlag<HardEvent::MTE1_MTE3>(2); WaitFlag<HardEvent::MTE1_MTE3>(2);
                        spatial_stage<TM>(raw, packed, image, input[(cb + CI) * h * width],
                            pos, CI, h, width, stride, oh, ow, maxP, imageBank ^ 1);
                    }
                }
            }
            for (int bank = 0; bank < 2; ++bank) WaitFlag<HardEvent::M_MTE1>(bank);
            SetFlag<HardEvent::M_V>(2); WaitFlag<HardEvent::M_V>(2);
            DataCopyEnhancedParams ep; ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
            DataCopy(rawBuf.Get<float>(), cBuf.Get<float>(),
                     DataCopyParams{TN / 16, TM / 16, 0, 0}, ep);
            PipeBarrier<PIPE_V>();
            if constexpr (HalfOut) {
                image_layout::finish_half<Fused>(raw, packed, rawBuf.Get<float>(), packBuf.Get<float>(),
                                                 biasBuf.Get<float>(), TM, TN, hasBias);
                SetFlag<HardEvent::V_MTE3>(2); WaitFlag<HardEvent::V_MTE3>(2);
                DataCopy(outH[oc * n + pos], raw,
                         DataCopyParams{TN, (uint16_t)(count / 16), (uint16_t)((TM - count) / 16),
                                        (uint16_t)((n - count) / 16)});
            } else {
                auto value = packBuf.Get<float>(), scratch = rawBuf.Get<float>();
                image_layout::transpose(value, scratch, TM, TN);
                PipeBarrier<PIPE_V>();
                if (hasBias) {
                    image_layout::broadcast(scratch, biasBuf.Get<float>(), TM, TN);
                    PipeBarrier<PIPE_V>(); Add(value, value, scratch, TN * TM); PipeBarrier<PIPE_V>();
                }
                if constexpr (Fused) {
                    Muls(scratch, value, -1.0f, TN * TM); PipeBarrier<PIPE_V>();
                    Exp(scratch, scratch, TN * TM); PipeBarrier<PIPE_V>();
                    Adds(scratch, scratch, 1.0f, TN * TM); PipeBarrier<PIPE_V>();
                    Div(value, value, scratch, TN * TM); PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::V_MTE3>(2); WaitFlag<HardEvent::V_MTE3>(2);
                DataCopy(outF[oc * n + pos], value,
                         DataCopyParams{TN, (uint16_t)(count / 8), (uint16_t)((TM - count) / 8),
                                        (uint16_t)((n - count) / 8)});
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
}
