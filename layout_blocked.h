#pragma once
template<bool ToBlocked>
__global__ __aicore__ void convert_layout(GM_ADDR x, GM_ADDR y, int channels, int n) {
    constexpr int M = 256;
    TPipe pipe;
    TBuf<TPosition::VECCALC> rawBuf, packedBuf;
    pipe.InitBuffer(rawBuf, 16 * M * 2);
    pipe.InitBuffer(packedBuf, 16 * M * 2);
    auto raw = rawBuf.Get<half>(), packed = packedBuf.Get<half>();
    GlobalTensor<half> input, output;
    input.SetGlobalBuffer((__gm__ half*)x); output.SetGlobalBuffer((__gm__ half*)y);
    int tiles = (n + M - 1) / M;
    for (int tile = GetBlockIdx(); tile < channels / 16 * tiles; tile += GetBlockNum()) {
        int cb = tile / tiles * 16, pos = tile % tiles * M, count = imin(M, n - pos);
        if constexpr (ToBlocked)
            DataCopy(raw, input[cb * n + pos],
                     DataCopyParams{16, (uint16_t)(count / 16), (uint16_t)((n - count) / 16),
                                    (uint16_t)((M - count) / 16)});
        else
            DataCopy(raw, input[cb * n + pos * 16], count * 16);
        SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
        uint64_t d[16], s[16];
        for (int i = 0; i < 16; ++i) {
            s[i] = (uint64_t)raw[i * (ToBlocked ? M : 16)].GetPhyAddr();
            d[i] = (uint64_t)packed[i * (ToBlocked ? 16 : M)].GetPhyAddr();
        }
        TransDataTo5HDParams p;
        p.repeatTimes = count / 16;
        p.dstRepStride = p.repeatTimes == 1 ? 0 : (ToBlocked ? 16 : 1);
        p.srcRepStride = p.repeatTimes == 1 ? 0 : (ToBlocked ? 1 : 16);
        TransDataTo5HD<half>(d, s, p);
        SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
        if constexpr (ToBlocked)
            DataCopy(output[cb * n + pos * 16], packed, count * 16);
        else
            DataCopy(output[cb * n + pos], packed,
                     DataCopyParams{16, (uint16_t)(count / 16), (uint16_t)((M - count) / 16),
                                    (uint16_t)((n - count) / 16)});
        PipeBarrier<PIPE_ALL>();
    }
}

at::Tensor pack(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 4 && x.size(0) == 1 && x.size(1) % 16 == 0 &&
                x.size(2) * x.size(3) % 16 == 0 && x.numel() > 0 &&
                x.is_contiguous() && x.storage_offset() == 0 && !x.requires_grad() &&
                x.scalar_type() == at::kHalf && x.device().type() == c10::DeviceType::PrivateUse1,
                "expected N=1 allocation-base FP16 NCHW with C and HW divisible by 16");
    thin::select(x.device());
    auto y = at::empty({1, x.size(1) / 16, x.size(2), x.size(3), 16}, x.options());
    convert_layout<true><<<8, nullptr, thin::stream()>>>((GM_ADDR)x.data_ptr(), (GM_ADDR)y.data_ptr(),
                                                       x.size(1), x.size(2) * x.size(3));
    thin::sync();
    return y;
}

at::Tensor unpack(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 5 && x.size(0) == 1 && x.size(4) == 16 &&
                x.size(2) * x.size(3) % 16 == 0 && x.numel() > 0 &&
                x.is_contiguous() && x.storage_offset() == 0 && !x.requires_grad() &&
                x.scalar_type() == at::kHalf && x.device().type() == c10::DeviceType::PrivateUse1,
                "expected N=1 allocation-base FP16 NC1HWC0 with HW divisible by 16");
    thin::select(x.device());
    auto y = at::empty({1, x.size(1) * 16, x.size(2), x.size(3)}, x.options());
    convert_layout<false><<<8, nullptr, thin::stream()>>>((GM_ADDR)x.data_ptr(), (GM_ADDR)y.data_ptr(),
                                                        x.size(1) * 16, x.size(2) * x.size(3));
    thin::sync();
    return y;
}
