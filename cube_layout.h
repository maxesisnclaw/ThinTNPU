#pragma once
#include "kernel_operator.h"

namespace image_layout {
__aicore__ inline void add_bias(AscendC::LocalTensor<float> value,
                                AscendC::LocalTensor<float> bias,
                                int spatial, int channels) {
    for (int group = 0; group < channels; group += 16)
        for (int pos = 0; pos < spatial; pos += 1020) {
            int repeats = (spatial - pos < 1020 ? spatial - pos : 1020) / 4;
            int base = group * spatial + pos * 16;
            AscendC::Add(value[base], value[base], bias[group * 4], 64, repeats,
                          {1,1,1,8,8,0});
        }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void transpose(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src,
                                 int spatial, int channels) {
    uint64_t d[16], s[16];
    AscendC::TransDataTo5HDParams p;
    p.repeatTimes = spatial / 16;
    p.dstRepStride = p.repeatTimes == 1 ? 0 : 2;
    p.srcRepStride = p.repeatTimes == 1 ? 0 : 32;
    for (int group = 0; group < channels; group += 16)
        for (int half = 0; half < 16; half += 8) {
            for (int i = 0; i < 16; i++)
                s[i] = (uint64_t)src[group * spatial + i * 16 + half].GetPhyAddr();
            for (int i = 0; i < 8; i++) {
                d[2 * i] = (uint64_t)dst[(group + half + i) * spatial].GetPhyAddr();
                d[2 * i + 1] = (uint64_t)dst[(group + half + i) * spatial + 8].GetPhyAddr();
            }
            AscendC::TransDataTo5HD<float>(d, s, p);
        }
}

__aicore__ inline void broadcast(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src,
                                 int spatial, int channels) {
    uint64_t d[16], s[16];
    AscendC::TransDataTo5HDParams p;
    p.repeatTimes = spatial / 16;
    p.dstRepStride = p.repeatTimes == 1 ? 0 : 2;
    p.srcRepStride = 0;
    for (int group = 0; group < channels; group += 8) {
        for (int i = 0; i < 16; i++)
            s[i] = (uint64_t)src[group].GetPhyAddr();
        for (int i = 0; i < 8; i++) {
            d[2 * i] = (uint64_t)dst[(group + i) * spatial].GetPhyAddr();
            d[2 * i + 1] = (uint64_t)dst[(group + i) * spatial + 8].GetPhyAddr();
        }
        AscendC::TransDataTo5HD<float>(d, s, p);
    }
}

__aicore__ inline void transpose_half(AscendC::LocalTensor<half> dst,
                                      AscendC::LocalTensor<half> src, int spatial, int channels) {
    uint64_t d[16], s[16];
    AscendC::TransDataTo5HDParams p;
    p.repeatTimes = spatial / 16;
    p.dstRepStride = p.repeatTimes == 1 ? 0 : 1;
    p.srcRepStride = p.repeatTimes == 1 ? 0 : 16;
    for (int group = 0; group < channels; group += 16) {
        for (int i = 0; i < 16; ++i) {
            s[i] = (uint64_t)src[group * spatial + i * 16].GetPhyAddr();
            d[i] = (uint64_t)dst[(group + i) * spatial].GetPhyAddr();
        }
        AscendC::TransDataTo5HD<half>(d, s, p);
    }
}

template<bool Fused>
__aicore__ inline void activate(AscendC::LocalTensor<float> value,
                                   AscendC::LocalTensor<float> scratch,
                                   AscendC::LocalTensor<float> bias,
                                   int spatial, int channels, bool hasBias) {
    if (hasBias) {
        for (int group = 0; group < channels; group += 16)
            for (int pos = 0; pos < spatial; pos += 128) {
                int count = spatial - pos < 128 ? spatial - pos : 128;
                int base = group * spatial + pos * 16;
                AscendC::Add(value[base], value[base], bias[group], 16, count,
                              {1, 1, 1, 2, 2, 0});
            }
        AscendC::PipeBarrier<PIPE_V>();
    }
    if constexpr (Fused) {
        AscendC::Muls(scratch, value, -1.0f, spatial * channels);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(scratch, scratch, spatial * channels);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(scratch, scratch, 1.0f, spatial * channels);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(value, value, scratch, spatial * channels);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template<bool Fused>
__aicore__ inline void finish_half(AscendC::LocalTensor<half> output,
                                   AscendC::LocalTensor<half> packed,
                                   AscendC::LocalTensor<float> value,
                                   AscendC::LocalTensor<float> scratch,
                                   AscendC::LocalTensor<float> bias,
                                   int spatial, int channels, bool hasBias) {
    activate<Fused>(value, scratch, bias, spatial, channels, hasBias);
    AscendC::Cast(packed, value, AscendC::RoundMode::CAST_NONE, spatial * channels);
    AscendC::PipeBarrier<PIPE_V>();
    transpose_half(output, packed, spatial, channels);
    AscendC::PipeBarrier<PIPE_V>();
}
}
