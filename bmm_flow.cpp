#include "kernel_operator.h"
#include <acl/acl_rt.h>
#include <torch/all.h>
#include <torch/library.h>
#include <limits>

using namespace AscendC;
namespace thin {
aclrtStream stream();
void sync();
void select(c10::Device);
}

namespace dynamic_bmm {
constexpr int TILE = 512;
constexpr int LEFT_CAPACITY = 16384;
constexpr int RIGHT_CAPACITY = 65536;
constexpr int ROWS = 32;

struct Matrix {
    int64_t batch, row, column, row_inner, column_inner;
    int row_block, column_block;
    int64_t skew;
    int row_origin = 0, column_origin = 0, row_group = 0, column_group = 0;
};

__aicore__ inline int64_t offset(Matrix d, int r, int c) {
    r += d.row_origin;
    c += d.column_origin;
    return d.skew + int64_t(r / d.row_block) * d.row + (r % d.row_block) * d.row_inner
        + int64_t(c / d.column_block) * d.column + (c % d.column_block) * d.column_inner;
}

__global__ __aicore__ void multiply(GM_ADDR ap, GM_ADDR bp, GM_ADDR yp,
                                    int batches, int m, int k, int n, int np, int width, bool sliced_right, Matrix ad, Matrix bd) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> leftBuf, rightBuf, halfBuf, rowBuf, accumBuf, indexBuf, baseBuf;
    pipe.InitBuffer(leftBuf, LEFT_CAPACITY * 2);
    pipe.InitBuffer(rightBuf, RIGHT_CAPACITY * 2);
    pipe.InitBuffer(halfBuf, TILE * 2);
    pipe.InitBuffer(rowBuf, TILE * 4);
    pipe.InitBuffer(accumBuf, ROWS * TILE * 4);
    pipe.InitBuffer(indexBuf, TILE * 4);
    pipe.InitBuffer(baseBuf, TILE * 4);
    auto left = leftBuf.Get<half>(), right = rightBuf.Get<half>(), halves = halfBuf.Get<half>();
    auto row = rowBuf.Get<float>(), accum = accumBuf.Get<float>();
    auto indices = indexBuf.Get<int32_t>(), columns = baseBuf.Get<int32_t>();
    GlobalTensor<half> a, b, y;
    a.SetGlobalBuffer((__gm__ half*)ap);
    b.SetGlobalBuffer((__gm__ half*)bp);
    y.SetGlobalBuffer((__gm__ half*)yp);
    int tiles = (n + width - 1) / width;
    int groups = (m + ROWS - 1) / ROWS;
    int64_t jobs = int64_t(batches) * groups * tiles;
    for (int64_t job = GetBlockIdx(); job < jobs; job += GetBlockNum()) {
        int start = (job % tiles) * width;
        int r = (job / tiles) % groups * ROWS;
        int rows = m - r < ROWS ? m - r : ROWS;
        int batch = job / (int64_t(tiles) * groups);
        auto aa = ad, bb = bd;
        aa.row_origin += batch * aa.row_group;
        aa.column_origin += batch * aa.column_group;
        bb.row_origin += batch * bb.row_group;
        bb.column_origin += batch * bb.column_group;
        int count = n - start < width ? n - start : width;
        int padded = (count + 15) / 16 * 16;
        int64_t alo = offset(aa, r, 0) + batch * ad.batch;
        int64_t abase = alo / 16 * 16;
        int64_t aspan = (alo - abase + offset(aa, r + rows - 1, k - 1) - offset(aa, r, 0) + 16) / 16 * 16;
        bool cached_left = aspan <= LEFT_CAPACITY;
        if (cached_left) DataCopy(left, a[abase], aspan);
        int64_t blo = offset(bb, 0, start) + batch * bd.batch;
        int64_t bspan = (blo % 16 + offset(bb, k - 1, start + count - 1) - offset(bb, 0, start) + 16) / 16 * 16;
        int64_t bbase = blo / 16 * 16;
        bool cached_right = bspan <= RIGHT_CAPACITY;
        if (cached_right && !sliced_right) DataCopy(right, b[bbase], bspan);
        for (int j = 0; j < padded; ++j) {
            int c = start + (j < count ? j : count - 1);
            columns.SetValue(j, sliced_right ? (j < count ? j : count - 1) * 32
                                             : int32_t((offset(bb, 0, c) - offset(bb, 0, start)) * 2));
        }
        Duplicate(accum, 0.0f, ROWS * TILE);
        PipeBarrier<PIPE_ALL>();
        for (int reduction = 0; reduction < k; ++reduction) {
            if (sliced_right) {
                if (reduction % 16 == 0) {
                    SetFlag<HardEvent::V_MTE2>(0);
                    WaitFlag<HardEvent::V_MTE2>(0);
                    int64_t bi = offset(bb, reduction, start) + batch * bd.batch;
                    DataCopy(right, b[bi], {uint16_t(count), 1, uint16_t(bd.column / 16 - 1), 0});
                    SetFlag<HardEvent::MTE2_V>(0);
                    WaitFlag<HardEvent::MTE2_V>(0);
                }
                Adds(indices, columns, int32_t(reduction % 16 * 2), padded);
                PipeBarrier<PIPE_V>();
            } else if (cached_right) {
                int32_t shift = int32_t((offset(bb, reduction, start) + batch * bd.batch - bbase) * 2);
                Adds(indices, columns, shift, padded);
                PipeBarrier<PIPE_V>();
            } else {
                PipeBarrier<PIPE_ALL>();
                for (int j = 0; j < padded; ++j) {
                    int c = start + (j < count ? j : count - 1);
                    int64_t bi = offset(bb, reduction, c) + batch * bd.batch;
                    DataCopy(right[j * 16], b[bi / 16 * 16], 16);
                    indices.SetValue(j, (j * 16 + bi % 16) * 2);
                }
                PipeBarrier<PIPE_ALL>();
            }
            Gather(halves, right, indices.ReinterpretCast<uint32_t>(), 0, padded);
            PipeBarrier<PIPE_V>();
            Cast(row, halves, RoundMode::CAST_NONE, padded);
            PipeBarrier<PIPE_V>();
            for (int rr = 0; rr < rows; ++rr) {
                int64_t left_index = offset(aa, r + rr, reduction) + batch * ad.batch;
                if (!cached_left) {
                    PipeBarrier<PIPE_ALL>();
                    DataCopy(left, a[left_index / 16 * 16], 16);
                    PipeBarrier<PIPE_ALL>();
                }
                float scalar = static_cast<float>(left.GetValue(cached_left ? left_index - abase : left_index % 16));
                // FP16 products are exact in FP32; Axpy keeps the sequential reduction's rounding.
                Axpy(accum[rr * TILE], row, scalar, padded);
            }
            PipeBarrier<PIPE_V>();
        }
        for (int rr = 0; rr < rows; ++rr) {
            Cast(halves, accum[rr * TILE], RoundMode::CAST_NONE, padded);
            PipeBarrier<PIPE_ALL>();
            DataCopy(y[(int64_t(batch) * m + r + rr) * np + start], halves, padded);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

__global__ __aicore__ void pack_channels(GM_ADDR xp, GM_ADDR yp, int channels, int tokens, int pitch) {
    constexpr int pixels = 32;
    TPipe pipe;
    TBuf<TPosition::VECCALC> inputBuf, outputBuf, indexBuf;
    pipe.InitBuffer(inputBuf, 16 * pixels * 2);
    pipe.InitBuffer(outputBuf, 16 * pixels * 2);
    pipe.InitBuffer(indexBuf, 16 * pixels * 4);
    auto input = inputBuf.Get<half>(), output = outputBuf.Get<half>();
    auto indices = indexBuf.Get<uint32_t>();
    GlobalTensor<half> x, y;
    x.SetGlobalBuffer((__gm__ half*)xp);
    y.SetGlobalBuffer((__gm__ half*)yp);
    int tiles = (tokens + pixels - 1) / pixels;
    for (int j = 0; j < pixels * 16; ++j)
        indices.SetValue(j, ((j % 16) * pixels + j / 16) * 2);
    PipeBarrier<PIPE_ALL>();
    for (int job = GetBlockIdx(); job < channels / 16 * tiles; job += GetBlockNum()) {
        int start = job % tiles * pixels;
        int count = tokens - start < pixels ? tokens - start : pixels;
        int padded = (count + 15) / 16 * 16;
        for (int c = 0; c < 16; ++c)
            DataCopy(input[c * pixels], x[(job / tiles * 16 + c) * pitch + start], padded);
        PipeBarrier<PIPE_ALL>();
        Gather(output, input, indices, 0, count * 16);
        PipeBarrier<PIPE_ALL>();
        DataCopy(y[(job / tiles * tokens + start) * 16], output, count * 16);
        PipeBarrier<PIPE_ALL>();
    }
}

Matrix matrix(const at::Tensor& x, bool transpose, int groups, int origin) {
    Matrix d{};
    d.batch = x.stride(0);
    d.row_block = x.dim() == 4 ? x.size(3) : 1;
    d.column_block = 1;
    d.row = x.stride(1);
    d.column = x.stride(2);
    d.row_inner = x.dim() == 4 ? x.stride(3) : 0;
    d.row_origin = origin;
    if (groups > 0 && x.dim() == 4) {
        d.batch = 0;
        d.row_group = x.size(1) * x.size(3) / groups;
    }
    if (transpose) {
        std::swap(d.row, d.column);
        std::swap(d.row_inner, d.column_inner);
        std::swap(d.row_block, d.column_block);
        std::swap(d.row_origin, d.column_origin);
        std::swap(d.row_group, d.column_group);
    }
    d.skew = reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 32 / 2;
    return d;
}

__global__ __aicore__ void channel_select_kernel(GM_ADDR xp, GM_ADDR yp, int channels,
                                                 int tokens, int group_channels, int begin, int selected) {
    constexpr int pixels = 64;
    TPipe pipe;
    TBuf<TPosition::VECCALC> inputBuf, outputBuf, indexBuf;
    pipe.InitBuffer(inputBuf, 16 * pixels * 16 * 2);
    pipe.InitBuffer(outputBuf, pixels * 16 * 2);
    pipe.InitBuffer(indexBuf, pixels * 16 * 4);
    auto input = inputBuf.Get<half>(), output = outputBuf.Get<half>();
    auto indices = indexBuf.Get<uint32_t>();
    GlobalTensor<half> x, y;
    x.SetGlobalBuffer((__gm__ half*)xp);
    y.SetGlobalBuffer((__gm__ half*)yp);
    int tiles = (tokens + pixels - 1) / pixels;
    for (int job = GetBlockIdx(); job < channels / 16 * tiles; job += GetBlockNum()) {
        int cb = job / tiles, start = job % tiles * pixels;
        int count = tokens - start < pixels ? tokens - start : pixels;
        int source = (cb * 16 / selected) * group_channels + begin + cb * 16 % selected;
        bool aligned = selected % 16 == 0 && source % 16 == 0;
        if (aligned) {
            DataCopy(input, x[(source / 16 * tokens + start) * 16], count * 16);
        } else {
            for (int lane = 0; lane < 16; ++lane) {
                int c = cb * 16 + lane;
                int sc = (c / selected) * group_channels + begin + c % selected;
                DataCopy(input[lane * pixels * 16], x[(sc / 16 * tokens + start) * 16], count * 16);
                for (int p = 0; p < count; ++p)
                    indices.SetValue(p * 16 + lane, (lane * pixels * 16 + p * 16 + sc % 16) * 2);
            }
        }
        PipeBarrier<PIPE_ALL>();
        if (!aligned) Gather(output, input, indices, 0, count * 16);
        PipeBarrier<PIPE_ALL>();
        DataCopy(y[(cb * tokens + start) * 16], aligned ? input : output, count * 16);
        PipeBarrier<PIPE_ALL>();
    }
}

at::Tensor channel_select(const at::Tensor& x, int64_t groups, int64_t begin, int64_t count) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && x.scalar_type() == at::kHalf
                && x.dim() == 4 && x.size(0) == 1 && x.size(3) == 16 && x.is_contiguous()
                && x.storage_offset() % 16 == 0 && !x.requires_grad(), "channel_select expects aligned N=1 FP16 blocked input");
    int64_t channels = x.size(1) * 16, tokens = x.size(2);
    TORCH_CHECK(groups > 0 && channels % groups == 0 && begin >= 0 && count > 0
                && begin + count <= channels / groups && groups * count % 16 == 0,
                "channel_select requires valid equal groups and C0-aligned total output channels");
    thin::select(x.device());
    auto output = at::empty({1, groups * count / 16, tokens, 16}, x.options());
    channel_select_kernel<<<8, nullptr, thin::stream()>>>((GM_ADDR)x.data_ptr(), (GM_ADDR)output.data_ptr(),
        groups * count, tokens, channels / groups, begin, count);
    thin::sync();
    return output;
}

at::Tensor run(const at::Tensor& a, const at::Tensor& b, bool ta, bool tb, bool blocked,
               int64_t groups = 0, int64_t a_offset = 0, int64_t a_channels = -1,
               int64_t b_offset = 0, int64_t b_channels = -1) {
    TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1 && b.device() == a.device()
                && a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf
                && (a.dim() == 3 || a.dim() == 4) && (b.dim() == 3 || b.dim() == 4)
                && !a.requires_grad() && !b.requires_grad(),
                "bmm requires inference-only same-device FP16 rank-3 or channel-blocked rank-4 tensors");
    int64_t ar = a.size(1) * (a.dim() == 4 ? a.size(3) : 1), ac = a.size(2);
    int64_t br = b.size(1) * (b.dim() == 4 ? b.size(3) : 1), bc = b.size(2);
    TORCH_CHECK(groups >= 0 && a_offset >= 0 && b_offset >= 0, "invalid channel grouping");
    int64_t abatch = a.size(0), bbatch = b.size(0);
    auto grouped = [&](const at::Tensor& x, int64_t& rows, int64_t& batches) {
        if (groups && x.dim() == 4) {
            TORCH_CHECK(x.size(0) == 1 && rows % groups == 0, "grouped blocked operand requires N=1 and divisible channels");
            rows /= groups;
            batches = groups;
        }
    };
    grouped(a, ar, abatch);
    grouped(b, br, bbatch);
    if (a_channels < 0) a_channels = ar - a_offset;
    if (b_channels < 0) b_channels = br - b_offset;
    TORCH_CHECK(a_channels > 0 && b_channels > 0 && a_offset + a_channels <= ar && b_offset + b_channels <= br,
                "channel selection exceeds its group");
    ar = a_channels;
    br = b_channels;
    int64_t batch = abatch, m = ta ? ac : ar, k = ta ? ar : ac;
    int64_t bk = tb ? bc : br, n = tb ? br : bc;
    TORCH_CHECK(batch == bbatch && k == bk && batch > 0 && m > 0 && k > 0 && n > 0,
                "bmm dimensions do not match or are empty");
    int64_t np = (n + 15) / 16 * 16;
    TORCH_CHECK(batch <= std::numeric_limits<int>::max() && m <= std::numeric_limits<int>::max()
                && k <= std::numeric_limits<int>::max() && np <= std::numeric_limits<int>::max(),
                "bmm dimensions exceed kernel index range");
    TORCH_CHECK(!blocked || (groups ? batch * m : m) % 16 == 0, "blocked output requires C0-aligned channels");
    thin::select(a.device());
    auto output = at::empty({batch, m, np}, a.options());
    auto ap = reinterpret_cast<GM_ADDR>(reinterpret_cast<uintptr_t>(a.const_data_ptr()) / 32 * 32);
    auto bp = reinterpret_cast<GM_ADDR>(reinterpret_cast<uintptr_t>(b.const_data_ptr()) / 32 * 32);
    auto ad = matrix(a, ta, groups, a_offset), bd = matrix(b, tb, groups, b_offset);
    int64_t bspan = (k / bd.row_block) * bd.row + bd.row_block * bd.row_inner
        + (n / bd.column_block) * bd.column + bd.column_block * bd.column_inner;
    bool sliced_right = bd.column_block == 1 && bd.column % 16 == 0
        && bd.column >= 16 && bd.column / 16 - 1 <= 65535
        && ((bd.row_block == 16 && bd.row_inner == 1) || (bd.row_block == 1 && bd.row == 1))
        && bd.skew == 0 && bd.batch % 16 == 0 && k % 16 == 0;
    sliced_right = sliced_right && bd.row_origin % 16 == 0 && bd.column_origin % 16 == 0
        && bd.row_group % 16 == 0 && bd.column_group % 16 == 0;
    int width = sliced_right || bspan <= RIGHT_CAPACITY - 16 ? TILE : 128;
    multiply<<<8, nullptr, thin::stream()>>>(ap, bp, (GM_ADDR)output.data_ptr(), batch, m, k, n, np,
                                            width, sliced_right, ad, bd);
    thin::sync();
    if (blocked) {
        auto result = groups ? at::empty({1, batch * m / 16, n, 16}, a.options())
                             : at::empty({batch, m / 16, n, 16}, a.options());
        pack_channels<<<8, nullptr, thin::stream()>>>((GM_ADDR)output.data_ptr(),
            (GM_ADDR)result.data_ptr(), batch * m, n, np);
        thin::sync();
        return result;
    }
    return np == n ? output : output.slice(2, 0, n);
}

at::Tensor bmm(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.dim() == 3 && b.dim() == 3, "bmm requires rank-3 operands");
    return run(a, b, false, false, false);
}
}

TORCH_LIBRARY_FRAGMENT(thin_native_flow, m) {
    m.def("bmm(Tensor a, Tensor b) -> Tensor");
    m.def("layout_bmm(Tensor a, Tensor b, bool transpose_a=False, bool transpose_b=False, bool blocked_output=False, int groups=0, int a_offset=0, int a_channels=-1, int b_offset=0, int b_channels=-1) -> Tensor");
    m.def("channel_select(Tensor x, int groups, int begin, int count) -> Tensor");
}
TORCH_LIBRARY_IMPL(thin_native_flow, PrivateUse1, m) {
    m.impl("bmm", TORCH_FN(dynamic_bmm::bmm));
    m.impl("layout_bmm", TORCH_FN(dynamic_bmm::run));
    m.impl("channel_select", TORCH_FN(dynamic_bmm::channel_select));
}
TORCH_LIBRARY_IMPL(thin_native_flow, AutogradPrivateUse1, m) {
    m.impl("bmm", torch::CppFunction::makeFallthrough());
    m.impl("layout_bmm", torch::CppFunction::makeFallthrough());
    m.impl("channel_select", torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("bmm", TORCH_FN(dynamic_bmm::bmm));
}
