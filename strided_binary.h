namespace strided_binary {
using namespace AscendC;
constexpr uint32_t tile = 2048;
constexpr uint32_t row_expansion = 16;

template<class T>
__aicore__ inline void load(LocalTensor<T> out, LocalTensor<T> scratch,
                            LocalTensor<uint32_t> indices, GlobalTensor<T> source,
                            uint64_t offset, uint64_t stride, uint32_t count) {
  constexpr uint32_t align = 32 / sizeof(T);
  uint64_t base = offset / align * align;
  uint32_t skew = offset - base;
  uint32_t span = stride ? skew + (count - 1) * stride + 1 : skew + 1;
  uint32_t copied = (span + align - 1) / align * align;
  if (stride == 1 && !skew) {
    DataCopy(out, source[base], count);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    return;
  }
  DataCopy(scratch, source[base], copied);
  if (!stride) {
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);
    T value = scratch.GetValue(skew);
    Duplicate(out, value, count);
  } else {
    for (uint32_t i = 0; i < 8; ++i)
      indices.SetValue(i, (skew + i * stride) * sizeof(T));
    SetFlag<HardEvent::S_V>(0);
    WaitFlag<HardEvent::S_V>(0);
    auto signed_indices = indices.ReinterpretCast<int32_t>();
    for (uint32_t built = 8; built < count;) {
      uint32_t n = count - built < built ? count - built : built;
      Adds(signed_indices[built], signed_indices, int32_t(built * stride * sizeof(T)), n);
      PipeBarrier<PIPE_V>();
      built += n;
    }
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    Gather(out, scratch, indices, 0, count);
  }
  PipeBarrier<PIPE_V>();
  SetFlag<HardEvent::V_MTE2>(0);
  WaitFlag<HardEvent::V_MTE2>(0);
}

template<class T>
__aicore__ inline void load_rows(LocalTensor<T> out, LocalTensor<T> scratch,
                                 LocalTensor<uint32_t> indices, GlobalTensor<T> source,
                                 uint64_t offset, uint64_t row_stride, uint64_t stride,
                                 uint32_t lanes, uint32_t count) {
  if ((!row_stride && !stride) || (row_stride == lanes && stride == 1)) {
    load(out, scratch, indices, source, offset, stride, count);
    return;
  }
  constexpr uint32_t align = 32 / sizeof(T);
  uint64_t base = offset / align * align;
  uint32_t skew = offset - base;
  uint32_t span = skew + (count / lanes - 1) * row_stride + (lanes - 1) * stride + 1;
  DataCopy(scratch, source[base], (span + align - 1) / align * align);
  for (uint32_t i = 0; i < 8; ++i)
    indices.SetValue(i, (skew + i / lanes * row_stride + i % lanes * stride) * sizeof(T));
  SetFlag<HardEvent::S_V>(0);
  WaitFlag<HardEvent::S_V>(0);
  auto signed_indices = indices.ReinterpretCast<int32_t>();
  for (uint32_t built = 8; built < count;) {
    uint32_t n = count - built < built ? count - built : built;
    Adds(signed_indices[built], signed_indices, int32_t(built / lanes * row_stride * sizeof(T)), n);
    PipeBarrier<PIPE_V>();
    built += n;
  }
  SetFlag<HardEvent::MTE2_V>(0);
  WaitFlag<HardEvent::MTE2_V>(0);
  Gather(out, scratch, indices, 0, count);
  PipeBarrier<PIPE_V>();
  SetFlag<HardEvent::V_MTE2>(0);
  WaitFlag<HardEvent::V_MTE2>(0);
}

template<class T, int Op, bool Rows = false>
__global__ __aicore__ void kernel(GM_ADDR xp, GM_ADDR yp, GM_ADDR zp, GM_ADDR metadata,
                                 uint64_t length, uint64_t inner, uint32_t rank,
                                 uint64_t xskew, uint64_t yskew, float alpha, uint32_t lanes) {
  TPipe pipe;
  TBuf<QuePosition::VECCALC> scratch_buffer, left_buffer, right_buffer, output_buffer,
      index_buffer, shape_buffer;
  pipe.InitBuffer(scratch_buffer, (tile * (Rows ? row_expansion : 2) + 32) * sizeof(T));
  pipe.InitBuffer(left_buffer, tile * sizeof(T));
  pipe.InitBuffer(right_buffer, tile * sizeof(T));
  pipe.InitBuffer(output_buffer, tile * sizeof(T));
  pipe.InitBuffer(index_buffer, tile * sizeof(uint32_t));
  pipe.InitBuffer(shape_buffer, (rank * 24 + 31) / 32 * 32);
  auto scratch = scratch_buffer.Get<T>();
  auto a = left_buffer.Get<T>();
  auto b = right_buffer.Get<T>();
  auto out = output_buffer.Get<T>();
  auto indices = index_buffer.Get<uint32_t>();
  auto meta = shape_buffer.Get<uint64_t>();
  GlobalTensor<T> x, y, z;
  GlobalTensor<uint64_t> mg;
  x.SetGlobalBuffer((__gm__ T*)xp);
  y.SetGlobalBuffer((__gm__ T*)yp);
  z.SetGlobalBuffer((__gm__ T*)zp);
  mg.SetGlobalBuffer((__gm__ uint64_t*)metadata);
  DataCopy(meta, mg, (rank * 3 + 3) / 4 * 4);
  SetFlag<HardEvent::MTE2_S>(0);
  WaitFlag<HardEvent::MTE2_S>(0);
  uint64_t xt = meta.GetValue((rank - 1) * 3 + 1);
  uint64_t yt = meta.GetValue((rank - 1) * 3 + 2);
  uint64_t xr = 0, yr = 0;
  if constexpr (Rows) {
    xr = meta.GetValue((rank - 2) * 3 + 1);
    yr = meta.GetValue((rank - 2) * 3 + 2);
  }
  uint64_t tiles = (inner + tile - 1) / tile;
  for (uint64_t task = GetBlockIdx(); task < length / inner * tiles; task += GetBlockNum()) {
    uint64_t row = task / tiles;
    uint64_t start = task % tiles * tile;
    uint32_t count = inner - start < tile ? inner - start : tile;
    uint64_t xo = xskew + (Rows ? start / lanes * xr : start * xt);
    uint64_t yo = yskew + (Rows ? start / lanes * yr : start * yt);
    for (uint32_t d = rank - (Rows ? 2 : 1); d > 0; --d) {
      uint64_t size = meta.GetValue((d - 1) * 3);
      uint64_t coord = row % size;
      row /= size;
      xo += coord * meta.GetValue((d - 1) * 3 + 1);
      yo += coord * meta.GetValue((d - 1) * 3 + 2);
    }
    if constexpr (Rows) {
      load_rows(a, scratch, indices, x, xo, xr, xt, lanes, count);
      load_rows(b, scratch, indices, y, yo, yr, yt, lanes, count);
    } else {
      load(a, scratch, indices, x, xo, xt, count);
      load(b, scratch, indices, y, yo, yt, count);
    }
    if constexpr (Op == 0 || Op == 3) {
      if (alpha != 1.0f) {
        Muls(b, b, static_cast<T>(alpha), count);
        PipeBarrier<PIPE_V>();
      }
    }
    if constexpr (Op == 0) Sub(out, a, b, count);
    if constexpr (Op == 1) Mul(out, a, b, count);
    if constexpr (Op == 2) Div(out, a, b, count);
    if constexpr (Op == 3) Add(out, a, b, count);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    DataCopy(z[task / tiles * inner + start], out, count);
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
  }
}

std::vector<uint64_t> coalesced(const at::Tensor& x, const at::Tensor& y) {
  std::vector<uint64_t> dimensions;
  for (int64_t d = 0; d < x.dim(); ++d) {
    if (x.size(d) == 1) continue;
    auto n = dimensions.size();
    if (n && dimensions[n - 2] == uint64_t(x.size(d) * x.stride(d)) &&
             dimensions[n - 1] == uint64_t(y.size(d) * y.stride(d))) {
      dimensions[n - 3] *= x.size(d);
      dimensions[n - 2] = x.stride(d);
      dimensions[n - 1] = y.stride(d);
    } else {
      dimensions.push_back(x.size(d));
      dimensions.push_back(x.stride(d));
      dimensions.push_back(y.stride(d));
    }
  }
  return dimensions;
}

bool short_rows_supported(const std::vector<uint64_t>& dims, int64_t align) {
  if (dims.size() < 6) return false;
  auto n = dims.size();
  uint64_t lanes = dims[n - 3];
  return lanes <= 8 && 8 % lanes == 0 && dims[n - 6] * lanes % align == 0 &&
      dims[n - 2] <= 2 && dims[n - 1] <= 2 &&
      dims[n - 5] <= row_expansion * lanes && dims[n - 4] <= row_expansion * lanes;
}

bool supported(const at::Tensor& x, const at::Tensor& y) {
  auto dims = coalesced(x, y);
  if (dims.empty()) return false;
  auto align = 32 / x.element_size();
  auto n = dims.size();
  return (dims[n - 3] % align == 0 && dims[n - 2] <= 2 && dims[n - 1] <= 2) ||
      short_rows_supported(dims, align);
}

template<class T>
void launch(const at::Tensor& x, const at::Tensor& y, at::Tensor& out,
            const char* op, float alpha) {
  auto dimensions = coalesced(x, y);
  auto cpu = at::empty({int64_t(dimensions.size())}, at::TensorOptions().dtype(at::kLong));
  std::memcpy(cpu.data_ptr(), dimensions.data(), dimensions.size() * sizeof(uint64_t));
  auto meta = cpu.to(x.device());
  uint64_t inner = dimensions[dimensions.size() - 3];
  bool rows = inner % (32 / sizeof(T)) != 0;
  uint32_t lanes = rows ? inner : 1;
  if (rows) inner *= dimensions[dimensions.size() - 6];
  uint64_t tasks = x.numel() / inner * ((inner + tile - 1) / tile);
  uint32_t blocks = std::min<uint64_t>(8, tasks);
  auto xa = reinterpret_cast<uintptr_t>(x.const_data_ptr());
  auto ya = reinterpret_cast<uintptr_t>(y.const_data_ptr());
  auto xp = reinterpret_cast<GM_ADDR>(xa / 32 * 32);
  auto yp = reinterpret_cast<GM_ADDR>(ya / 32 * 32);
  auto zp = (GM_ADDR)out.data_ptr();
  auto mp = (GM_ADDR)meta.data_ptr();
  uint64_t xs = xa % 32 / sizeof(T), ys = ya % 32 / sizeof(T);
  uint32_t rank = dimensions.size() / 3;
#define RUN(OP) \
  if (rows) kernel<T, OP, true><<<blocks, nullptr, thin::stream()>>>(xp, yp, zp, mp, x.numel(), inner, rank, xs, ys, alpha, lanes); \
  else kernel<T, OP><<<blocks, nullptr, thin::stream()>>>(xp, yp, zp, mp, x.numel(), inner, rank, xs, ys, alpha, lanes)
  if (!strcmp(op, "sub")) { RUN(0); }
  if (!strcmp(op, "mul")) { RUN(1); }
  if (!strcmp(op, "div")) { RUN(2); }
  if (!strcmp(op, "add")) { RUN(3); }
#undef RUN
  thin::sync();
}
}
