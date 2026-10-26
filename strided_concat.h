#pragma once
namespace strided_concat {
using namespace AscendC;

constexpr uint32_t plan_capacity = 8;
constexpr uint32_t tile = 2048;
constexpr uint32_t max_rows = 256;
constexpr uint32_t source_capacity = 4096;
constexpr uint32_t meta_stride = 8;

// metadata layout per input: base, base_off, run, row_pitch, colbase, regbase, slot, mode
// mode: 0 = CONTIG (row_pitch == run), 1 = BROADCAST (row_pitch == 0), 2 = STRIDED

template<class T>
__global__ __aicore__ void concat_kernel(GM_ADDR dst, GM_ADDR metadata,
                                         uint64_t outer, uint64_t width,
                                         uint32_t count, uint32_t rows_per_tile, bool stationary) {
  constexpr uint32_t align = 32 / sizeof(T);
  TPipe pipe;
  TBuf<QuePosition::VECCALC> source_buf, stage_buf, index_buf, pos_buf, colin_buf, local_buf, shape_buf;
  pipe.InitBuffer(source_buf, source_capacity * sizeof(T));
  pipe.InitBuffer(stage_buf, tile * sizeof(T));
  pipe.InitBuffer(index_buf, tile * sizeof(uint32_t));
  pipe.InitBuffer(pos_buf, plan_capacity * max_rows * sizeof(uint32_t));
  pipe.InitBuffer(colin_buf, tile * sizeof(uint32_t));
  pipe.InitBuffer(local_buf, tile * sizeof(uint32_t));
  pipe.InitBuffer(shape_buf, plan_capacity * meta_stride * sizeof(uint64_t));
  auto src = source_buf.Get<T>();
  auto stage = stage_buf.Get<T>();
  auto idx = index_buf.Get<uint32_t>();
  auto pos = pos_buf.Get<uint32_t>();
  auto colin = colin_buf.Get<uint32_t>();
  auto collocal = local_buf.Get<uint32_t>();
  auto meta = shape_buf.Get<uint64_t>();

  GlobalTensor<T> x, y;
  GlobalTensor<uint64_t> mg;
  y.SetGlobalBuffer((__gm__ T*)dst);
  mg.SetGlobalBuffer((__gm__ uint64_t*)metadata);
  DataCopy(meta, mg, (count * meta_stride + 3) / 4 * 4);
  SetFlag<HardEvent::MTE2_S>(0);
  WaitFlag<HardEvent::MTE2_S>(0);

  if (!stationary) for (uint32_t i = 0; i < count; ++i) {
    uint32_t cb = meta.GetValue(i * meta_stride + 4);
    uint32_t run = meta.GetValue(i * meta_stride + 2);
    for (uint32_t l = 0; l < run; ++l) {
      colin.SetValue(cb + l, i);
      collocal.SetValue(cb + l, l);
    }
  }
  if (stationary) {
    GlobalTensor<uint32_t> indices;
    indices.SetGlobalBuffer((__gm__ uint32_t*)metadata + count * meta_stride * 2);
    DataCopy(idx, indices, (rows_per_tile * width + 7) / 8 * 8);
  }

  uint64_t tiles = (outer + rows_per_tile - 1) / rows_per_tile;
  for (uint64_t t = GetBlockIdx(); t < tiles; t += GetBlockNum()) {
    uint64_t rowbase = t * rows_per_tile;
    uint32_t rows = outer - rowbase < rows_per_tile ? (uint32_t)(outer - rowbase) : rows_per_tile;

    for (uint32_t i = 0; i < count; ++i) {
      uint64_t base = meta.GetValue(i * meta_stride + 0);
      uint64_t base_off = meta.GetValue(i * meta_stride + 1);
      uint64_t run = meta.GetValue(i * meta_stride + 2);
      uint64_t pitch = meta.GetValue(i * meta_stride + 3);
      uint64_t regbase = meta.GetValue(i * meta_stride + 5);
      uint64_t slot = meta.GetValue(i * meta_stride + 6);
      uint32_t mode = meta.GetValue(i * meta_stride + 7);
      x.SetGlobalBuffer((__gm__ T*)base);
      if (mode == 1) {
        uint64_t off = base_off;
        uint64_t floor = off / align * align;
        uint32_t skew = off - floor;
        DataCopy(src[regbase], x[floor], (skew + run + align - 1) / align * align);
        if (!stationary)
          for (uint32_t r = 0; r < rows; ++r) pos.SetValue(i * max_rows + r, regbase + skew);
      } else if (mode == 0) {
        uint64_t off0 = base_off + rowbase * run;
        uint64_t floor = off0 / align * align;
        uint32_t skew = off0 - floor;
        DataCopy(src[regbase], x[floor], (skew + (uint64_t)rows * run + align - 1) / align * align);
        if (!stationary)
          for (uint32_t r = 0; r < rows; ++r) pos.SetValue(i * max_rows + r, regbase + skew + r * run);
      } else {
        for (uint32_t r = 0; r < rows; ++r) {
          uint64_t off = base_off + (rowbase + r) * pitch;
          uint64_t floor = off / align * align;
          uint32_t skew = off - floor;
          DataCopy(src[regbase + r * slot], x[floor], (skew + run + align - 1) / align * align);
          if (!stationary) pos.SetValue(i * max_rows + r, regbase + r * slot + skew);
        }
      }
    }

    uint32_t total = rows * width;
    if (!stationary)
      for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < width; ++c) {
          uint32_t i = colin.GetValue(c);
          uint32_t p = pos.GetValue(i * max_rows + r) + collocal.GetValue(c);
          idx.SetValue(r * width + c, p * sizeof(T));
        }

    SetFlag<HardEvent::S_V>(0);
    WaitFlag<HardEvent::S_V>(0);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    Gather(stage, src, idx, 0, total);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    DataCopy(y[rowbase * width], stage, total);
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(0);
  }
}

inline uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

inline uint64_t super_row_pitch(const at::Tensor& t, int64_t dim, uint64_t run) {
  for (int64_t k = dim - 1; k >= 0; --k)
    if (t.size(k) != 1) return (uint64_t)t.stride(k);
  return run;
}

inline bool supported(const std::vector<at::Tensor>& tensors, const at::Tensor& out,
                      int64_t dim, int64_t inner) {
  if (out.scalar_type() != at::kFloat && out.scalar_type() != at::kHalf) return false;
  const uint64_t elsize = out.element_size();
  const uint64_t width = (uint64_t)out.size(dim) * (uint64_t)inner;
  if (!width || (width * elsize) % 32 != 0 || width > tile) return false;

  uint32_t live = 0;
  for (const auto& t : tensors) if (t.numel()) ++live;
  if (!live || live > plan_capacity) return false;

  for (const auto& t : tensors) {
    if (!t.numel()) continue;
    if (reinterpret_cast<uintptr_t>(t.const_data_ptr()) == 0) return false;
    const int64_t r = t.dim();
    int64_t expect = 1;
    for (int64_t d = r - 1; d >= dim; --d) {
      if (t.size(d) == 1) continue;
      if (t.stride(d) != expect) return false;
      expect *= t.size(d);
    }
    bool first = true;
    int64_t prev_stride = 0, prev_size = 0;
    for (int64_t d = dim - 1; d >= 0; --d) {
      if (t.size(d) == 1) continue;
      if (!first && t.stride(d) != prev_stride * prev_size) return false;
      first = false;
      prev_stride = t.stride(d);
      prev_size = t.size(d);
    }
  }
  return true;
}

template<class T>
void launch_typed(const std::vector<at::Tensor>& tensors, at::Tensor& out,
                  int64_t dim, int64_t inner) {
  constexpr uint64_t align = 32 / sizeof(T);
  const uint64_t width = (uint64_t)out.size(dim) * (uint64_t)inner;
  const uint64_t outer = (uint64_t)out.numel() / width;

  struct Desc { uint64_t base, base_off, run, pitch, colbase, slot, mode; };
  std::vector<Desc> desc;
  uint64_t col_start = 0;
  for (const auto& t : tensors) {
    if (!t.numel()) continue;
    Desc e{};
    const auto addr = reinterpret_cast<uintptr_t>(t.const_data_ptr());
    e.base = addr / 32 * 32;
    e.base_off = (addr % 32) / sizeof(T);
    const uint64_t seg = (uint64_t)t.size(dim);
    e.run = seg * (uint64_t)inner;
    e.colbase = col_start * (uint64_t)inner;
    e.pitch = super_row_pitch(t, dim, e.run);
    e.mode = e.pitch == e.run ? 0 : (e.pitch == 0 ? 1 : 2);
    e.slot = align_up(e.run + align - 1, align);
    desc.push_back(e);
    col_start += seg;
  }
  const uint32_t count = (uint32_t)desc.size();

  auto region = [&](const Desc& e, uint32_t R) -> uint64_t {
    if (e.mode == 1) return align_up(e.run, align) + align;
    if (e.mode == 0) return align_up((uint64_t)R * e.run, align) + align;
    return (uint64_t)R * e.slot;
  };
  uint32_t R = (uint32_t)std::min<uint64_t>({ (uint64_t)max_rows, tile / width, outer });
  if (R < 1) R = 1;
  for (;; --R) {
    uint64_t sum = 0;
    for (const auto& e : desc) sum += region(e, R);
    if (sum <= source_capacity || R == 1) break;
  }
  bool stationary = std::all_of(desc.begin(), desc.end(), [&](const Desc& e) {
    return R * e.pitch % align == 0;
  });

  std::vector<uint64_t> meta((size_t)count * meta_stride + (stationary ? (R * width + 7) / 8 * 4 : 0));
  uint64_t regbase = 0;
  for (uint32_t i = 0; i < count; ++i) {
    const auto& e = desc[i];
    meta[i * meta_stride + 0] = e.base;
    meta[i * meta_stride + 1] = e.base_off;
    meta[i * meta_stride + 2] = e.run;
    meta[i * meta_stride + 3] = e.pitch;
    meta[i * meta_stride + 4] = e.colbase;
    meta[i * meta_stride + 5] = regbase;
    meta[i * meta_stride + 6] = e.slot;
    meta[i * meta_stride + 7] = e.mode;
    regbase += region(e, R);
  }
  if (stationary) {
    auto indices = reinterpret_cast<uint32_t*>(meta.data() + count * meta_stride);
    for (uint32_t r = 0; r < R; ++r)
      for (uint32_t i = 0; i < count; ++i) {
        const auto& e = desc[i];
        uint64_t offset = e.mode == 0 ? e.base_off + r * e.pitch :
            (e.mode == 1 ? e.base_off : r * e.slot + (e.base_off + r * e.pitch) % align);
        for (uint64_t c = 0; c < e.run; ++c)
          indices[r * width + e.colbase + c] = (meta[i * meta_stride + 5] + offset + c) * sizeof(T);
      }
  }
  auto cpu = at::empty({(int64_t)meta.size()}, at::TensorOptions().dtype(at::kLong));
  std::memcpy(cpu.data_ptr(), meta.data(), meta.size() * sizeof(uint64_t));
  auto gpu = cpu.to(out.device());

  const uint32_t blocks = (uint32_t)std::min<uint64_t>(8, (outer + R - 1) / R);
  concat_kernel<T><<<blocks, nullptr, thin::stream()>>>(
      (GM_ADDR)out.data_ptr(), (GM_ADDR)gpu.data_ptr(), outer, width, count, R, stationary);
}

inline void launch(const std::vector<at::Tensor>& tensors, at::Tensor& out,
                   int64_t dim, int64_t inner) {
  thin::select(out.device());
  if (out.scalar_type() == at::kFloat) launch_typed<uint32_t>(tensors, out, dim, inner);
  else launch_typed<uint16_t>(tensors, out, dim, inner);
  thin::sync();
}
}
