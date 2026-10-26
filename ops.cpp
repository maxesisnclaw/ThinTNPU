#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/ExpandUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/ops/split_native.h>
#include <ATen/ops/split_with_sizes_native.h>
#include <torch/library.h>
#include <acl/acl_rt.h>
#include <functional>
#include <cstring>
#include <numeric>
#include "graph_capture.h"
#include "kernel_operator.h"

namespace thin {
void check(aclError, const char*);
void select(c10::Device);
aclrtStream stream();
void sync();
aclrtFuncHandle kernel(const char*, const char*);
at::Tensor& copy(at::Tensor&, const at::Tensor&, bool);
}
namespace thin_ops {
constexpr auto npu = c10::DeviceType::PrivateUse1;
namespace materialize { bool copy(at::Tensor&, const at::Tensor&); }
void checked(aclError e) { thin::check(e, "thin_ops ACL runtime"); }
void inference(const at::Tensor& x) {
  TORCH_CHECK(x.device().type() == npu && !x.requires_grad(), "thin ops require inference-only NPU tensors");
}
at::Tensor& copy_strided(at::Tensor& dst, const at::Tensor& src, bool nonblocking) {
  TORCH_CHECK(dst.sizes() == src.sizes() && dst.scalar_type() == src.scalar_type(), "copy requires matching sizes and dtype");
  const bool dn = dst.device().type() == npu, sn = src.device().type() == npu;
  TORCH_CHECK((dn || dst.is_cpu()) && (sn || src.is_cpu()) && (dn || sn), "Unsupported copy devices");
  if (dst.is_same(src) || !dst.numel()) return dst;
  at::assert_no_internal_overlap(dst);
  at::assert_no_partial_overlap(dst, src);
  thin::select(c10::Device(npu, 0));
  TORCH_CHECK(!thin::capturing() || dn, "Device-to-host copy is not allowed during graph capture");
  if (dn && sn && !src.is_contiguous() && materialize::copy(dst, src)) return dst;
  thin::sync();
  auto kind = dn ? (sn ? ACL_MEMCPY_DEVICE_TO_DEVICE : ACL_MEMCPY_HOST_TO_DEVICE) : ACL_MEMCPY_DEVICE_TO_HOST;
  const auto bytes = dst.element_size();
  auto* dp = static_cast<char*>(dst.data_ptr());
  auto* sp = static_cast<const char*>(src.const_data_ptr());
  int64_t suffix = dst.dim(), run = 1;
  while (suffix > 0) {
    auto d = suffix - 1;
    if (dst.size(d) != 1 && (dst.stride(d) != run || src.stride(d) != run)) break;
    run *= dst.size(d);
    --suffix;
  }
  std::function<void(int64_t, int64_t, int64_t)> transfer = [&](int64_t dim, int64_t di, int64_t si) {
    if (dim == suffix) {
      if (thin::capturing()) {
        const void* source = sp + si * bytes;
        if (!sn) {
          // Pageable H2D uploads synchronize internally and are rejected in capture.
          void* pinned = nullptr;
          checked(aclrtMallocHost(&pinned, run * bytes));
          auto owner = std::make_shared<thin::CapturedHostAllocation>(pinned);
          thin::capture_memory->host_inputs.push_back(std::move(owner));
          std::memcpy(pinned, source, run * bytes);
          source = pinned;
        }
        checked(aclrtMemcpyAsync(dp + di * bytes, run * bytes, source,
                                run * bytes, kind, thin::stream()));
      } else
        checked(aclrtMemcpy(dp + di * bytes, run * bytes, sp + si * bytes, run * bytes, kind));
      return;
    }
    for (int64_t i = 0; i < dst.size(dim); ++i)
      transfer(dim + 1, di + i * dst.stride(dim), si + i * src.stride(dim));
  };
  transfer(0, 0, 0);
  return dst;
}
at::Tensor packed(const at::Tensor& x) {
  if (x.is_contiguous() && reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 32 == 0) return x;
  auto y = at::empty(x.sizes(), x.options());
  copy_strided(y, x, false);
  return y;
}
namespace materialize {
using namespace AscendC;
constexpr uint32_t tile = 1024;
constexpr uint32_t source_capacity = tile * 16;

__global__ __aicore__ void transpose_kernel(GM_ADDR src, GM_ADDR dst,
                                            uint64_t outer, uint64_t rows, uint64_t columns) {
  constexpr uint32_t span = 128;
  TPipe pipe;
  TBuf<QuePosition::VECCALC> input, output;
  pipe.InitBuffer(input, span * 16 * sizeof(half));
  pipe.InitBuffer(output, span * 16 * sizeof(half));
  auto in = input.Get<half>(), out = output.Get<half>();
  GlobalTensor<half> x, y;
  x.SetGlobalBuffer((__gm__ half*)src);
  y.SetGlobalBuffer((__gm__ half*)dst);
  uint64_t tiles = (rows + span - 1) / span;
  for (uint64_t task = GetBlockIdx(); task < outer * (columns / 16) * tiles;
       task += GetBlockNum()) {
    uint64_t pos = task % tiles * span;
    uint64_t channel = task / tiles % (columns / 16) * 16;
    uint64_t batch = task / tiles / (columns / 16);
    uint32_t count = rows - pos < span ? rows - pos : span;
    uint64_t source = batch * rows * columns + channel * rows + pos;
    if ((rows - count) / 16 <= 65535)
      DataCopy(in, x[source], DataCopyParams{16, (uint16_t)(count / 16),
                                           (uint16_t)((rows - count) / 16), 0});
    else
      for (uint32_t c = 0; c < 16; ++c) DataCopy(in[c * count], x[source + c * rows], count);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    uint64_t sources[16], destinations[16];
    for (uint32_t i = 0; i < 16; ++i) {
      sources[i] = (uint64_t)in[i * count].GetPhyAddr();
      destinations[i] = (uint64_t)out[i * 16].GetPhyAddr();
    }
    TransDataTo5HDParams params;
    params.repeatTimes = count / 16;
    params.srcRepStride = count == 16 ? 0 : 1;
    params.dstRepStride = count == 16 ? 0 : 16;
    TransDataTo5HD<half>(destinations, sources, params);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    uint64_t destination = batch * rows * columns + pos * columns + channel;
    if ((columns - 16) / 16 <= 65535)
      DataCopy(y[destination], out, DataCopyParams{(uint16_t)count, 1, 0,
                                                (uint16_t)((columns - 16) / 16)});
    else
      for (uint32_t r = 0; r < count; ++r) DataCopy(y[destination + r * columns], out[r * 16], 16);
    SetFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(0);
  }
}

__aicore__ inline uint64_t source_offset(uint64_t linear, LocalTensor<uint64_t> meta,
                                        uint32_t rank, uint64_t skew) {
  uint64_t offset = skew;
  for (uint32_t d = rank; d > 0; --d) {
    uint64_t size = meta.GetValue((d - 1) * 2);
    offset += (linear % size) * meta.GetValue((d - 1) * 2 + 1);
    linear /= size;
  }
  return offset;
}

template<class T>
__global__ __aicore__ void gather_kernel(GM_ADDR src, GM_ADDR dst, GM_ADDR metadata,
                                        uint64_t length, uint32_t rank, uint64_t skew, uint32_t step) {
  constexpr uint32_t align = 32 / sizeof(T);
  TPipe pipe;
  TBuf<QuePosition::VECCALC> input, output, offsets, shape;
  pipe.InitBuffer(input, source_capacity * sizeof(T));
  pipe.InitBuffer(output, tile * sizeof(T));
  pipe.InitBuffer(offsets, tile * sizeof(uint32_t));
  pipe.InitBuffer(shape, (rank * 16 + 31) / 32 * 32);
  auto in = input.Get<T>();
  auto out = output.Get<T>();
  auto idx = offsets.Get<uint32_t>();
  auto meta = shape.Get<uint64_t>();
  GlobalTensor<T> xg, yg;
  GlobalTensor<uint64_t> mg;
  xg.SetGlobalBuffer((__gm__ T*)src);
  yg.SetGlobalBuffer((__gm__ T*)dst);
  mg.SetGlobalBuffer((__gm__ uint64_t*)metadata);
  DataCopy(meta, mg, (rank * 2 + 3) / 4 * 4);
  SetFlag<HardEvent::MTE2_S>(0);
  WaitFlag<HardEvent::MTE2_S>(0);
  for (uint64_t start = GetBlockIdx() * step; start < length;
       start += GetBlockNum() * step) {
    uint32_t count = length - start < step ? length - start : step;
    uint32_t padded = (count + align - 1) / align * align;
    uint64_t inner = meta.GetValue((rank - 1) * 2);
    uint64_t stride = meta.GetValue((rank - 1) * 2 + 1);
    bool affine = padded == count && start % inner + count <= inner;
    uint64_t a = inner, b = 8;
    while (b) { uint64_t remainder = a % b; a = b; b = remainder; }
    uint64_t period = inner / a * 8;
    bool repeated = !affine && padded == count && period <= count && count % inner == 0 &&
                    start % inner == 0 && rank >= 2 &&
                    (start / inner) % meta.GetValue((rank - 2) * 2) + count / inner <=
                        meta.GetValue((rank - 2) * 2);
    uint64_t row_stride = repeated ? meta.GetValue((rank - 2) * 2 + 1) : 0;
    uint64_t low = source_offset(start, meta, rank, skew), high = low;
    if (affine) high += (count - 1) * stride;
    else if (repeated) high += (count / inner - 1) * row_stride + (inner - 1) * stride;
    else {
      for (uint32_t j = 1; j < count; ++j) {
        uint64_t offset = source_offset(start + j, meta, rank, skew);
        low = offset < low ? offset : low;
        high = offset > high ? offset : high;
      }
    }
    uint64_t base = low / align * align;
    uint64_t copied = (high - base + align) / align * align;
    if (copied <= source_capacity) {
      if (affine || repeated) {
        uint32_t seed = repeated ? period : 8;
        for (uint32_t j = 0; j < seed; ++j)
          idx.SetValue(j, (low - base + (repeated ? j / inner * row_stride + j % inner * stride : j * stride)) * sizeof(T));
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        auto signed_idx = offsets.Get<int32_t>();
        for (uint32_t built = seed; built < padded;) {
          uint32_t n = padded - built < built ? padded - built : built;
          Adds(signed_idx[built], signed_idx, int32_t((repeated ? built / inner * row_stride : built * stride) * sizeof(T)), n);
          PipeBarrier<PIPE_V>();
          built += n;
        }
      } else {
        for (uint32_t j = 0; j < padded; ++j) {
          uint64_t offset = source_offset(start + (j < count ? j : count - 1), meta, rank, skew);
          idx.SetValue(j, (offset - base) * sizeof(T));
        }
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
      }
      DataCopy(in, xg[base], copied);
    } else {
      // Sparse views need disjoint aligned source blocks, not an oversized UB span.
      for (uint32_t j = 0; j < padded; ++j) {
        uint64_t offset = source_offset(start + (j < count ? j : count - 1), meta, rank, skew);
        DataCopy(in[j * align], xg[offset / align * align], align);
        idx.SetValue(j, (j * align + offset % align) * sizeof(T));
      }
      SetFlag<HardEvent::S_V>(0);
      WaitFlag<HardEvent::S_V>(0);
    }
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    Gather(out, in, idx, 0, padded);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    DataCopy(yg[start], out, padded);
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(0);
  }
}

bool copy(at::Tensor& out, const at::Tensor& x) {
  if (!out.is_contiguous() || reinterpret_cast<uintptr_t>(out.const_data_ptr()) % 32 ||
      (x.scalar_type() != at::kHalf && x.scalar_type() != at::kFloat)) return false;
  std::vector<uint64_t> dimensions;
  for (int64_t d = 0; d < x.dim(); ++d) {
    if (x.size(d) == 1) continue;
    if (!dimensions.empty() && dimensions.back() == uint64_t(x.size(d) * x.stride(d))) {
      dimensions[dimensions.size() - 2] *= x.size(d);
      dimensions.back() = x.stride(d);
    } else {
      dimensions.push_back(x.size(d));
      dimensions.push_back(x.stride(d));
    }
  }
  if (dimensions.empty()) dimensions = {1, 0};
  auto address = reinterpret_cast<uintptr_t>(x.const_data_ptr());
  size_t n = dimensions.size();
  if (x.scalar_type() == at::kHalf && address % 32 == 0 && (n == 4 || n == 6) &&
      dimensions[n - 3] == 1 && dimensions[n - 1] == dimensions[n - 4] &&
      dimensions[n - 4] % 16 == 0 && dimensions[n - 2] % 16 == 0 &&
      (n == 4 || dimensions[1] == dimensions[2] * dimensions[4])) {
    uint64_t rows = dimensions[n - 4], columns = dimensions[n - 2];
    uint64_t outer = x.numel() / rows / columns;
    uint64_t tasks = outer * (columns / 16) * ((rows + 127) / 128);
    transpose_kernel<<<std::min<uint64_t>(8, tasks), nullptr, thin::stream()>>>(
        (GM_ADDR)x.const_data_ptr(), (GM_ADDR)out.data_ptr(), outer, rows, columns);
    thin::sync();
    return true;
  }
  auto cpu_meta = at::empty({int64_t(dimensions.size())}, at::TensorOptions().dtype(at::kLong));
  std::memcpy(cpu_meta.data_ptr(), dimensions.data(), dimensions.size() * sizeof(uint64_t));
  auto meta = cpu_meta.to(x.device());
  auto base = reinterpret_cast<GM_ADDR>(address / 32 * 32);
  uint64_t skew = address % 32 / x.element_size();
  uint64_t quantum = std::lcm<uint64_t>(dimensions[n - 2], 32 / x.element_size());
  uint32_t step = quantum <= tile ? tile / quantum * quantum : tile;
  auto blocks = std::min<int64_t>(8, (x.numel() + step - 1) / step);
  if (x.scalar_type() == at::kFloat)
    gather_kernel<uint32_t><<<blocks, nullptr, thin::stream()>>>(base, (GM_ADDR)out.data_ptr(), (GM_ADDR)meta.data_ptr(), x.numel(), dimensions.size() / 2, skew, step);
  else
    gather_kernel<uint16_t><<<blocks, nullptr, thin::stream()>>>(base, (GM_ADDR)out.data_ptr(), (GM_ADDR)meta.data_ptr(), x.numel(), dimensions.size() / 2, skew, step);
  thin::sync();
  return true;
}

at::Tensor pack(const at::Tensor& x) {
  if (!x.numel() || (x.is_contiguous() && reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 32 == 0)) return x;
  auto out = at::empty(x.sizes(), x.options());
  TORCH_CHECK(copy(out, x), "materialization supports float16/float32");
  return out;
}
}
namespace concatenate {
using namespace AscendC;
constexpr uint32_t tile = 8192;

constexpr uint32_t plan_capacity = 8;
struct CopyPlan {
  uint64_t source[plan_capacity];
  uint64_t run[plan_capacity];
  uint64_t offset[plan_capacity];
  uint32_t count;
};

__global__ __aicore__ void concat_kernel(GM_ADDR dst, uint64_t pitch,
                                        uint64_t outer, CopyPlan plan) {
  TPipe pipe;
  TBuf<QuePosition::VECCALC> buffer;
  pipe.InitBuffer(buffer, 2 * tile * sizeof(uint32_t));
  auto local = buffer.Get<uint32_t>();
  GlobalTensor<uint32_t> x, y;
  y.SetGlobalBuffer((__gm__ uint32_t*)dst);
  for (uint32_t input = 0; input < plan.count; ++input) {
    uint64_t run = plan.run[input];
    uint64_t offset = plan.offset[input];
    x.SetGlobalBuffer((__gm__ uint32_t*)plan.source[input]);
    if (outer >= GetBlockNum() && run <= tile && (pitch - run) / 8 <= 65535) {
      uint64_t rows_per_tile = (outer + GetBlockNum() - 1) / GetBlockNum();
      rows_per_tile = rows_per_tile < 2 * tile / run ? rows_per_tile : 2 * tile / run;
      uint64_t tasks = (outer + rows_per_tile - 1) / rows_per_tile;
      for (uint64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
        uint64_t row = task * rows_per_tile;
        uint32_t rows = outer - row < rows_per_tile ? outer - row : rows_per_tile;
        DataCopy(local, x[row * run], rows * run);
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
        DataCopy(y[row * pitch + offset], local,
                 DataCopyParams{(uint16_t)rows, (uint16_t)(run / 8), 0, (uint16_t)((pitch - run) / 8)});
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
      }
      continue;
    }
    uint64_t row_blocks = (GetBlockNum() + outer - 1) / outer;
    uint64_t chunk = (run + row_blocks * 8 - 1) / (row_blocks * 8) * 8;
    chunk = chunk < tile ? chunk : tile;
    uint64_t tiles = (run + chunk - 1) / chunk;
    uint64_t previous_row = 0, previous_start = 0;
    uint32_t previous_count = 0, iteration = 0;
    for (uint64_t task = GetBlockIdx(); task < outer * tiles; task += GetBlockNum()) {
      uint32_t bank = iteration % 2;
      uint64_t row = task / tiles;
      uint64_t start = task % tiles * chunk;
      uint32_t count = run - start < chunk ? run - start : chunk;
      if (iteration >= 2) WaitFlag<HardEvent::MTE3_MTE2>(bank);
      DataCopy(local[bank * tile], x[row * run + start], count);
      SetFlag<HardEvent::MTE2_MTE3>(bank);
      if (iteration) {
        uint32_t previous_bank = (iteration - 1) % 2;
        WaitFlag<HardEvent::MTE2_MTE3>(previous_bank);
        DataCopy(y[previous_row * pitch + offset + previous_start], local[previous_bank * tile], previous_count);
        SetFlag<HardEvent::MTE3_MTE2>(previous_bank);
      }
      previous_row = row;
      previous_start = start;
      previous_count = count;
      ++iteration;
    }
    if (iteration) {
      uint32_t bank = (iteration - 1) % 2;
      WaitFlag<HardEvent::MTE2_MTE3>(bank);
      DataCopy(y[previous_row * pitch + offset + previous_start], local[bank * tile], previous_count);
      SetFlag<HardEvent::MTE3_MTE2>(bank);
      WaitFlag<HardEvent::MTE3_MTE2>(0);
      if (iteration > 1) WaitFlag<HardEvent::MTE3_MTE2>(1);
    }
  }
}

bool aligned(const std::vector<at::Tensor>& tensors, int64_t dim, int64_t inner) {
  for (const auto& t : tensors) {
    if (!t.numel()) continue;
    if (!t.is_contiguous() || reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 32 ||
        (t.size(dim) * inner * t.element_size()) % 32) return false;
  }
  return true;
}

void copy(const std::vector<at::Tensor>& tensors, at::Tensor& out, int64_t dim, int64_t inner) {
  thin::select(out.device());
  uint64_t pitch = out.size(dim) * inner * out.element_size() / sizeof(uint32_t);
  uint64_t outer = out.numel() * out.element_size() / sizeof(uint32_t) / pitch;
  uint64_t offset = 0;
  CopyPlan plan{};
  uint64_t tasks = 0;
  auto launch = [&] {
    uint32_t blocks = std::min<uint64_t>(8, tasks);
    concat_kernel<<<blocks, nullptr, thin::stream()>>>((GM_ADDR)out.data_ptr(), pitch, outer, plan);
    plan.count = 0;
    tasks = 0;
  };
  for (const auto& t : tensors) {
    if (!t.numel()) continue;
    uint64_t run = t.size(dim) * inner * t.element_size() / sizeof(uint32_t);
    uint64_t source = reinterpret_cast<uintptr_t>(t.const_data_ptr());
    if (outer == 1 && plan.count &&
        plan.source[plan.count - 1] + plan.run[plan.count - 1] * sizeof(uint32_t) == source) {
      plan.run[plan.count - 1] += run;
      tasks = std::max<uint64_t>(tasks, (plan.run[plan.count - 1] + 7) / 8);
    } else {
      uint32_t index = plan.count++;
      plan.source[index] = source;
      plan.run[index] = run;
      plan.offset[index] = offset;
      tasks = std::max<uint64_t>(tasks, outer * ((run + 7) / 8));
    }
    offset += run;
    if (plan.count == plan_capacity) launch();
  }
  if (plan.count) launch();
  thin::sync();
}
}
#include "strided_concat.h"
at::Tensor cat(const at::ITensorListRef& inputs, int64_t dim) {
  std::vector<at::Tensor> tensors;
  for (const at::Tensor& t : inputs) tensors.push_back(t);
  TORCH_CHECK(!tensors.empty(), "cat expects a nonempty tensor list");
  const at::Tensor* reference = &tensors[0];
  for (const auto& t : tensors) {
    inference(t);
    TORCH_CHECK(t.dim() > 0, "cat cannot concatenate scalar tensors");
    if (!(t.dim() == 1 && t.numel() == 0)) reference = &t;
  }
  dim = at::maybe_wrap_dim(dim, reference->dim());
  auto sizes = reference->sizes().vec();
  sizes[dim] = 0;
  for (const auto& t : tensors) {
    TORCH_CHECK(t.device() == reference->device() && t.scalar_type() == reference->scalar_type(), "cat requires matching device and dtype");
    if (t.dim() == 1 && !t.numel()) continue;
    TORCH_CHECK(t.dim() == reference->dim(), "cat rank mismatch");
    for (int64_t d = 0; d < t.dim(); ++d)
      TORCH_CHECK(d == dim || t.size(d) == reference->size(d), "cat shape mismatch outside concatenation dimension");
    sizes[dim] += t.size(dim);
  }
  auto out = at::empty(sizes, reference->options());
  if (!out.numel()) return out;
  int64_t inner = 1;
  for (int64_t d = dim + 1; d < out.dim(); ++d) inner *= out.size(d);
  if (concatenate::aligned(tensors, dim, inner)) {
    concatenate::copy(tensors, out, dim, inner);
    return out;
  }
  if (strided_concat::supported(tensors, out, dim, inner)) {
    strided_concat::launch(tensors, out, dim, inner);
    return out;
  }
  int64_t start = 0;
  for (const auto& t : tensors) {
    if (t.dim() == 1 && !t.numel()) continue;
    auto shape = sizes;
    shape[dim] = t.size(dim);
    auto dst = out.as_strided(shape, out.strides(), start * out.stride(dim));
    copy_strided(dst, t, false);
    start += t.size(dim);
  }
  return out;
}
#include "strided_binary.h"
at::Tensor binary(const at::Tensor& input, const at::Tensor& other, const char* op, const at::Scalar& alpha = 1) {
  auto x = input, y = other;
  if (x.is_cpu() && x.dim() == 0) x = x.to(y.scalar_type()).to(y.device());
  if (y.is_cpu() && y.dim() == 0) y = y.to(x.scalar_type()).to(x.device());
  auto sizes = at::infer_size(x.sizes(), y.sizes());
  if (x.sizes() != sizes) x = x.expand(sizes);
  if (y.sizes() != sizes) y = y.expand(sizes);
  inference(x); inference(y);
  TORCH_CHECK(x.device() == y.device() && x.sizes() == y.sizes() && x.scalar_type() == y.scalar_type(), "binary requires matching device, shape and dtype");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf, "binary supports float32/float16");
  auto out = at::empty(x.sizes(), x.options());
  if (!x.numel()) return out;
  bool packed = x.is_contiguous() && y.is_contiguous() &&
      reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 32 == 0 &&
      reinterpret_cast<uintptr_t>(y.const_data_ptr()) % 32 == 0;
  if (!packed && strided_binary::supported(x, y)) {
    if (x.scalar_type() == at::kHalf) strided_binary::launch<half>(x, y, out, op, alpha.toFloat());
    else strided_binary::launch<float>(x, y, out, op, alpha.toFloat());
    return out;
  }
  auto a = materialize::pack(x), b = materialize::pack(y);
  std::string symbol = std::string(op) + (x.scalar_type() == at::kFloat ? "_fp32" : "_fp16");
  auto fn = thin::kernel("THIN_ELEMENTWISE_BINARY", symbol.c_str());
  struct Args { void* a; void* b; void* out; uint64_t length; float alpha; uint32_t padding; };
  static_assert(sizeof(Args) == 40);
  Args args{a.data_ptr(), b.data_ptr(), out.data_ptr(), static_cast<uint64_t>(x.numel()), alpha.toFloat(), 0};
  checked(aclrtLaunchKernelWithHostArgs(fn, std::min<int64_t>(8, (x.numel() + 2047) / 2048), thin::stream(), nullptr, &args, sizeof(args), nullptr, 0));
  thin::sync();
  return out;
}
at::Tensor sub(const at::Tensor& x, const at::Tensor& y, const at::Scalar& alpha) { return binary(x, y, "sub", alpha); }
at::Tensor add(const at::Tensor& x, const at::Tensor& y, const at::Scalar& alpha) { return binary(x, y, "add", alpha); }
at::Tensor mul(const at::Tensor& x, const at::Tensor& y) { return binary(x, y, "mul"); }
at::Tensor div(const at::Tensor& x, const at::Tensor& y) { return binary(x, y, "div"); }
at::Tensor& silu_(at::Tensor& x) {
  inference(x);
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf, "silu_ supports float32/float16");
  at::assert_no_internal_overlap(x);
  if (!x.numel()) return x;
  auto a = packed(x);
  auto out = at::empty(x.sizes(), x.options());
  auto fn = thin::kernel("THIN_ELEMENTWISE_BINARY", x.scalar_type() == at::kFloat ? "silu_fp32" : "silu_fp16");
  struct Args { void* a; void* out; uint64_t length; };
  Args args{a.data_ptr(), out.data_ptr(), static_cast<uint64_t>(x.numel())};
  checked(aclrtLaunchKernelWithHostArgs(fn, std::min<int64_t>(8, (x.numel() + 2047) / 2048), thin::stream(), nullptr, &args, sizeof(args), nullptr, 0));
  thin::sync();
  copy_strided(x, out, false);
  return x;
}
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("as_strided", TORCH_FN(at::native::as_strided_tensorimpl));
  m.impl("view", TORCH_FN(at::native::view));
  m.impl("split.Tensor", TORCH_FN(at::native::split));
  m.impl("split_with_sizes", TORCH_FN(at::native::split_with_sizes));
  m.impl("cat", TORCH_FN(cat));
  m.impl("silu_", TORCH_FN(silu_));
  m.impl("sub.Tensor", TORCH_FN(sub));
  m.impl("mul.Tensor", TORCH_FN(mul));
  m.impl("div.Tensor", TORCH_FN(div));
  m.impl("add.Tensor", TORCH_FN(add));
}
}
