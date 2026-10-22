#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/library.h>
#include <torch/custom_class.h>
#include <thread>
#include "graph_capture.h"
#include <acl/acl_rt.h>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>

namespace thin_ops { at::Tensor& copy_strided(at::Tensor&, const at::Tensor&, bool); }

namespace conv_cube { at::Tensor conv(const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&, int64_t, int64_t, bool); }

namespace image_conv {
at::Tensor prepare(const at::Tensor&);
at::Tensor conv(const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&,
                int64_t, int64_t, bool, bool);
at::Tensor conv_prepared(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                         const std::optional<at::Tensor>&, int64_t, int64_t, bool, bool);
}

namespace thin {
constexpr auto device_type = c10::DeviceType::PrivateUse1;
void check(aclError e, const char* call) { TORCH_CHECK(e == ACL_SUCCESS, call, " failed: ", e); }
#define ACL(call) check((call), #call)
void select(c10::Device d) {
  TORCH_CHECK(d.type() == device_type && (d.index() == -1 || d.index() == 0), "Only logical NPU 0 is exposed");
  ACL(aclrtSetDevice(0));
}
aclrtStream stream() {
  thread_local aclrtStream s = [] {
    select(c10::Device(device_type, 0));
    aclrtStream p = nullptr;
    ACL(aclrtCreateStream(&p));
    return p;
  }();
  return s;
}
thread_local CaptureMemory* capture_memory = nullptr;
void sync() { if (!capturing()) ACL(aclrtSynchronizeStream(stream())); }
CapturedAllocation::~CapturedAllocation() {
  if (pointer) {
    auto error = aclrtFree(pointer);
    if (error != ACL_SUCCESS) std::fprintf(stderr, "aclrtFree captured allocation failed: %d\n", error);
  }
}
CapturedHostAllocation::~CapturedHostAllocation() {
  if (pointer) {
    auto error = aclrtFreeHost(pointer);
    if (error != ACL_SUCCESS) std::fprintf(stderr, "aclrtFreeHost captured allocation failed: %d\n", error);
  }
}
void release_captured(void* context) {
  delete static_cast<std::shared_ptr<CapturedAllocation>*>(context);
}
class RuntimeGraph : public torch::CustomClassHolder {
  CaptureMemory memory_;
  aclmdlRI graph_ = nullptr;
  aclrtStream stream_ = stream();
  std::thread::id owner_ = std::this_thread::get_id();
  bool active_ = false;
  bool closed_ = false;
  void same_thread() const {
    TORCH_CHECK(owner_ == std::this_thread::get_id(), "RuntimeGraph must stay on its creating thread/stream");
  }
 public:
  void begin() {
    same_thread();
    TORCH_CHECK(!closed_ && !active_ && !graph_ && !capturing(), "RuntimeGraph cannot begin in its current state");
    sync();
    ACL(aclmdlRICaptureBegin(stream_, ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL));
    capture_memory = &memory_;
    active_ = true;
  }
  void end() {
    same_thread();
    TORCH_CHECK(active_ && capture_memory == &memory_, "RuntimeGraph is not capturing");
    auto error = aclmdlRICaptureEnd(stream_, &graph_);
    capture_memory = nullptr;
    active_ = false;
    ACL(error);
  }
  void replay() {
    same_thread();
    TORCH_CHECK(!closed_ && graph_ && !active_ && !capturing(), "RuntimeGraph is not ready to replay");
    ACL(aclmdlRIExecuteAsync(graph_, stream_));
  }
  void close() {
    same_thread();
    if (closed_) return;
    if (active_) end();
    TORCH_CHECK(!capturing(), "Cannot close a graph during another capture");
    ACL(aclrtSynchronizeStream(stream_));
    if (graph_) { ACL(aclmdlRIDestroy(graph_)); graph_ = nullptr; }
    memory_.host_inputs.clear();
    memory_.allocations.clear();
    closed_ = true;
  }
  ~RuntimeGraph() override {
    try { close(); }
    catch (const std::exception& error) {
      std::fprintf(stderr, "RuntimeGraph cleanup failed: %s\n", error.what());
      // Live graph addresses must not be freed when runtime cleanup fails.
      (void)new CaptureMemory(std::move(memory_));
    }
  }
};
struct Guard final : c10::impl::DeviceGuardImplInterface {
  c10::DeviceType type() const override { return device_type; }
  c10::Device getDevice() const override { return c10::Device(device_type, 0); }
  c10::Device exchangeDevice(c10::Device d) const override { select(d); return getDevice(); }
  void setDevice(c10::Device d) const override { select(d); }
  void uncheckedSetDevice(c10::Device) const noexcept override { (void)aclrtSetDevice(0); }
  c10::Stream getStream(c10::Device d) const override {
    select(d); (void)stream(); return c10::Stream(c10::Stream::DEFAULT, getDevice());
  }
  c10::Stream getDefaultStream(c10::Device d) const override { return getStream(d); }
  c10::Stream exchangeStream(c10::Stream s) const override {
    TORCH_CHECK(s.id() == 0, "Only the default stream is supported");
    return getStream(s.device());
  }
  c10::DeviceIndex deviceCount() const noexcept override {
    uint32_t n = 0; return aclrtGetDeviceCount(&n) == ACL_SUCCESS && n > 0 ? 1 : 0;
  }
  void synchronizeStream(const c10::Stream&) const override { sync(); }
  void synchronizeDevice(c10::DeviceIndex) const override { sync(); }
};
void release(void* p) {
  if (p) {
    auto e = aclrtFree(p);
    if (e != ACL_SUCCESS) std::fprintf(stderr, "aclrtFree failed: %d\n", e);
  }
}
struct Allocator final : c10::Allocator {
  c10::DataPtr allocate(size_t n) override {
    select(c10::Device(device_type, 0));
    void* p = nullptr;
    // DataCopy operates on whole 32-byte blocks, including the tensor tail.
    if (n) ACL(aclrtMalloc(&p, (n + 31) / 32 * 32, ACL_MEM_MALLOC_HUGE_FIRST));
    if (p && capturing()) {
      auto allocation = std::make_shared<CapturedAllocation>(p);
      capture_memory->allocations.push_back(allocation);
      return {p, new std::shared_ptr<CapturedAllocation>(std::move(allocation)),
              release_captured, c10::Device(device_type, 0)};
    }
    return {p, p, release, c10::Device(device_type, 0)};
  }
  c10::DeleterFnPtr raw_deleter() const override { return nullptr; }
  void copy_data(void* dst, const void* src, size_t n) const override {
    if (capturing()) {
      if (n) ACL(aclrtMemcpyAsync(dst, n, src, n, ACL_MEMCPY_DEVICE_TO_DEVICE, stream()));
    } else {
      sync(); if (n) ACL(aclrtMemcpy(dst, n, src, n, ACL_MEMCPY_DEVICE_TO_DEVICE));
    }
  }
};
Allocator allocator;
void validate_options(std::optional<at::Layout> layout, std::optional<c10::Device> device, std::optional<bool> pin) {
  TORCH_CHECK(!layout || *layout == at::kStrided, "Only strided layout is supported");
  TORCH_CHECK(!pin.value_or(false), "Pinned allocation is not implemented");
  select(device.value_or(c10::Device(device_type, 0)));
}
at::Tensor empty(c10::SymIntArrayRef size, std::optional<at::ScalarType> dtype,
                 std::optional<at::Layout> layout, std::optional<c10::Device> device,
                 std::optional<bool> pin, std::optional<c10::MemoryFormat> format) {
  validate_options(layout, device, pin);
  return at::Tensor(at::detail::empty_generic_symint(size, &allocator,
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), dtype.value_or(at::kFloat), format));
}
at::Tensor empty_strided(c10::SymIntArrayRef size, c10::SymIntArrayRef stride,
                 std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
                 std::optional<c10::Device> device, std::optional<bool> pin) {
  validate_options(layout, device, pin);
  return at::Tensor(at::detail::empty_strided_symint_generic(size, stride, &allocator,
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), dtype.value_or(at::kFloat)));
}
at::Tensor& copy(at::Tensor& dst, const at::Tensor& src, bool nonblocking) {
  return thin_ops::copy_strided(dst, src, nonblocking);
}
aclrtFuncHandle kernel(const char* binary_env, const char* symbol) {
  static std::mutex mutex;
  static std::map<std::string, aclrtBinHandle> binaries;
  static std::map<std::pair<std::string, std::string>, aclrtFuncHandle> functions;
  std::lock_guard<std::mutex> lock(mutex);
  const char* path = std::getenv(binary_env);
  TORCH_CHECK(path && *path, "Set ", binary_env, " to a precompiled kernel binary");
  auto key = std::make_pair(std::string(path), std::string(symbol));
  auto found = functions.find(key);
  if (found != functions.end()) return found->second;
  auto bin = binaries.find(path);
  if (bin == binaries.end()) {
    aclrtBinHandle handle = nullptr;
    ACL(aclrtBinaryLoadFromFile(path, nullptr, &handle));
    bin = binaries.emplace(path, handle).first;
  }
  aclrtFuncHandle fn = nullptr;
  ACL(aclrtBinaryGetFunction(bin->second, symbol, &fn));
  functions.emplace(std::move(key), fn);
  return fn;
}
at::Tensor add(const at::Tensor& x, const at::Tensor& y) {
  TORCH_CHECK(!x.requires_grad() && !y.requires_grad(), "thin_backend.add is inference-only");
  TORCH_CHECK(x.device() == y.device() && x.device().type() == device_type, "add requires tensors on the same NPU");
  TORCH_CHECK(x.scalar_type() == at::kFloat && y.scalar_type() == at::kFloat && x.sizes() == y.sizes(), "add requires matching float32 tensors");
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "add requires contiguous tensors");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 32 == 0 &&
              reinterpret_cast<uintptr_t>(y.const_data_ptr()) % 32 == 0,
              "add requires 32-byte aligned inputs; copy offset views first");
  select(x.device());
  auto z = at::empty(x.sizes(), x.options());
  if (!x.numel()) return z;
  auto fn = kernel("THIN_ADD_BINARY", "add_fp32");
  struct Args { void* x; void* y; void* z; uint64_t length; };
  Args args{x.data_ptr(), y.data_ptr(), z.data_ptr(), static_cast<uint64_t>(x.numel())};
  auto blocks = std::min<int64_t>(8, (x.numel() + 2047) / 2048);
  ACL(aclrtLaunchKernelWithHostArgs(fn, blocks, stream(), nullptr, &args, sizeof(args), nullptr, 0));
  sync();
  return z;
}
at::Tensor sigmoid(const at::Tensor& x) {
  TORCH_CHECK(!x.requires_grad(), "thin_backend.sigmoid is inference-only");
  TORCH_CHECK(x.device().type() == device_type, "sigmoid requires an NPU tensor");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf,
              "sigmoid supports float32 and float16");
  TORCH_CHECK(x.is_contiguous(), "sigmoid requires a contiguous tensor");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 32 == 0,
              "sigmoid requires a 32-byte aligned data pointer; copy offset views first");
  select(x.device());
  auto y = at::empty(x.sizes(), x.options());
  if (!x.numel()) return y;
  auto fn = kernel("THIN_SIGMOID_BINARY", x.scalar_type() == at::kFloat ? "sigmoid_fp32" : "sigmoid_fp16");
  struct Args { void* x; void* y; uint64_t length; };
  Args args{x.data_ptr(), y.data_ptr(), static_cast<uint64_t>(x.numel())};
  auto blocks = std::min<int64_t>(8, (x.numel() + 2047) / 2048);
  ACL(aclrtLaunchKernelWithHostArgs(fn, blocks, stream(), nullptr, &args, sizeof(args), nullptr, 0));
  sync();
  return y;
}
at::Tensor maxpool2d(const at::Tensor& x) {
  TORCH_CHECK(!x.requires_grad(), "thin_backend.maxpool2d is inference-only");
  TORCH_CHECK(x.device().type() == device_type, "maxpool2d requires an NPU tensor");
  TORCH_CHECK(x.dim() == 4 && x.size(1) > 0 && x.size(2) > 0 && x.size(3) > 0,
              "maxpool2d requires NCHW with positive C/H/W");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf,
              "maxpool2d supports float32 and float16");
  TORCH_CHECK(x.is_contiguous(), "maxpool2d requires contiguous NCHW");
  TORCH_CHECK(x.storage_offset() == 0, "maxpool2d requires an allocation-base tensor; copy views first");
  select(x.device());
  auto y = at::empty(x.sizes(), x.options());
  if (!x.numel()) return y;
  auto fn = kernel("THIN_MAXPOOL_BINARY", x.scalar_type() == at::kFloat ? "maxpool_fp32" : "maxpool_fp16");
  struct Args { void* x; void* y; uint64_t length; uint64_t height; uint64_t width; };
  Args args{x.data_ptr(), y.data_ptr(), static_cast<uint64_t>(x.numel()),
            static_cast<uint64_t>(x.size(2)), static_cast<uint64_t>(x.size(3))};
  auto blocks = std::min<int64_t>(8, (x.numel() + 127) / 128);
  ACL(aclrtLaunchKernelWithHostArgs(fn, blocks, stream(), nullptr, &args, sizeof(args), nullptr, 0));
  sync();
  return y;
}
at::Tensor maxpool2d_meta(const at::Tensor& x) {
  return at::empty(x.sizes(), x.options());
}
at::Tensor upsample_nearest2d(const at::Tensor& x) {
  TORCH_CHECK(!x.requires_grad(), "upsample_nearest2d is inference-only");
  TORCH_CHECK(x.device().type() == device_type, "upsample_nearest2d requires an NPU tensor");
  TORCH_CHECK(x.dim() == 4 && x.is_contiguous(), "upsample_nearest2d requires contiguous NCHW");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf, "upsample_nearest2d supports float32/float16");
  TORCH_CHECK(x.size(1) > 0 && x.size(2) > 0 && x.size(3) > 0, "C, H, W must be positive");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 32 == 0, "upsample_nearest2d requires 32-byte aligned input");
  select(x.device());
  auto y = at::empty({x.size(0), x.size(1), x.size(2)*2, x.size(3)*2}, x.options());
  if (!x.numel()) return y;
  auto fn = kernel("THIN_UPSAMPLE_BINARY", x.scalar_type() == at::kFloat ? "upsample_fp32" : "upsample_fp16");
  struct Args { void* x; void* y; uint64_t length; uint64_t height; uint64_t width; };
  Args args{x.data_ptr(), y.data_ptr(), static_cast<uint64_t>(x.numel()), static_cast<uint64_t>(x.size(2)), static_cast<uint64_t>(x.size(3))};
  auto blocks = std::min<int64_t>(8, (y.numel() + 2047) / 2048);
  ACL(aclrtLaunchKernelWithHostArgs(fn, blocks, stream(), nullptr, &args, sizeof(args), nullptr, 0));
  sync();
  return y;
}
at::Tensor upsample_nearest2d_meta(const at::Tensor& x) {
  TORCH_CHECK(!x.requires_grad(), "upsample_nearest2d is inference-only");
  TORCH_CHECK(x.dim() == 4 && x.is_contiguous(), "upsample_nearest2d requires contiguous NCHW");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf, "upsample_nearest2d supports float32/float16");
  TORCH_CHECK(x.size(1) > 0 && x.size(2) > 0 && x.size(3) > 0, "C, H, W must be positive");
  return at::empty({x.size(0), x.size(1), x.size(2)*2, x.size(3)*2}, x.options());
}
at::Tensor add_meta(const at::Tensor& x, const at::Tensor& y) {
  TORCH_CHECK(x.sizes() == y.sizes() && x.scalar_type() == at::kFloat && y.scalar_type() == at::kFloat,
              "add requires matching float32 tensors");
  return at::empty(x.sizes(), x.options());
}
at::Tensor sigmoid_meta(const at::Tensor& x) {
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf,
              "sigmoid supports float32 and float16");
  return at::empty(x.sizes(), x.options());
}
at::Tensor conv2d_meta(const at::Tensor& x, const at::Tensor& w, const std::optional<at::Tensor>& bias,
                       int64_t stride, int64_t pad, bool fp32_output) {
  TORCH_CHECK(x.dim() == 4 && w.dim() == 4 && x.size(0) == 1 && w.size(1) == x.size(1) &&
              (w.size(2) == 1 || w.size(2) == 3) && w.size(3) == w.size(2), "expected N=1, 1x1/3x3 NCHW convolution");
  TORCH_CHECK((stride == 1 || stride == 2) && pad == w.size(2)/2, "unsupported stride/padding");
  return at::empty({1, w.size(0), (x.size(2)+2*pad-w.size(2))/stride+1, (x.size(3)+2*pad-w.size(3))/stride+1},
                  x.options().dtype(fp32_output ? at::kFloat : x.scalar_type()));
}
C10_REGISTER_GUARD_IMPL(PrivateUse1, Guard);
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &allocator);
TORCH_LIBRARY(thin_backend, m) {
  m.class_<RuntimeGraph>("RuntimeGraph")
    .def(torch::init<>())
    .def("begin", &RuntimeGraph::begin)
    .def("end", &RuntimeGraph::end)
    .def("replay", &RuntimeGraph::replay)
    .def("close", &RuntimeGraph::close);
  m.def("conv2d_prepare(Tensor w) -> Tensor");
  m.def("conv2d_prepared(Tensor x, Tensor w, Tensor packed, Tensor? bias, int stride, int pad, bool fp32_output=False, bool fused=False) -> Tensor");
  m.def("conv2d_fused(Tensor x, Tensor w, Tensor? bias, int stride, int pad, bool fp32_output=False, bool fused=True) -> Tensor");
  m.def("add(Tensor x, Tensor y) -> Tensor");
  m.def("sigmoid(Tensor x) -> Tensor");
  m.def("maxpool2d(Tensor x) -> Tensor");
  m.def("upsample_nearest2d(Tensor x) -> Tensor");
  m.def("conv2d(Tensor x, Tensor w, Tensor? bias=None, int stride=1, int pad=1, bool fp32_output=False) -> Tensor");
}
TORCH_LIBRARY_IMPL(thin_backend, PrivateUse1, m) {
  m.impl("conv2d_prepare", TORCH_FN(image_conv::prepare));
  m.impl("conv2d_prepared", TORCH_FN(image_conv::conv_prepared));
  m.impl("conv2d_fused", TORCH_FN(image_conv::conv));
  m.impl("add", TORCH_FN(add));
  m.impl("sigmoid", TORCH_FN(sigmoid));
  m.impl("maxpool2d", TORCH_FN(maxpool2d));
  m.impl("upsample_nearest2d", TORCH_FN(upsample_nearest2d));
  m.impl("conv2d", TORCH_FN(conv_cube::conv));
}
TORCH_LIBRARY_IMPL(thin_backend, AutogradPrivateUse1, m) {
  m.impl("conv2d_prepare", torch::CppFunction::makeFallthrough());
  m.impl("conv2d_prepared", torch::CppFunction::makeFallthrough());
  m.impl("conv2d_fused", torch::CppFunction::makeFallthrough());
  m.impl("add", torch::CppFunction::makeFallthrough());
  m.impl("sigmoid", torch::CppFunction::makeFallthrough());
  m.impl("maxpool2d", torch::CppFunction::makeFallthrough());
  m.impl("upsample_nearest2d", torch::CppFunction::makeFallthrough());
  m.impl("conv2d", torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL(thin_backend, Meta, m) {
  m.impl("add", TORCH_FN(add_meta));
  m.impl("sigmoid", TORCH_FN(sigmoid_meta));
  m.impl("maxpool2d", TORCH_FN(maxpool2d_meta));
  m.impl("upsample_nearest2d", TORCH_FN(upsample_nearest2d_meta));
  m.impl("conv2d", TORCH_FN(conv2d_meta));
}
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", TORCH_FN(empty));
  m.impl("empty_strided", TORCH_FN(empty_strided));
  m.impl("copy_", TORCH_FN(copy));
}
}
