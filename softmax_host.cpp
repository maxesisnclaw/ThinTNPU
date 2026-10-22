#include <ATen/ATen.h>
#include <ATen/WrapDimUtils.h>
#include <torch/library.h>
#include <acl/acl_rt.h>

namespace thin {
void check(aclError, const char*);
void select(c10::Device);
aclrtStream stream();
void sync();
aclrtFuncHandle kernel(const char*, const char*);
}
namespace thin_ops {
at::Tensor packed(const at::Tensor&);
at::Tensor& copy_strided(at::Tensor&, const at::Tensor&, bool);
}
namespace thin_softmax {
at::Tensor padded_view(const at::Tensor& x, at::ScalarType dtype, int64_t cut, int64_t pitch, int64_t elements) {
    auto storage = at::empty({elements}, x.options().dtype(dtype));
    auto strides = x.sizes().vec();
    int64_t stride = 1;
    for (int64_t d = x.dim() - 1; d >= 0; --d) {
        if (d == cut - 1) stride = pitch;
        strides[d] = stride;
        stride *= x.size(d);
    }
    return storage.as_strided(x.sizes(), strides);
}
bool dfl_layout(const at::Tensor& x, int64_t dim, at::ScalarType dtype) {
    if (x.dim() != 4 || dim != 1 || x.size(1) != 16 || x.size(2) != 4 ||
        x.stride(3) != 1 || x.storage_offset() % 16 != 0 || x.size(3) % 16 != 0 || x.stride(1) < x.size(3)) return false;
    for (int64_t d = 0; d < 3; ++d)
        if (x.stride(d) % 16 != 0) return false;
    // DataCopy uses 32-byte blocks and 16-bit inter-block strides.
    const int64_t output_bytes = dtype == at::kHalf ? 2 : 4;
    return (x.stride(1) - 16) * x.element_size() / 32 <= 65535 &&
           (4 * x.size(3) - 16) * output_bytes / 32 <= 65535;
}
void dfl_softmax(const at::Tensor& x, at::Tensor& out, bool half_to_float) {
    std::string symbol = x.scalar_type() == at::kFloat ? "softmax_dfl_fp32" : "softmax_dfl_fp16";
    if (half_to_float) symbol += "_float";
    auto fn = thin::kernel("THIN_SOFTMAX_BINARY", symbol.c_str());
    struct Args { void* x; void* y; uint64_t batches, width, batch_stride, reduce_stride, box_stride; };
    Args args{x.data_ptr(), out.data_ptr(), static_cast<uint64_t>(x.size(0)), static_cast<uint64_t>(x.size(3)),
              static_cast<uint64_t>(x.stride(0)), static_cast<uint64_t>(x.stride(1)), static_cast<uint64_t>(x.stride(2))};
    const int64_t tasks = x.size(0) * 4 * ((x.size(3) + 1055) / 1056);
    thin::check(aclrtLaunchKernelWithHostArgs(fn, std::min<int64_t>(8, tasks), thin::stream(), nullptr,
                &args, sizeof(args), nullptr, 0), "DFL softmax launch");
    thin::sync();
}
at::Tensor softmax(const at::Tensor& x, int64_t dim, bool half_to_float) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && !x.requires_grad(), "softmax requires inference-only NPU input");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf, "softmax supports float32/float16");
    TORCH_CHECK(!half_to_float || x.scalar_type() == at::kHalf, "half_to_float requires float16 input");
    dim = at::maybe_wrap_dim(dim, x.dim());
    thin::select(x.device());
    auto dtype = half_to_float ? at::kFloat : x.scalar_type();
    auto out = at::empty(x.sizes(), x.options().dtype(dtype));
    if (!x.numel()) return out;
    if (dfl_layout(x, dim, dtype)) {
        dfl_softmax(x, out, half_to_float);
        return out;
    }
    int64_t outer = 1, inner = 1;
    const int64_t reduction = x.dim() ? x.size(dim) : 1;
    for (int64_t d = 0; d < dim; ++d) outer *= x.size(d);
    for (int64_t d = dim + 1; d < x.dim(); ++d) inner *= x.size(d);
    const bool last = inner == 1;
    const int64_t width = last ? reduction : inner;
    const int64_t pitch = (width + 15) / 16 * 16;
    const int64_t elements = last ? outer * pitch : outer * reduction * pitch;
    const int64_t cut = last ? dim : dim + 1;
    auto input = pitch == width ? thin_ops::packed(x) : padded_view(x, x.scalar_type(), cut, pitch, elements);
    if (pitch != width) thin_ops::copy_strided(input, x, false);
    auto result = pitch == width ? out : padded_view(x, dtype, cut, pitch, elements);
    std::string symbol = last ? "softmax_last" : "softmax_inner";
    symbol += x.scalar_type() == at::kFloat ? "_fp32" : "_fp16";
    if (half_to_float) symbol += "_float";
    auto fn = thin::kernel("THIN_SOFTMAX_BINARY", symbol.c_str());
    struct Args { void* x; void* y; uint64_t outer; uint64_t reduction; uint64_t inner; uint64_t pitch; };
    Args args{input.data_ptr(), result.data_ptr(), static_cast<uint64_t>(outer), static_cast<uint64_t>(reduction),
              static_cast<uint64_t>(inner), static_cast<uint64_t>(pitch)};
    const int64_t tasks = last ? outer : outer * ((inner + 2047) / 2048);
    thin::check(aclrtLaunchKernelWithHostArgs(fn, std::min<int64_t>(8, tasks), thin::stream(), nullptr,
                &args, sizeof(args), nullptr, 0), "softmax launch");
    thin::sync();
    if (pitch != width) thin_ops::copy_strided(out, result, false);
    return out;
}
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_softmax", TORCH_FN(softmax));
}
}
