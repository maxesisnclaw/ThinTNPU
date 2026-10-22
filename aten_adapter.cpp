#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/GradMode.h>
#include <acl/acl_rt.h>
#include <cstdio>
#include <cstdlib>
#include "graph_capture.h"

namespace thin {
at::Tensor sigmoid(const at::Tensor&);
at::Tensor maxpool2d(const at::Tensor&);
at::Tensor upsample_nearest2d(const at::Tensor&);
at::Tensor& copy(at::Tensor&, const at::Tensor&, bool);
void select(c10::Device);
void sync();
aclrtStream stream();
void check(aclError, const char*);
}
namespace thin_ops { at::Tensor& silu_(at::Tensor&); }
namespace conv_cube {
at::Tensor conv(const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&, int64_t, int64_t, bool);
}

namespace thin_aten {
void trace(const char* aten, const char* target) {
  const char* enabled = std::getenv("THIN_ATEN_TRACE");
  if (enabled && std::string(enabled) == "1")
    std::fprintf(stderr, "THIN_ATEN %s -> %s device=npu:0\n", aten, target);
}
int64_t square(at::IntArrayRef values, const char* name) {
  TORCH_CHECK(values.size() == 1 || values.size() == 2, "thin aten: ", name, " must have one or two elements");
  TORCH_CHECK(values.size() == 1 || values[0] == values[1], "thin aten: asymmetric ", name, " is unsupported");
  return values[0];
}
at::Tensor allocation_base(const at::Tensor& x) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "thin aten requires an NPU tensor");
  TORCH_CHECK(!x.requires_grad() || !c10::GradMode::is_enabled(), "thin aten is inference-only; use no_grad/inference_mode");
  if (x.is_contiguous() && x.storage_offset() == 0) return x.detach();
  auto out = at::empty(x.sizes(), x.options());
  thin::copy(out, x, false);
  trace("pack", "empty + copy_ (NPU D2D, offset=0)");
  return out;
}
at::Tensor convolution(const at::Tensor& input, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {
  TORCH_CHECK(!transposed && groups == 1, "thin aten convolution: transposed/groups are unsupported");
  TORCH_CHECK(input.dim() == 4 && weight.dim() == 4 && input.size(0) == 1,
              "thin aten convolution requires N=1, 2D NCHW/OIHW");
  const auto s = square(stride, "stride"), p = square(padding, "padding");
  TORCH_CHECK(square(dilation, "dilation") == 1 && square(output_padding, "output_padding") == 0,
              "thin aten convolution requires dilation=1 and output_padding=0");
  const auto k = weight.size(2);
  TORCH_CHECK((k == 1 || k == 3) && weight.size(3) == k && p == k / 2 && (s == 1 || s == 2),
              "thin aten convolution supports only k1/p0 or k3/p1, stride=1/2");
  auto x = allocation_base(input), w = allocation_base(weight);
  std::optional<at::Tensor> b = bias && bias->defined() ? std::optional<at::Tensor>(allocation_base(*bias)) : std::nullopt;
  trace("aten::convolution", "conv_cube::conv");
  return conv_cube::conv(x, w, b, s, p, false);
}
std::tuple<at::Tensor, at::Tensor> max_pool2d_with_indices(const at::Tensor& input,
    at::IntArrayRef kernel, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool ceil_mode) {
  TORCH_CHECK(square(kernel, "kernel_size") == 5 &&
              (stride.empty() ? 5 : square(stride, "stride")) == 1 &&
              square(padding, "padding") == 2 && square(dilation, "dilation") == 1 && !ceil_mode,
              "thin aten maxpool supports only k5/s1/p2/d1, ceil_mode=False");
  auto x = allocation_base(input);
  trace("aten::max_pool2d_with_indices", "thin::maxpool2d (indices=zeros placeholder)");
  auto values = thin::maxpool2d(x);
  auto indices = at::empty(values.sizes(), values.options().dtype(at::kLong));
  if (indices.numel()) {
    thin::select(indices.device());
    thin::sync();
    if (thin::capturing())
      thin::check(aclrtMemsetAsync(indices.data_ptr(), indices.nbytes(), 0, indices.nbytes(), thin::stream()), "zero placeholder indices async");
    else
      thin::check(aclrtMemset(indices.data_ptr(), indices.nbytes(), 0, indices.nbytes()), "zero placeholder indices");
  }
  return {values, indices};
}
void validate_scale(std::optional<double> scale) {
  TORCH_CHECK(!scale || *scale == 2.0, "thin aten nearest2d supports scale=2 only");
}
at::Tensor upsample_nearest2d(const at::Tensor& input, at::IntArrayRef output_size,
    std::optional<double> scales_h, std::optional<double> scales_w) {
  TORCH_CHECK(input.dim() == 4 && output_size.size() == 2 &&
              output_size[0] == input.size(2) * 2 && output_size[1] == input.size(3) * 2,
              "thin aten nearest2d requires a 2x output size");
  validate_scale(scales_h); validate_scale(scales_w);
  trace("aten::upsample_nearest2d", "thin::upsample_nearest2d");
  return thin::upsample_nearest2d(allocation_base(input));
}
at::Tensor sigmoid(const at::Tensor& input) {
  trace("aten::sigmoid", "thin::sigmoid");
  return thin::sigmoid(allocation_base(input));
}
at::Tensor silu(const at::Tensor& input) {
  TORCH_CHECK(!input.requires_grad() || !c10::GradMode::is_enabled(), "thin aten silu is inference-only; use no_grad/inference_mode");
  auto out = at::empty(input.sizes(), input.options());
  thin::copy(out, input, false);
  trace("aten::silu", "thin_ops::silu_ on NPU copy");
  thin_ops::silu_(out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("convolution", TORCH_FN(convolution));
  m.impl("max_pool2d_with_indices", TORCH_FN(max_pool2d_with_indices));
  m.impl("upsample_nearest2d", TORCH_FN(upsample_nearest2d));
  m.impl("sigmoid", TORCH_FN(sigmoid));
  m.impl("silu", TORCH_FN(silu));
}
}
