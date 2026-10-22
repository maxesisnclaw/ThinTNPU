#include <ATen/ATen.h>
#include <torch/library.h>

namespace thin { at::Tensor& copy(at::Tensor&, const at::Tensor&, bool); }
namespace thin_decode {
at::Tensor& arange_out(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step, at::Tensor& out) {
  auto constant = at::arange(start, end, step, out.options().device(at::kCPU));
  if (out.sizes() != constant.sizes()) {
    auto replacement = at::empty(constant.sizes(), out.options());
    out.unsafeGetTensorImpl()->set_storage_keep_dtype(replacement.storage());
    out.unsafeGetTensorImpl()->set_storage_offset(0);
    out.unsafeGetTensorImpl()->set_sizes_contiguous(constant.sizes());
  }
  return thin::copy(out, constant, false);
}
at::Tensor& arange_end_out(const at::Scalar& end, at::Tensor& out) {
  return arange_out(0, end, 1, out);
}
at::Tensor& fill(at::Tensor& out, const at::Scalar& value) {
  auto constant = at::full(out.sizes(), value, out.options().device(at::kCPU));
  return thin::copy(out, constant, false);
}
at::Scalar local_scalar(const at::Tensor& x) {
  TORCH_CHECK(x.numel() == 1, "_local_scalar_dense requires one element");
  auto host = at::empty(x.sizes(), x.options().device(at::kCPU));
  thin::copy(host, x, false);
  return host.item();
}
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("arange.start_out", TORCH_FN(arange_out));
  m.impl("arange.out", TORCH_FN(arange_end_out));
  m.impl("fill_.Scalar", TORCH_FN(fill));
  m.impl("_local_scalar_dense", TORCH_FN(local_scalar));
}
}
