#include <ATen/ATen.h>
#include <torch/library.h>
#include <acl/acl_rt.h>
#include <cmath>
#include <limits>

namespace thin {
void select(c10::Device);
aclrtStream stream();
void sync();
void check(aclError, const char*);
aclrtFuncHandle kernel(const char*, const char*);
}
namespace thin_nms {
float threshold_float(double value) {
    float result = static_cast<float>(value);
    return double(result) > value ? std::nextafter(result, -std::numeric_limits<float>::infinity()) : result;
}
std::tuple<at::Tensor, at::Tensor> nms(const at::Tensor& boxes, const at::Tensor& scores,
                                    double iou_threshold, double score_threshold, int64_t max_out) {
    TORCH_CHECK(boxes.device().type() == c10::DeviceType::PrivateUse1 && boxes.device() == scores.device(),
                "nms requires boxes and scores on the same NPU");
    TORCH_CHECK(!boxes.requires_grad() && !scores.requires_grad(), "nms is inference-only");
    TORCH_CHECK(boxes.scalar_type() == at::kFloat && scores.scalar_type() == at::kFloat,
                "nms requires float32 boxes and scores");
    TORCH_CHECK(boxes.dim() == 2 && boxes.size(1) == 4 && scores.dim() == 1 && boxes.size(0) == scores.size(0),
                "nms expects boxes [M,4], scores [M]");
    TORCH_CHECK(boxes.is_contiguous() && scores.is_contiguous() &&
                reinterpret_cast<uintptr_t>(boxes.const_data_ptr())%32 == 0 &&
                reinterpret_cast<uintptr_t>(scores.const_data_ptr())%32 == 0,
                "nms requires contiguous 32-byte aligned inputs");
    TORCH_CHECK(iou_threshold >= 0 && iou_threshold <= 1 && !std::isnan(score_threshold),
                "nms requires iou_threshold in [0,1] and non-NaN score_threshold");
    TORCH_CHECK(max_out >= 0 && max_out <= std::numeric_limits<int32_t>::max() &&
                boxes.size(0) <= std::numeric_limits<int32_t>::max(), "nms index/count must fit int32");
    thin::select(boxes.device());
    auto opts = boxes.options().dtype(at::kInt);
    auto indices = at::empty({max_out}, opts);
    auto count = at::empty({1}, opts);
    int64_t tiles=(boxes.size(0)+1023)/1024;
    auto scratch=at::empty({tiles*6*1024}, boxes.options());
    struct Args {
        void *boxes, *scores, *scratch, *indices, *count;
        uint64_t length, max_out;
        float iou_threshold, score_threshold;
    } args{boxes.data_ptr(), scores.data_ptr(), scratch.data_ptr(), indices.data_ptr(), count.data_ptr(),
           uint64_t(boxes.size(0)), uint64_t(max_out), threshold_float(iou_threshold), threshold_float(score_threshold)};
    auto fn=thin::kernel("THIN_NMS_BINARY", "nms_fp32");
    thin::check(aclrtLaunchKernelWithHostArgs(fn, 1, thin::stream(), nullptr,
                                           &args, sizeof(args), nullptr, 0), "nms launch");
    thin::sync();
    return {indices, count};
}
}
TORCH_LIBRARY_FRAGMENT(thin_backend, m) {
    m.def("nms(Tensor boxes, Tensor scores, float iou_threshold, float score_threshold, int max_out) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(thin_backend, PrivateUse1, m) { m.impl("nms", TORCH_FN(thin_nms::nms)); }
TORCH_LIBRARY_IMPL(thin_backend, AutogradPrivateUse1, m) { m.impl("nms", torch::CppFunction::makeFallthrough()); }
