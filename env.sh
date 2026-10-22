export TORCH_DEVICE_BACKEND_AUTOLOAD=0
THIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export THIN_ADD_BINARY="$THIN_ROOT/build-kernels/add_kernels_merge_obj_dir/device.o"
export THIN_SIGMOID_BINARY="$THIN_ROOT/build-kernels/sigmoid_kernels_merge_obj_dir/device.o"
export THIN_MAXPOOL_BINARY="$THIN_ROOT/build-kernels/maxpool_kernels_merge_obj_dir/device.o"
export THIN_UPSAMPLE_BINARY="$THIN_ROOT/build-kernels/upsample_kernels_merge_obj_dir/device.o"
export THIN_ELEMENTWISE_BINARY="$THIN_ROOT/build-kernels/elementwise_kernels_merge_obj_dir/device.o"
export THIN_SOFTMAX_BINARY="$THIN_ROOT/build-kernels/softmax_kernels_merge_obj_dir/device.o"
unset THIN_ROOT
export THIN_NMS_BINARY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build-kernels/nms_kernels_merge_obj_dir/device.o"
