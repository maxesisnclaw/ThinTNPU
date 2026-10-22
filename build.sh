#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
T=$(python3 -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')
ABI=$(python3 -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
CANN=/usr/local/Ascend/ascend-toolkit/latest
BISHENG="$CANN/aarch64-linux/ccec_compiler/bin/bisheng"
cmake -S . -B build-kernels -DSOC_VERSION=ascend310p1 -DCMAKE_BUILD_TYPE=Release
cmake --build build-kernels -j8
mkdir -p build-host thin_backend
INCLUDES=(-I"$T/include" -I"$T/include/torch/csrc/api/include" -I"$CANN/include"
  -I"$CANN/include/op_common" -I"$CANN/compiler/tikcpp/include"
  -I"$CANN/compiler/ascendc/include/basic_api/impl"
  -I"$CANN/compiler/ascendc/include/basic_api/interface"
  -I"$CANN/compiler/ascendc/include/highlevel_api/impl"
  -I"$CANN/compiler/ascendc/include/highlevel_api/tiling"
  -I"$CANN/compiler/ascendc/impl/aicore/basic_api"
  -I"$CANN/pkg_inc/op_common" -I"$CANN/pkg_inc/base" -I"$CANN/pkg_inc")
c++ -std=c++17 -O2 -fPIC -D_GLIBCXX_USE_CXX11_ABI="$ABI" "${INCLUDES[@]}" \
  -c backend.cpp -o build-host/backend.o
c++ -std=c++17 -O2 -fPIC -D_GLIBCXX_USE_CXX11_ABI="$ABI" "${INCLUDES[@]}" \
  -c aten_adapter.cpp -o build-host/aten_adapter.o
# Match the precompiled elementwise kernels' FP16 overflow/NaN mode.
"$BISHENG" -std=c++17 -O3 -fPIC -D_GLIBCXX_USE_CXX11_ABI="$ABI" --npu-arch=dav-2002 -xasc "${INCLUDES[@]}" \
  -mllvm -cce-aicore-fp-ceiling=2 -c ops.cpp -o build-host/ops.o
c++ -std=c++17 -O2 -fPIC -D_GLIBCXX_USE_CXX11_ABI="$ABI" "${INCLUDES[@]}" -c decode.cpp -o build-host/decode.o
c++ -std=c++17 -O2 -fPIC -D_GLIBCXX_USE_CXX11_ABI="$ABI" "${INCLUDES[@]}" -c softmax_host.cpp -o build-host/softmax_host.o
c++ -std=c++17 -O2 -fPIC -D_GLIBCXX_USE_CXX11_ABI="$ABI" "${INCLUDES[@]}" -c nms.cpp -o build-host/nms.o
"$BISHENG" -std=c++17 -O3 -fPIC -D_GLIBCXX_USE_CXX11_ABI="$ABI" \
  --npu-arch=dav-2002 -xasc "${INCLUDES[@]}" \
  -c conv_cube.cpp -o build-host/conv_cube.o
for source in conv_img conv_c4; do
  "$BISHENG" -std=c++17 -O3 -fPIC -D_GLIBCXX_USE_CXX11_ABI="$ABI" \
    --npu-arch=dav-2002 -xasc "${INCLUDES[@]}" \
    -c "$source.cpp" -o "build-host/$source.o"
done
"$BISHENG" -shared -fPIC -Wl,--as-needed build-host/backend.o build-host/aten_adapter.o build-host/ops.o build-host/decode.o build-host/softmax_host.o build-host/nms.o build-host/conv_cube.o build-host/conv_img.o build-host/conv_c4.o \
  -L"$T/lib" -lc10 -ltorch_cpu -ltorch -L"$CANN/lib64" \
  -lacl_rt -ltiling_api -lplatform -lregister -lascendcl -lruntime \
  -Wl,-rpath,"$T/lib" -Wl,-rpath,"$CANN/lib64" \
  -o thin_backend/libthin_backend.so
readelf -d thin_backend/libthin_backend.so | grep NEEDED
