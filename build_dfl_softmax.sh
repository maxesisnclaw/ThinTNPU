#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
T=$(python3 -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')
ABI=$(python3 -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
CANN=/usr/local/Ascend/ascend-toolkit/latest
BISHENG="$CANN/aarch64-linux/ccec_compiler/bin/bisheng"
cmake --build build-kernels -j8
c++ -std=c++17 -O2 -fPIC -D_GLIBCXX_USE_CXX11_ABI="$ABI" \
    -I"$T/include" -I"$T/include/torch/csrc/api/include" -I"$CANN/include" \
    -c softmax_host.cpp -o build-host/softmax_host.o
"$BISHENG" -shared -fPIC -Wl,--as-needed \
    build-host/backend.o build-host/aten_adapter.o build-host/ops.o build-host/decode.o \
    build-host/softmax_host.o build-host/nms.o build-host/conv_cube.o build-host/conv_img.o build-host/conv_c4.o \
    -L"$T/lib" -lc10 -ltorch_cpu -ltorch -L"$CANN/lib64" \
    -lacl_rt -ltiling_api -lplatform -lregister -lascendcl -lruntime \
    -Wl,-rpath,"$T/lib" -Wl,-rpath,"$CANN/lib64" -o thin_backend/libthin_backend.so
