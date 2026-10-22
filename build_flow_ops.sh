#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
T=$(python3 -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')
ABI=$(python3 -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
CANN=/usr/local/Ascend/ascend-toolkit/latest
BISHENG="$CANN/aarch64-linux/ccec_compiler/bin/bisheng"
INCLUDES=(-I"$T/include" -I"$T/include/torch/csrc/api/include" -I"$CANN/include"
  -I"$CANN/include/op_common" -I"$CANN/compiler/tikcpp/include"
  -I"$CANN/compiler/ascendc/include/basic_api/impl"
  -I"$CANN/compiler/ascendc/include/basic_api/interface"
  -I"$CANN/compiler/ascendc/include/highlevel_api/impl"
  -I"$CANN/compiler/ascendc/include/highlevel_api/tiling"
  -I"$CANN/compiler/ascendc/impl/aicore/basic_api"
  -I"$CANN/pkg_inc/op_common" -I"$CANN/pkg_inc/base" -I"$CANN/pkg_inc")
mkdir -p build-host thin_backend
"$BISHENG" -std=c++17 -O3 -fPIC -D_GLIBCXX_USE_CXX11_ABI="$ABI" --npu-arch=dav-2002 -xasc "${INCLUDES[@]}" \
  -c flow_ops.cpp -o build-host/flow_ops.o
"$BISHENG" -shared -fPIC -Wl,--as-needed build-host/flow_ops.o \
  -L"$PWD/thin_backend" -lthin_backend -L"$T/lib" -lc10 -ltorch_cpu -ltorch -L"$CANN/lib64" \
  -lacl_rt -lascendcl -lruntime -Wl,-rpath,'$ORIGIN' -Wl,-rpath,"$T/lib" -Wl,-rpath,"$CANN/lib64" \
  -o thin_backend/libflow_ops.so
