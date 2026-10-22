#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
T=$(python3 -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')
suffix=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
include=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
ABI=$(python3 -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
c++ -D_GLIBCXX_USE_CXX11_ABI="$ABI" -std=c++17 -O3 -shared -fPIC -fvisibility=hidden -I"$include" -I"$T/include" \
    region_guard.cpp -L"$T/lib" -ltorch_python -ltorch_cpu -ltorch -lc10 \
    -Wl,-rpath,"$T/lib" -o "_region_guard$suffix"
