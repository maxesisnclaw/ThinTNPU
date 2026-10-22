#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
bash build.sh 2>&1 | tee build-full.log
mkdir -p evidence-full
readelf -d thin_backend/libthin_backend.so > evidence-full/dependencies.txt
ldd thin_backend/libthin_backend.so > evidence-full/ldd.txt
nm -D thin_backend/libthin_backend.so > evidence-full/symbols.txt
sha256sum backend.cpp ops.cpp conv_cube.cpp add.cpp sigmoid.cpp maxpool.cpp upsample.cpp elementwise.cpp softmax.cpp softmax_host.cpp nms.cpp nms_kernels.cpp thin_backend/__init__.py thin_backend/libthin_backend.so build-kernels/*_merge_obj_dir/device.o > evidence-full/artifacts.sha256
python3 -u verify_full.py 2>&1 | tee verify-full.log
