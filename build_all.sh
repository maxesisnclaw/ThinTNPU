#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash build.sh
bash build_blocked_c2f.sh
bash build_flow_conv.sh
bash build_flow_ops.sh
