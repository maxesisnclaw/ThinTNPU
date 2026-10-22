#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
model=${1:?Usage: run_capture_comparison.sh MODEL REFERENCE OUTPUT}
reference=${2:?Supply a trusted CPU reference fixture}
output=${3:?Supply a new output directory}
mkdir "$output"
output=$(cd "$output" && pwd)
source env.sh
for round in 1 2 3 4 5; do
    if (( round % 2 )); then variants=(thin-whole native-capture); else variants=(native-capture thin-whole); fi
    printf '%s\n' "${variants[*]}" > "$output/order-$round.txt"
    for variant in "${variants[@]}"; do
        python3 benchmark_backends.py --variant "$variant" --model "$model" \
            --reference "$reference" --warmup 30 --iterations 101 \
            --output "$output/$round-$variant" > "$output/$round-$variant.log" 2>&1
    done
done