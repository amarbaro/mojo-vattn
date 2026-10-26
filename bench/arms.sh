#!/usr/bin/env bash
# S3 round on the real dump: gates first (fail-closed), then REPS repeats alternating arm order.
# Usage (GPU): gpu-wait run --vram 8 --timeout 1800 -- bench/arms.sh REPS ITERS BIN...
set -euo pipefail
cd "$(dirname "$0")/.."
REPS=$1; ITERS=$2; shift 2
PY=$HOME/Projects/imports/ComfyUI/.venv/bin/python
real=.work/real
read -r L H < <(python3 -c "import json;m=json.load(open('$real/meta.json'));print(m['L'],m['H'])")
for b in "$@"; do
  BIN=$b bench/parity.sh .work/case-129-3 .work/case-4097-2 $real || { echo "FAIL gate $b"; exit 1; }
done
for r in $(seq 1 "$REPS"); do
  arms=("sdpa" "$@"); (( r % 2 == 0 )) && arms=("$@" "sdpa")
  for a in "${arms[@]}"; do
    if [ "$a" = sdpa ]; then echo "rep $r $($PY tools/sdpa-bench.py $real "$ITERS")"
    else echo "rep $r $a $("$a" $real "$L" "$H" "$ITERS" | tail -1)"; fi
  done
done
