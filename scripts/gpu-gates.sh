#!/usr/bin/env bash
# GPU parity gate: generate cases, run vattn_fwd, compare with the fp64 reference. Needs a gfx1100 GPU.
# Usage: scripts/gpu-gates.sh   (wrap in `gpu-wait run --vram 4 --timeout 600 --` on a shared box)
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --group oracle || { echo "FAIL setup: uv sync"; exit 1; }
export PATH="$PWD/.venv/bin:$PATH"
[ -x .work/run_vattn ] || { echo "FAIL setup: run scripts/ci.sh first"; exit 1; }
cases=()
for c in "129 4 3" "1000 2 2" "4097 2 2"; do
  set -- $c
  d=".work/case-$1-$3"; mkdir -p "$d"
  python3 tools/vattn-ref.py gen "$d" "$1" "$2" "$3" >/dev/null || { echo "FAIL gen: $d"; exit 1; }
  cases+=("$d")
done
bench/parity.sh "${cases[@]}"
echo "PASS gpu-gates"
