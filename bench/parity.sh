#!/usr/bin/env bash
# S2 parity gate: run vattn_fwd on every case dir given, compare against the fp64/f32 reference.
# Usage (GPU): gpu-wait run --vram 4 --timeout 600 -- bench/parity.sh .work/case-*
set -euo pipefail
cd "$(dirname "$0")/.."
fails=()
for d in "$@"; do
  read -r L H < <(python3 -c "import json;m=json.load(open('$d/meta.json'));print(m['L'],m['H'])")
  if ! "${BIN:-.work/run_vattn}" "$d" "$L" "$H" 0 > "$d/run.log" 2>&1; then
    echo "FAIL run $d: see $d/run.log"; fails+=("$d"); continue
  fi
  python3 tools/vattn-ref.py check "$d" || fails+=("$d")
done
[ ${#fails[@]} -eq 0 ] || { echo "FAIL ${#fails[@]}/$# ${fails[*]}"; exit 1; }
echo "PASS $#/$#"
