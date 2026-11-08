#!/usr/bin/env bash
# CI entry point, same on a GitHub runner and a local box: install, then build every target.
# CPU only. The numeric gates need a GPU and live in scripts/gpu-gates.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
v=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml); c=$(sed -n 's/^- \[v\([0-9.]*\)\].*/\1/p' CHANGELOG.md | head -1)
[ "$v" = "$c" ] || { echo "FAIL version: pyproject $v vs newest CHANGELOG entry $c"; exit 1; }
command -v uv >/dev/null || { echo "FAIL setup: uv not installed"; exit 1; }
if [ -z "${MOJO:-}" ]; then
  uv sync || { echo "FAIL setup: uv sync"; exit 1; }
  MOJO=.venv/bin/mojo
fi
export MOJO
mkdir -p .work
M=$MOJO
build() { # name source [mojo flags...]
  local out=".work/$1" src=$2; shift 2
  $M build --target-accelerator amdgpu:gfx1100 -I src "$@" "$src" -o "$out" 2>.work/build.err || { echo "FAIL build: $src $*"; cat .work/build.err; exit 1; }
}
build run_vattn tests/run_vattn.mojo
build run_vattn_128_64 tests/run_vattn.mojo -D VATTN_BM=128 -D VATTN_BN=64
build quickstart examples/quickstart.mojo
# negative control: a tile config the loaders cannot cover must fail at compile time, not corrupt output
if $M build --target-accelerator amdgpu:gfx1100 -I src -D VATTN_BM=96 tests/run_vattn.mojo -o .work/run_vattn_96 2>.work/build.err; then
  echo "FAIL assert: -D VATTN_BM=96 built (loaders would cover 384 of 512 vectors)"; exit 1
fi
grep -q "multiples of 16" .work/build.err || { echo "FAIL assert: VATTN_BM=96 failed for another reason"; cat .work/build.err; exit 1; }
if [ -e /dev/kfd ]; then  # the example launches the kernel: only where an AMD GPU is present
  .work/quickstart || { echo "FAIL example: examples/quickstart.mojo"; exit 1; }
  echo "PASS ci"
else
  echo "PASS ci (build only: no GPU here, examples/quickstart and scripts/gpu-gates.sh not run)"
fi
