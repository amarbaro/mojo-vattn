#!/usr/bin/env bash
# CI entry point, same on a GitHub runner and a local box: install, then build every target.
# CPU only. The numeric gates need a GPU and live in scripts/gpu-gates.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "FAIL setup: uv not installed"; exit 1; }
uv sync || { echo "FAIL setup: uv sync"; exit 1; }
mkdir -p .work
M=.venv/bin/mojo
build() { # name source [mojo flags...]
  local out=".work/$1" src=$2; shift 2
  $M build -I src "$@" "$src" -o "$out" 2>.work/build.err || { echo "FAIL build: $src $*"; cat .work/build.err; exit 1; }
}
build run_vattn tests/run_vattn.mojo
build run_vattn_128_64 tests/run_vattn.mojo -D VATTN_BM=128 -D VATTN_BN=64
build quickstart examples/quickstart.mojo
.work/quickstart || { echo "FAIL example: examples/quickstart.mojo"; exit 1; }
echo "PASS ci"
