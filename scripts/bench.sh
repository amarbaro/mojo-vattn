#!/usr/bin/env bash
# Speed at the real Wan 2.2 shape (bench/prereg-S3.md) against torch SDPA: parity gates first, then 3 alternating
# rounds of ITERS calls (default 5), all in one gpu-wait job. Needs .work/real (the dump) and the ComfyUI venv.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -e .work/real/meta.json ] || { echo "FAIL setup: no .work/real dump"; exit 1; }
[ -x .work/run_vattn ] || .venv/bin/mojo build -O3 -I src tests/run_vattn.mojo -o .work/run_vattn \
  || { echo "FAIL build: tests/run_vattn.mojo"; exit 1; }
gpu-wait run --vram 8 --timeout 900 --label vattn-bench -- bench/arms.sh 3 "${ITERS:-5}" .work/run_vattn
