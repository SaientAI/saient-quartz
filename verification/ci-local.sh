#!/usr/bin/env bash
# Dependency-free local validation; no private model required by default.
set -euo pipefail
cd "$(dirname "$0")/.."
export RAYON_NUM_THREADS=2
export CARGO_BUILD_JOBS=2
export CARGO_INCREMENTAL=0
echo "QUARTZ_VERIFY_CI: Python verifier self-tests"
python3 -m unittest discover -s verification -p 'test_*.py' -v
echo "QUARTZ_VERIFY_CI: native LoRA Rust math tests"
cargo test --bin quartz train:: --offline --quiet -- --test-threads=2
if [ "$#" -ge 1 ] && [ "$1" = "--with-audit" ]; then
  echo "QUARTZ_VERIFY_CI: pinned private GGUF/adapter structural audit"
  python3 verification/verify.py --mode audit
fi
echo "QUARTZ_VERIFY_CI: PASS"
