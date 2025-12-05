#!/usr/bin/env bash
# Lightweight runner that executes the prebuilt benchmarks from bin-hc or
# bin-cuda. Usage: ./scripts/run_suite.sh hc|cuda

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hc|cuda>" >&2
  exit 1
fi

MODE=$1
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="$ROOT_DIR/bin-$MODE"

if [[ ! -d "$BIN_DIR" ]]; then
  echo "Binary directory $BIN_DIR not found. Build with scripts/build_${MODE}.sh first." >&2
  exit 1
fi

if [[ "$MODE" == "hc" ]]; then
  FP64_BIN="$BIN_DIR/fp64_flops"
  BW_BIN="$BIN_DIR/memory_bandwidth"
  GEMM_BIN="$BIN_DIR/hcblas_gemm"
else
  FP64_BIN="$BIN_DIR/cuda_fp64_flops"
  BW_BIN="$BIN_DIR/cuda_memory_bandwidth"
  GEMM_BIN="$BIN_DIR/cublas_gemm"
fi

set -x
"$FP64_BIN" --blocks 512 --threads 256 --iterations 8192 --repeats 40
"$BW_BIN" --size 1048576 --size 67108864 --size 536870912 --iterations 50
"$GEMM_BIN" --dtype double --dtype float --dtype float16 --shape 4096x4096x4096 --shape 8192x8192x8192 --shape 16384x16384x16384 --repeats 20
set +x
