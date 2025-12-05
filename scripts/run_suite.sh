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
  VADD_BIN="$BIN_DIR/vector_add"
  MADD_BIN="$BIN_DIR/matrix_add"
else
  FP64_BIN="$BIN_DIR/cuda_fp64_flops"
  BW_BIN="$BIN_DIR/cuda_memory_bandwidth"
  GEMM_BIN="$BIN_DIR/cublas_gemm"
  VADD_BIN="$BIN_DIR/cuda_vector_add"
  MADD_BIN="$BIN_DIR/cuda_matrix_add"
fi

BW_SIZES=(
  $((1 << 20))
  $((2 << 20))
  $((4 << 20))
  $((8 << 20))
  $((16 << 20))
  $((32 << 20))
  $((64 << 20))
  $((128 << 20))
  $((256 << 20))
  $((512 << 20))
  $((1024 << 20))
)

set -x
"$FP64_BIN" --blocks 512 --threads 256 --iterations 8192 --repeats 40
BW_ARGS=(--iterations 50)
for size in "${BW_SIZES[@]}"; do
  BW_ARGS+=(--size "$size")
done
"$BW_BIN" "${BW_ARGS[@]}"
"$GEMM_BIN" --dtype double --dtype float --dtype float16 --dtype tf32 --shape 4096x4096x4096 --shape 8192x8192x8192 --shape 16384x16384x16384 --repeats 20
"$VADD_BIN" --dtype float --dtype double --size 1048576 --size 16777216 --size 67108864 --repeats 80
"$MADD_BIN" --dtype float --dtype double --shape 2048x2048 --shape 4096x4096 --shape 4096x8192 --shape 8192x8192 --repeats 40
set +x
