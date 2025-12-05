#!/usr/bin/env bash
# Build helper for compiling the benchmarks directly with nvcc/cuBLAS
# without going through CMake. CUDA toolchain paths are inferred from
# CUDA_HOME or default installation prefixes.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR=${OUT_DIR:-"$ROOT_DIR/bin-cuda"}
mkdir -p "$OUT_DIR"

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
NVCC=${NVCC:-"$CUDA_HOME/bin/nvcc"}

COMMON_FLAGS=(
  -std=c++14 -O3
  -I"$ROOT_DIR/include"
)

LINK_FLAGS=(
  -L"$CUDA_HOME/lib64" -lcudart -lcublas
)

set -x
"$NVCC" "${COMMON_FLAGS[@]}" "$ROOT_DIR/src/cuda_fp64_flops.cu" -o "$OUT_DIR/cuda_fp64_flops" -lcudart
"$NVCC" "${COMMON_FLAGS[@]}" "$ROOT_DIR/src/cuda_memory_bandwidth.cu" -o "$OUT_DIR/cuda_memory_bandwidth" -lcudart
"$NVCC" "${COMMON_FLAGS[@]}" "$ROOT_DIR/src/cublas_gemm.cu" -o "$OUT_DIR/cublas_gemm" "${LINK_FLAGS[@]}"
set +x

echo "Built CUDA benchmarks under $OUT_DIR"
