#!/usr/bin/env bash
# Simple build helper for compiling MXGPU benchmarks against the native hc SDK
# without relying on CMake. Configure include/library paths via env vars.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR=${OUT_DIR:-"$ROOT_DIR/bin-hc"}
mkdir -p "$OUT_DIR"

HPCC_PATH=/opt/hpcc
HTCXX=${HTCXX:-htcc}
HT_INCLUDE_DIR=${HT_INCLUDE_DIR:-$HPCC_PATH/include}
HT_LIB_DIR=${HT_LIB_DIR:-$HPCC_PATH/lib:$HPCC_PATH/lib64}

COMMON_FLAGS=(
  -std=c++14 -O3
  -I"$ROOT_DIR/include"
  -I"$HT_INCLUDE_DIR"
)

LINK_FLAGS=(
  -L"$HT_LIB_DIR" -lhcruntime -lhcblas
)

set -x
"$HTCXX" "${COMMON_FLAGS[@]}" "$ROOT_DIR/src/fp64_flops.cu" -o "$OUT_DIR/fp64_flops" "${LINK_FLAGS[@]}"
"$HTCXX" "${COMMON_FLAGS[@]}" "$ROOT_DIR/src/memory_bandwidth.cu" -o "$OUT_DIR/memory_bandwidth" "${LINK_FLAGS[@]}"
"$HTCXX" "${COMMON_FLAGS[@]}" "$ROOT_DIR/src/hcblas_gemm.cu" -o "$OUT_DIR/hcblas_gemm" "${LINK_FLAGS[@]}"
"$HTCXX" "${COMMON_FLAGS[@]}" "$ROOT_DIR/src/vector_add.cu" -o "$OUT_DIR/vector_add" "${LINK_FLAGS[@]}"
"$HTCXX" "${COMMON_FLAGS[@]}" "$ROOT_DIR/src/matrix_add.cu" -o "$OUT_DIR/matrix_add" "${LINK_FLAGS[@]}"
"$HTCXX" "${COMMON_FLAGS[@]}" "$ROOT_DIR/src/device_info.cu" -o "$OUT_DIR/device_info" "${LINK_FLAGS[@]}"
set +x

echo "Built hc benchmarks under $OUT_DIR"
