#!/usr/bin/env bash
# Build and run hc benchmarks without CMake.
# Requires HTHPCC toolchain; configure HPCC_PATH, HTCXX, etc.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OUT_DIR=${OUT_DIR:-"$ROOT_DIR/bin-hc"}

HPCC_PATH=${HPCC_PATH:-/opt/hpcc}
HT_INCLUDE_DIR=${HT_INCLUDE_DIR:-$HPCC_PATH/include}
HT_LIB_DIR=${HT_LIB_DIR:-$HPCC_PATH/lib}
HTCXX=${HTCXX:-$HPCC_PATH/bin/htcc}
export ISU_FASTMODEL=${ISU_FASTMODEL:-1}

mkdir -p "$OUT_DIR"

COMMON_FLAGS=(
  -O3
  -I"$ROOT_DIR/include"
  -I"$HT_INCLUDE_DIR"
)

LINK_FLAGS=(
  -L"$HT_LIB_DIR"
  -lhc_runtime
  -lhcblas
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

echo "Collecting device information..."
"$OUT_DIR/device_info"

FP64_CASES=(
  "256 256 4096 60"
  "512 256 8192 50"
  "512 512 8192 40"
  "768 512 4096 30"
)

echo "Running FP64 compute sweep..."
for case in "${FP64_CASES[@]}"; do
  read -r blocks threads iters reps <<<"$case"
  "$OUT_DIR/fp64_flops" --blocks "$blocks" --threads "$threads" --iterations "$iters" --repeats "$reps"
done

echo "Running memory bandwidth sweep..."
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
BW_ARGS=(--iterations 50)
for size in "${BW_SIZES[@]}"; do
  BW_ARGS+=(--size "$size")
done
"$OUT_DIR/memory_bandwidth" "${BW_ARGS[@]}"

echo "Running hcBLAS GEMM sweep..."
"$OUT_DIR/hcblas_gemm" \
  --repeats 20 \
  --dtype double --dtype float --dtype float16 --dtype tf32 \
  --shape 1024x1024x1024 \
  --shape 2048x2048x2048 \
  --shape 3072x3072x3072 \
  --shape 4096x4096x4096 \
  --shape 6144x6144x6144 \
  --shape 8192x8192x8192 \
  --shape 12288x12288x12288 \
  --shape 16384x16384x16384

echo "Running vector add sweep..."
"$OUT_DIR/vector_add" \
  --repeats 80 \
  --dtype float --dtype double \
  --size $((1 << 20)) \
  --size $((1 << 22)) \
  --size $((1 << 24)) \
  --size $((1 << 26))

echo "Running matrix add sweep..."
"$OUT_DIR/matrix_add" \
  --repeats 40 \
  --dtype float --dtype double \
  --shape 2048x2048 \
  --shape 4096x4096 \
  --shape 4096x8192 \
  --shape 8192x8192

echo "hc run completed"
