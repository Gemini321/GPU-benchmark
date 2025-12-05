# GPU-benchmark

Toolkit for evaluating MXGPU/hc devices with three micro-benchmarks: raw FP64 throughput, global-memory bandwidth, and dense DGEMM throughput via hcBLAS. The code mirrors CUDA samples but swaps CUDA runtime calls with their `hc*` counterparts so it runs natively on MXGPU hardware. When the hc SDK is unavailable you can still build and iterate via CUDA by passing `-DUSE_CUDA_FALLBACK=ON`.

## Layout

- `src/fp64_flops.cu` – custom FP64 fused-multiply-add stress test, configurable grid/block sizes and loop counts.
- `src/memory_bandwidth.cu` – STREAM-style H2D/D2H/D2D bandwidth sweep that uses pinned host buffers and event timing.
- `src/hcblas_gemm.cu` – multi-shape GEMM benchmark exercising FP64 (`hcblasDgemm`), FP32 (`hcblasSgemm`), and FP16 (`hcblasGemmEx`)，兼容矩形/正方矩阵。
- `include/benchmark_utils_cuda.h`, `src/cuda_*.cu`, `src/cublas_gemm.cu` – pure CUDA/cuBLAS equivalents for running the same tests on NVIDIA GPUs locally.
- `scripts/run_benchmarks.py` – helper that configures CMake, builds the targets, and runs a curated sweep of FP64/bandwidth/GEMM cases.

## Building

```bash
cd /root/GPU-benchmark
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DHC_RUNTIME_LIB=/path/to/libhc_runtime.so \
      -DHCBLAS_LIB=/path/to/libhcblas.so
cmake --build build -j
```

If you need to validate on CUDA hardware first, append `-DUSE_CUDA_FALLBACK=ON` and the build links against CUDA's `cudart`/`cublas` automatically.
To entirely skip hc detection (e.g., no hc SDK present and you only want CUDA binaries), set `-DBUILD_HC_BENCHES=OFF` and enable `-DBUILD_NVIDIA_BENCHES=ON` instead.
By default the CUDA build targets `sm_70`, `sm_75`, and `sm_80` (set through `CMAKE_CUDA_ARCHITECTURES`); override with `-DCMAKE_CUDA_ARCHITECTURES=<list>` if your GPU requires a different set to avoid PTX/toolchain mismatches.

## Automated test sweep

```bash
cd /root/GPU-benchmark
./scripts/run_benchmarks.py                                # uses hc SDK
./scripts/run_benchmarks.py --use-cuda-fallback            # hc API routed through CUDA
./scripts/run_benchmarks.py --platform cuda --skip-build   # CUDA-only binaries (no hc SDK)
./scripts/run_benchmarks.py --platform both                # run both hc and CUDA variants
```

The script performs the following:

1. **FP64 compute** – four kernel shapes covering 256–768 blocks × 256–512 threads with 4K–8K inner-loop iterations，覆盖更高占用率与指令混合。
2. **Memory bandwidth** – H2D/D2H/D2D 传输覆盖 1MB 到 1.5GB 多个尺寸（1、4、16、64、256、512、1024、1536 MB），更细致地观察 cache/TLB 与分页影响。
3. **hcBLAS / cuBLAS GEMM** – 同时测试 FP64、FP32、FP16（FP16 走 GEMMEx/Compute32F），覆盖 1024³ 到 16384³（及多个矩形尺寸）的多种矩阵形状，并重复 20 次以稳定计时。

Each executable is also independently configurable:

```bash
# FP64 kernel
build/bin/fp64_flops --blocks 512 --threads 256 --iterations 8192 --repeats 40

# Memory bandwidth sweep (custom sizes in bytes)
build/bin/memory_bandwidth --size 33554432 --size 134217728 --iterations 100

# GEMM shapes / dtypes
build/bin/hcblas_gemm --dtype double --dtype float --dtype float16 \\
    --shape 3072x3072x3072 --shape 6144x6144x6144 --shape 12288x12288x12288 --repeats 30
```

The binaries print time-per-iteration and sustained throughput (GFLOP/s or GB/s). Combine these measurements with device telemetry (frequency, temperature, power) from mxGPU tools to understand stability and throttling limits.

### hcBLAS / cuBLAS options

- `--shape MxNxK`：可重复多次指定，默认覆盖从 1024³ 到 16384³ 的 8 个大小（等比例放大）。
- `--dtype {float,double,float16}`：可重复，用于并行测试 FP32/FP64/FP16；默认三种都会跑。
- `--repeats N`：每种尺寸、数据类型重复次数。

## NVIDIA / CUDA-only workflow

When you want to validate the benchmarks entirely on an NVIDIA GPU (without the hc SDK), build the CUDA set of binaries:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_HC_BENCHES=OFF -DBUILD_NVIDIA_BENCHES=ON
cmake --build build -j
```

You can either run each executable manually (`build/bin/cuda_fp64_flops`, `build/bin/cuda_memory_bandwidth`, `build/bin/cublas_gemm`) or let the script orchestrate everything:

```bash
./scripts/run_benchmarks.py --platform cuda --skip-build   # assumes build/bin already exists
```

This produces the same metrics as the hc variant, but every kernel/callsite uses native CUDA/cuBLAS symbols so you can profile and iterate directly on an NVIDIA workstation.
