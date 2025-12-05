#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build"
BIN_DIR = BUILD_DIR / "bin"

FP64_CASES = [
    {"blocks": 256, "threads": 256, "iterations": 1 << 12, "repeats": 60},
    {"blocks": 512, "threads": 256, "iterations": 1 << 13, "repeats": 50},
    {"blocks": 512, "threads": 512, "iterations": 1 << 13, "repeats": 40},
    {"blocks": 768, "threads": 512, "iterations": 1 << 12, "repeats": 30},
]

BANDWIDTH_SIZES = [
    1 << 20,
    2 << 20,
    4 << 20,
    8 << 20,
    16 << 20,
    32 << 20,
    64 << 20,
    128 << 20,
    256 << 20,
    512 << 20,
    1024 << 20,
]

GEMM_SHAPES = [
    "1024x1024x1024",
    "2048x2048x2048",
    "3072x3072x3072",
    "4096x4096x4096",
    "6144x6144x6144",
    "8192x8192x8192",
    "12288x12288x12288",
    "16384x16384x16384",
]

GEMM_DTYPES = ["double", "float", "float16", "tf32"]

VECTOR_SIZES = [1 << 20, 1 << 22, 1 << 24, 1 << 26]
VECTOR_DTYPES = ["float", "double"]

MATRIX_SHAPES = ["2048x2048", "4096x4096", "4096x8192", "8192x8192"]
MATRIX_DTYPES = ["float", "double"]


def run_cmd(cmd, cwd=REPO_ROOT):
    print(f"[cmd] {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def configure(args):
    build_hc = args.platform in ("hc", "both")
    build_cuda = args.platform in ("cuda", "both")
    cmake_cmd = [
        "cmake",
        "-S",
        str(REPO_ROOT),
        "-B",
        str(BUILD_DIR),
        f"-DCMAKE_BUILD_TYPE={args.build_type}",
        f"-DUSE_CUDA_FALLBACK={'ON' if args.use_cuda_fallback else 'OFF'}",
        f"-DBUILD_HC_BENCHES={'ON' if build_hc else 'OFF'}",
        f"-DBUILD_NVIDIA_BENCHES={'ON' if build_cuda else 'OFF'}",
    ]
    run_cmd(cmake_cmd)


def build():
    run_cmd(["cmake", "--build", str(BUILD_DIR), "-j"])


def run_fp64(platform: str):
    binary = BIN_DIR / ("cuda_fp64_flops" if platform == "cuda" else "fp64_flops")
    for case in FP64_CASES:
        cmd = [
            str(binary),
            "--blocks",
            str(case["blocks"]),
            "--threads",
            str(case["threads"]),
            "--iterations",
            str(case["iterations"]),
            "--repeats",
            str(case["repeats"]),
        ]
        run_cmd(cmd)


def run_bandwidth(platform: str):
    binary = BIN_DIR / ("cuda_memory_bandwidth" if platform == "cuda" else "memory_bandwidth")
    cmd = [str(binary), "--iterations", "50"]
    for size in BANDWIDTH_SIZES:
        cmd.extend(["--size", str(size)])
    run_cmd(cmd)


def run_gemm(platform: str):
    binary = BIN_DIR / ("cublas_gemm" if platform == "cuda" else "hcblas_gemm")
    cmd = [str(binary), "--repeats", "20"]
    for dtype in GEMM_DTYPES:
        cmd.extend(["--dtype", dtype])
    for shape in GEMM_SHAPES:
        cmd.extend(["--shape", shape])
    run_cmd(cmd)


def run_vector_add(platform: str):
    binary = BIN_DIR / ("cuda_vector_add" if platform == "cuda" else "vector_add")
    cmd = [str(binary), "--repeats", "80"]
    for dtype in VECTOR_DTYPES:
        cmd.extend(["--dtype", dtype])
    for size in VECTOR_SIZES:
        cmd.extend(["--size", str(size)])
    run_cmd(cmd)


def run_matrix_add(platform: str):
    binary = BIN_DIR / ("cuda_matrix_add" if platform == "cuda" else "matrix_add")
    cmd = [str(binary), "--repeats", "40"]
    for dtype in MATRIX_DTYPES:
        cmd.extend(["--dtype", dtype])
    for shape in MATRIX_SHAPES:
        cmd.extend(["--shape", shape])
    run_cmd(cmd)


def main():
    parser = argparse.ArgumentParser(description="Build and run MXGPU benchmarks")
    parser.add_argument("--skip-build", action="store_true", help="Assume binaries already built")
    parser.add_argument("--use-cuda-fallback", action="store_true", help="Compile against CUDA toolkit")
    parser.add_argument("--build-type", default="Release")
    parser.add_argument(
        "--platform",
        choices=["hc", "cuda", "both"],
        default="hc",
        help="Benchmark backend to run (hc SDK, CUDA-only, or both)",
    )
    args = parser.parse_args()

    if not args.skip_build or not BIN_DIR.exists():
        configure(args)
        build()

    if args.platform in ("hc", "both"):
        run_fp64("hc")
        run_bandwidth("hc")
        run_gemm("hc")
        run_vector_add("hc")
        run_matrix_add("hc")
    if args.platform in ("cuda", "both"):
        run_fp64("cuda")
        run_bandwidth("cuda")
        run_gemm("cuda")
        run_vector_add("cuda")
        run_matrix_add("cuda")


if __name__ == "__main__":
    main()
