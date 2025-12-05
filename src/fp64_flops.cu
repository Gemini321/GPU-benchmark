#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "benchmark_utils.h"

__global__ void fp64_fma_kernel(double *out, size_t iterations) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double x = 1.0 + static_cast<double>(idx & 0xFF);
    double y = 2.0 + static_cast<double>((idx >> 8) & 0xFF);
    double acc0 = x;
    double acc1 = y;
    for (size_t i = 0; i < iterations; ++i) {
        acc0 = acc0 * acc1 + x;
        acc1 = acc1 * acc0 + y;
    }
    out[idx] = acc0 + acc1;
}

struct Fp64Config {
    size_t blocks = 512;
    size_t threads = 256;
    size_t iterations = 1 << 12;
    int repeats = 20;
};

Fp64Config parse_args(int argc, char **argv) {
    Fp64Config cfg;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
            cfg.blocks = std::strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg.threads = std::strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            cfg.iterations = std::strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            cfg.repeats = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: fp64_flops [--blocks N] [--threads N] [--iterations N] [--repeats N]\n");
            std::exit(0);
        }
    }
    return cfg;
}

int main(int argc, char **argv) {
    Fp64Config cfg = parse_args(argc, argv);
    const size_t thread_count = cfg.blocks * cfg.threads;

    double *d_out = nullptr;
    HC_CHECK(hcMalloc(reinterpret_cast<void **>(&d_out), thread_count * sizeof(double)));

    // Warm-up launch
    fp64_fma_kernel<<<cfg.blocks, cfg.threads>>>(d_out, cfg.iterations);
    HC_CHECK(hcPeekAtLastError());
    HC_CHECK(hcDeviceSynchronize());

    const auto host_start = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.repeats; ++i) {
        fp64_fma_kernel<<<cfg.blocks, cfg.threads>>>(d_out, cfg.iterations);
        HC_CHECK(hcPeekAtLastError());
    }
    HC_CHECK(hcDeviceSynchronize());
    const auto host_end = std::chrono::steady_clock::now();
    const double total_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(host_end - host_start).count();

    const double ops_per_launch = static_cast<double>(cfg.iterations) * static_cast<double>(thread_count) * 2.0;
    const double total_ops = ops_per_launch * static_cast<double>(cfg.repeats);
    const double seconds = total_ms / 1e3;
    const double gflops = (total_ops / seconds) / 1e9;

    printf("FP64 FLOPs benchmark\n");
    printf("Blocks: %zu Threads/block: %zu Iterations/thread: %zu\n", cfg.blocks, cfg.threads, cfg.iterations);
    printf("Avg kernel time: %.3f ms\n", total_ms / cfg.repeats);
    printf("Achieved FP64 throughput: %.2f GFLOP/s\n", gflops);

    HC_CHECK(hcFree(d_out));
    return 0;
}
