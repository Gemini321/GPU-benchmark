#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "benchmark_utils.h"

template <typename T>
__global__ void vector_add_kernel(const T *a, const T *b, T *c, size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

enum class VectorDType { Float32, Float64 };

struct VectorAddConfig {
    std::vector<size_t> sizes;
    std::vector<VectorDType> dtypes;
    size_t threads = 256;
    size_t blocks = 0; // auto
    int repeats = 50;
};

VectorDType parse_dtype(const std::string &token) {
    if (token == "float" || token == "fp32" || token == "single") {
        return VectorDType::Float32;
    }
    if (token == "double" || token == "fp64") {
        return VectorDType::Float64;
    }
    fprintf(stderr, "Unknown dtype '%s'. Use float or double.\n", token.c_str());
    std::exit(1);
}

VectorAddConfig parse_args(int argc, char **argv) {
    VectorAddConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            cfg.sizes.push_back(std::strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) {
            cfg.dtypes.push_back(parse_dtype(argv[++i]));
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg.threads = std::strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
            cfg.blocks = std::strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            cfg.repeats = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: vector_add [--size N] [--dtype float|double] [--threads N] [--blocks N] [--repeats N]\n");
            std::exit(0);
        }
    }
    if (cfg.sizes.empty()) {
        cfg.sizes = default_size_sweep().sizes;
    }
    if (cfg.dtypes.empty()) {
        cfg.dtypes = {VectorDType::Float32, VectorDType::Float64};
    }
    if (cfg.threads == 0) {
        cfg.threads = 256;
    }
    return cfg;
}

template <typename T>
void init_host(std::vector<T> &buf) {
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<T>((static_cast<double>(i % 1024) - 512.0) / 1024.0);
    }
}

size_t compute_grid(size_t elements, const VectorAddConfig &cfg) {
    if (cfg.blocks > 0) {
        return cfg.blocks;
    }
    size_t grid = (elements + cfg.threads - 1) / cfg.threads;
    if (grid == 0) {
        grid = 1;
    }
    const size_t max_grid = 65535;
    return std::min(grid, max_grid);
}

template <typename T>
const char *dtype_name();

template <>
const char *dtype_name<float>() {
    return "FP32";
}

template <>
const char *dtype_name<double>() {
    return "FP64";
}

template <typename T>
void run_dtype(const VectorAddConfig &cfg) {
    for (size_t elements : cfg.sizes) {
        const size_t bytes = elements * sizeof(T);
        T *d_a = nullptr;
        T *d_b = nullptr;
        T *d_c = nullptr;
        HC_CHECK(hcMalloc(reinterpret_cast<void **>(&d_a), bytes));
        HC_CHECK(hcMalloc(reinterpret_cast<void **>(&d_b), bytes));
        HC_CHECK(hcMalloc(reinterpret_cast<void **>(&d_c), bytes));

        std::vector<T> h_a(elements);
        std::vector<T> h_b(elements);
        init_host(h_a);
        init_host(h_b);
        HC_CHECK(hcMemcpy(d_a, h_a.data(), bytes, hcMemcpyHostToDevice));
        HC_CHECK(hcMemcpy(d_b, h_b.data(), bytes, hcMemcpyHostToDevice));
        HC_CHECK(hcMemset(d_c, 0, bytes));

        const unsigned int grid = static_cast<unsigned int>(compute_grid(elements, cfg));
        const unsigned int threads = static_cast<unsigned int>(cfg.threads);
        vector_add_kernel<<<grid, threads>>>(d_a, d_b, d_c, elements);
        HC_CHECK(hcPeekAtLastError());
        HC_CHECK(hcDeviceSynchronize());

        hcEvent_t start, stop;
        HC_CHECK(hcEventCreate(&start));
        HC_CHECK(hcEventCreate(&stop));
        HC_CHECK(hcEventRecord(start, 0));
        for (int i = 0; i < cfg.repeats; ++i) {
            vector_add_kernel<<<grid, threads>>>(d_a, d_b, d_c, elements);
            HC_CHECK(hcPeekAtLastError());
        }
        HC_CHECK(hcEventRecord(stop, 0));
        HC_CHECK(hcEventSynchronize(stop));

        float ms = 0.0f;
        HC_CHECK(hcEventElapsedTime(&ms, start, stop));
        HC_CHECK(hcEventDestroy(start));
        HC_CHECK(hcEventDestroy(stop));

        const double seconds = static_cast<double>(ms) / 1e3;
        const double ops = static_cast<double>(elements) * static_cast<double>(cfg.repeats);
        const double gflops = (ops / seconds) / 1e9;

        char size_buf[32];
        char time_buf[32];
        char flops_buf[32];
        std::snprintf(size_buf, sizeof(size_buf), "%zu", elements);
        std::snprintf(time_buf, sizeof(time_buf), "%.3f", ms / cfg.repeats);
        std::snprintf(flops_buf, sizeof(flops_buf), "%.1f", gflops);
        print_table_row({size_buf, time_buf, flops_buf});

        HC_CHECK(hcFree(d_a));
        HC_CHECK(hcFree(d_b));
        HC_CHECK(hcFree(d_c));
    }
}

int main(int argc, char **argv) {
    VectorAddConfig cfg = parse_args(argc, argv);

    for (const auto dtype : cfg.dtypes) {
        if (dtype == VectorDType::Float32) {
            printf("Vector add FP32 (threads=%zu)\n", cfg.threads);
            print_table_header({"Elements", "Time (ms)", "GFLOP/s"});
            run_dtype<float>(cfg);
        } else {
            printf("Vector add FP64 (threads=%zu)\n", cfg.threads);
            print_table_header({"Elements", "Time (ms)", "GFLOP/s"});
            run_dtype<double>(cfg);
        }
    }
    return 0;
}
