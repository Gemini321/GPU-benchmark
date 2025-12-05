#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "benchmark_utils.h"

template <typename T>
__global__ void matrix_add_kernel(const T *a, const T *b, T *c, size_t elements) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < elements; i += stride) {
        c[i] = a[i] + b[i];
    }
}

struct MatrixShape {
    size_t m;
    size_t n;
};

enum class MatrixDType { Float32, Float64 };

struct MatrixAddConfig {
    std::vector<MatrixShape> shapes;
    std::vector<MatrixDType> dtypes;
    size_t threads = 256;
    size_t blocks = 0;
    int repeats = 30;
};

MatrixShape parse_shape(const std::string &token) {
    size_t pos = token.find_first_of("xX");
    if (pos == std::string::npos) {
        size_t dim = std::strtoull(token.c_str(), nullptr, 10);
        return MatrixShape{dim, dim};
    }
    size_t m = std::strtoull(token.substr(0, pos).c_str(), nullptr, 10);
    size_t n = std::strtoull(token.substr(pos + 1).c_str(), nullptr, 10);
    return MatrixShape{m, n};
}

MatrixAddConfig parse_args(int argc, char **argv) {
    MatrixAddConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--shape") == 0 && i + 1 < argc) {
            cfg.shapes.push_back(parse_shape(argv[++i]));
        } else if (strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) {
            const std::string token = argv[++i];
            if (token == "float" || token == "fp32" || token == "single") {
                cfg.dtypes.push_back(MatrixDType::Float32);
            } else if (token == "double" || token == "fp64") {
                cfg.dtypes.push_back(MatrixDType::Float64);
            } else {
                fprintf(stderr, "Unknown dtype '%s'\n", token.c_str());
                std::exit(1);
            }
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg.threads = std::strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
            cfg.blocks = std::strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            cfg.repeats = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: matrix_add [--shape MxN] [--dtype float|double] [--threads N] [--blocks N] [--repeats N]\n");
            std::exit(0);
        }
    }
    if (cfg.shapes.empty()) {
        cfg.shapes = {MatrixShape{2048, 2048}, MatrixShape{4096, 4096}, MatrixShape{4096, 8192},
                      MatrixShape{8192, 8192}};
    }
    if (cfg.dtypes.empty()) {
        cfg.dtypes = {MatrixDType::Float32, MatrixDType::Float64};
    }
    if (cfg.threads == 0) {
        cfg.threads = 256;
    }
    return cfg;
}

size_t compute_grid(size_t elements, const MatrixAddConfig &cfg) {
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
void init_host(std::vector<T> &buf) {
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<T>((static_cast<double>(i % 2048) - 1024.0) / 2048.0);
    }
}

template <typename T>
void run_dtype(const MatrixAddConfig &cfg) {
    for (const auto &shape : cfg.shapes) {
        const size_t elements = shape.m * shape.n;
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
        matrix_add_kernel<<<grid, threads>>>(d_a, d_b, d_c, elements);
        HC_CHECK(hcPeekAtLastError());
        HC_CHECK(hcDeviceSynchronize());

        hcEvent_t start, stop;
        HC_CHECK(hcEventCreate(&start));
        HC_CHECK(hcEventCreate(&stop));
        HC_CHECK(hcEventRecord(start, 0));
        for (int i = 0; i < cfg.repeats; ++i) {
            matrix_add_kernel<<<grid, threads>>>(d_a, d_b, d_c, elements);
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

        char shape_buf[64];
        char elems_buf[32];
        char time_buf[32];
        char flops_buf[32];
        std::snprintf(shape_buf, sizeof(shape_buf), "%zux%zu", shape.m, shape.n);
        std::snprintf(elems_buf, sizeof(elems_buf), "%zu", elements);
        std::snprintf(time_buf, sizeof(time_buf), "%.3f", ms / cfg.repeats);
        std::snprintf(flops_buf, sizeof(flops_buf), "%.1f", gflops);
        print_table_row({shape_buf, elems_buf, time_buf, flops_buf});

        HC_CHECK(hcFree(d_a));
        HC_CHECK(hcFree(d_b));
        HC_CHECK(hcFree(d_c));
    }
}

int main(int argc, char **argv) {
    MatrixAddConfig cfg = parse_args(argc, argv);

    for (const auto dtype : cfg.dtypes) {
        if (dtype == MatrixDType::Float32) {
            printf("Matrix add FP32 (threads=%zu)\n", cfg.threads);
            print_table_header({"MxN", "Elements", "Time (ms)", "GFLOP/s"});
            run_dtype<float>(cfg);
        } else {
            printf("Matrix add FP64 (threads=%zu)\n", cfg.threads);
            print_table_header({"MxN", "Elements", "Time (ms)", "GFLOP/s"});
            run_dtype<double>(cfg);
        }
    }
    return 0;
}
