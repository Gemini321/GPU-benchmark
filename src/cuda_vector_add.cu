#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "benchmark_utils_cuda.h"

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
    size_t blocks = 0;
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
            printf(
                "Usage: cuda_vector_add [--size N] [--dtype float|double] [--threads N] [--blocks N] [--repeats N]\n");
            std::exit(0);
        }
    }
    if (cfg.sizes.empty()) {
        cfg.sizes = {1ull << 20, 1ull << 22, 1ull << 24, 1ull << 26};
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

namespace {
bool g_vector_kernel_supported = true;
bool g_vector_kernel_warned = false;
}

template <typename T>
struct GeamCaller;

template <>
struct GeamCaller<float> {
    static void geam(cublasHandle_t handle, int m, int n, const float *alpha, const float *A, int lda, const float *beta,
                     const float *B, int ldb, float *C, int ldc) {
        CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
    }
};

template <>
struct GeamCaller<double> {
    static void geam(cublasHandle_t handle, int m, int n, const double *alpha, const double *A, int lda,
                     const double *beta, const double *B, int ldb, double *C, int ldc) {
        CUBLAS_CHECK(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
    }
};

template <typename T>
float fallback_vector_add(cublasHandle_t handle, const T *d_a, const T *d_b, T *d_c, size_t elements, int repeats) {
    const T alpha = static_cast<T>(1.0);
    const T beta = static_cast<T>(1.0);
    const int m = static_cast<int>(elements);
    const int n = 1;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < repeats; ++i) {
        GeamCaller<T>::geam(handle, m, n, &alpha, d_a, m, &beta, d_b, m, d_c, m);
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

template <typename T>
bool try_launch_kernel(const T *d_a, const T *d_b, T *d_c, size_t elements, unsigned int grid, unsigned int threads) {
    if (!g_vector_kernel_supported) {
        return false;
    }
    vector_add_kernel<<<grid, threads>>>(d_a, d_b, d_c, elements);
    cudaError_t err = cudaGetLastError();
    if (err == cudaErrorUnsupportedPtxVersion) {
        g_vector_kernel_supported = false;
        if (!g_vector_kernel_warned) {
            fprintf(stderr,
                    "Warning: CUDA vector_add kernel disabled (unsupported PTX version). Falling back to cuBLAS geam.\n");
            g_vector_kernel_warned = true;
        }
        cudaGetLastError(); // clear sticky error
        return false;
    }
    CUDA_CHECK(err);
    return true;
}

template <typename T>
void run_dtype(const VectorAddConfig &cfg, cublasHandle_t handle) {
    for (size_t elements : cfg.sizes) {
        const size_t bytes = elements * sizeof(T);
        T *d_a = nullptr;
        T *d_b = nullptr;
        T *d_c = nullptr;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));

        std::vector<T> h_a(elements);
        std::vector<T> h_b(elements);
        init_host(h_a);
        init_host(h_b);
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_c, 0, bytes));

        const unsigned int grid = static_cast<unsigned int>(compute_grid(elements, cfg));
        const unsigned int threads = static_cast<unsigned int>(cfg.threads);
        bool kernel_mode = try_launch_kernel(d_a, d_b, d_c, elements, grid, threads);
        if (kernel_mode) {
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        float total_ms = 0.0f;
        if (kernel_mode) {
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            CUDA_CHECK(cudaEventRecord(start, 0));
            for (int i = 0; i < cfg.repeats; ++i) {
                vector_add_kernel<<<grid, threads>>>(d_a, d_b, d_c, elements);
            }
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        } else {
            total_ms = fallback_vector_add(handle, d_a, d_b, d_c, elements, cfg.repeats);
        }

        const double seconds = static_cast<double>(total_ms) / 1e3;
        const double ops = static_cast<double>(elements) * static_cast<double>(cfg.repeats);
        const double gflops = (ops / seconds) / 1e9;

        char size_buf[32];
        char time_buf[32];
        char flops_buf[32];
        std::snprintf(size_buf, sizeof(size_buf), "%zu", elements);
        std::snprintf(time_buf, sizeof(time_buf), "%.3f", total_ms / cfg.repeats);
        std::snprintf(flops_buf, sizeof(flops_buf), "%.1f", gflops);
        print_table_row({size_buf, time_buf, flops_buf});

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }
}

int main(int argc, char **argv) {
    VectorAddConfig cfg = parse_args(argc, argv);
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    for (const auto dtype : cfg.dtypes) {
        if (dtype == VectorDType::Float32) {
            printf("CUDA vector add FP32 (threads=%zu)\n", cfg.threads);
            print_table_header({"Elements", "Time (ms)", "GFLOP/s"});
            run_dtype<float>(cfg, handle);
        } else {
            printf("CUDA vector add FP64 (threads=%zu)\n", cfg.threads);
            print_table_header({"Elements", "Time (ms)", "GFLOP/s"});
            run_dtype<double>(cfg, handle);
        }
    }
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
