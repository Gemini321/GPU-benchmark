#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "benchmark_utils.h"

struct GemmShape {
    size_t m;
    size_t n;
    size_t k;
};

enum class GemmDType { Float32, Float64, Float16 };

struct GemmConfig {
    std::vector<GemmShape> shapes;
    std::vector<GemmDType> dtypes;
    int repeats = 10;
};

GemmShape parse_shape(const std::string &token) {
    GemmShape shape{0, 0, 0};
    size_t first = token.find_first_of("xX");
    size_t second = token.find_last_of("xX");
    if (first == std::string::npos) {
        size_t n = std::strtoull(token.c_str(), nullptr, 10);
        shape = GemmShape{n, n, n};
    } else {
        std::string m_str = token.substr(0, first);
        std::string n_str = token.substr(first + 1, second - first - 1);
        std::string k_str = token.substr(second + 1);
        shape.m = std::strtoull(m_str.c_str(), nullptr, 10);
        shape.n = std::strtoull(n_str.c_str(), nullptr, 10);
        shape.k = std::strtoull(k_str.c_str(), nullptr, 10);
    }
    return shape;
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

GemmDType parse_dtype(const std::string &token) {
    const std::string lowered = to_lower(token);
    if (lowered == "float" || lowered == "fp32" || lowered == "single") {
        return GemmDType::Float32;
    }
    if (lowered == "double" || lowered == "fp64") {
        return GemmDType::Float64;
    }
    if (lowered == "float16" || lowered == "fp16" || lowered == "half") {
        return GemmDType::Float16;
    }
    fprintf(stderr, "Unknown dtype '%s' (expected float/fp32, double/fp64, or float16/fp16)\n", token.c_str());
    std::exit(1);
}

GemmConfig parse_args(int argc, char **argv) {
    GemmConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--shape") == 0 && i + 1 < argc) {
            cfg.shapes.push_back(parse_shape(argv[++i]));
        } else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            cfg.repeats = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) {
            cfg.dtypes.push_back(parse_dtype(argv[++i]));
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: hcblas_gemm [--shape MxNxK] [--dtype float|double|float16] [--repeats N]\n");
            std::exit(0);
        }
    }
    if (cfg.shapes.empty()) {
        cfg.shapes = {GemmShape{1024, 1024, 1024},   GemmShape{2048, 2048, 2048},   GemmShape{3072, 3072, 3072},
                      GemmShape{4096, 4096, 4096},   GemmShape{6144, 6144, 6144},   GemmShape{8192, 8192, 8192},
                      GemmShape{12288, 12288, 12288}, GemmShape{16384, 16384, 16384}};
    }
    if (cfg.dtypes.empty()) {
        cfg.dtypes = {GemmDType::Float64, GemmDType::Float32, GemmDType::Float16};
    }
    return cfg;
}

template <typename T>
void init_host(std::vector<T> &buf) {
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<T>((static_cast<double>(i % 1024) - 512.0) / 1024.0);
    }
}

template <>
void init_host<hcHalf_t>(std::vector<hcHalf_t> &buf) {
    for (size_t i = 0; i < buf.size(); ++i) {
        float val = static_cast<float>((static_cast<double>(i % 1024) - 512.0) / 1024.0);
#ifdef USE_CUDA_FALLBACK
        buf[i] = __float2half(val);
#else
        buf[i] = static_cast<hcHalf_t>(val);
#endif
    }
}

template <typename T>
struct GemmCaller;

template <>
struct GemmCaller<float> {
    using ScalarType = float;
    static constexpr const char *label = "FP32";
    static void gemm(hcblasHandle_t handle, int m, int n, int k, const float *alpha, const float *A, int lda,
                     const float *B, int ldb, const float *beta, float *C, int ldc) {
        HCBLAS_CHECK(hcblasSgemm(handle, HCBLAS_OP_N, HCBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
};

template <>
struct GemmCaller<double> {
    using ScalarType = double;
    static constexpr const char *label = "FP64";
    static void gemm(hcblasHandle_t handle, int m, int n, int k, const double *alpha, const double *A, int lda,
                     const double *B, int ldb, const double *beta, double *C, int ldc) {
        HCBLAS_CHECK(hcblasDgemm(handle, HCBLAS_OP_N, HCBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    }
};

template <>
struct GemmCaller<hcHalf_t> {
    using ScalarType = float;
    static constexpr const char *label = "FP16";
    static void gemm(hcblasHandle_t handle, int m, int n, int k, const float *alpha, const hcHalf_t *A, int lda,
                     const hcHalf_t *B, int ldb, const float *beta, hcHalf_t *C, int ldc) {
        HCBLAS_CHECK(hcblasGemmEx(handle, HCBLAS_OP_N, HCBLAS_OP_N, m, n, k, alpha, A, HPCC_R_16F, lda, B, HPCC_R_16F,
                                  ldb, beta, C, HPCC_R_16F, ldc, HCBLAS_COMPUTE_32F, HCBLAS_GEMM_DEFAULT));
    }
};

template <typename T>
void run_dtype(const GemmConfig &cfg, hcblasHandle_t handle, hcStream_t stream) {
    using Scalar = typename GemmCaller<T>::ScalarType;
    for (const auto &shape : cfg.shapes) {
        const size_t a_elems = shape.m * shape.k;
        const size_t b_elems = shape.k * shape.n;
        const size_t c_elems = shape.m * shape.n;

        T *d_A = nullptr;
        T *d_B = nullptr;
        T *d_C = nullptr;
        HC_CHECK(hcMalloc(reinterpret_cast<void **>(&d_A), a_elems * sizeof(T)));
        HC_CHECK(hcMalloc(reinterpret_cast<void **>(&d_B), b_elems * sizeof(T)));
        HC_CHECK(hcMalloc(reinterpret_cast<void **>(&d_C), c_elems * sizeof(T)));

        std::vector<T> h_A(a_elems);
        std::vector<T> h_B(b_elems);
        init_host(h_A);
        init_host(h_B);
        HC_CHECK(hcMemcpy(d_A, h_A.data(), h_A.size() * sizeof(T), hcMemcpyHostToDevice));
        HC_CHECK(hcMemcpy(d_B, h_B.data(), h_B.size() * sizeof(T), hcMemcpyHostToDevice));
        HC_CHECK(hcMemset(d_C, 0, c_elems * sizeof(T)));

        const Scalar alpha = static_cast<Scalar>(1.0f);
        const Scalar beta = static_cast<Scalar>(0.0f);
        // Warm-up
        GemmCaller<T>::gemm(handle, static_cast<int>(shape.m), static_cast<int>(shape.n), static_cast<int>(shape.k),
                            &alpha, d_A, static_cast<int>(shape.m), d_B, static_cast<int>(shape.k), &beta, d_C,
                            static_cast<int>(shape.m));
        HC_CHECK(hcStreamSynchronize(stream));

        hcEvent_t start, stop;
        HC_CHECK(hcEventCreate(&start));
        HC_CHECK(hcEventCreate(&stop));
        HC_CHECK(hcEventRecord(start, stream));
        for (int i = 0; i < cfg.repeats; ++i) {
            GemmCaller<T>::gemm(handle, static_cast<int>(shape.m), static_cast<int>(shape.n), static_cast<int>(shape.k),
                                &alpha, d_A, static_cast<int>(shape.m), d_B, static_cast<int>(shape.k), &beta, d_C,
                                static_cast<int>(shape.m));
        }
        HC_CHECK(hcEventRecord(stop, stream));
        HC_CHECK(hcEventSynchronize(stop));

        float ms = 0.0f;
        HC_CHECK(hcEventElapsedTime(&ms, start, stop));
        HC_CHECK(hcEventDestroy(start));
        HC_CHECK(hcEventDestroy(stop));

        const double seconds = ms / 1e3;
        const double ops = 2.0 * static_cast<double>(shape.m) * static_cast<double>(shape.n) *
                           static_cast<double>(shape.k) * static_cast<double>(cfg.repeats);
        const double gflops = (ops / seconds) / 1e9;

        char shape_buf[64];
        std::snprintf(shape_buf, sizeof(shape_buf), "%zux%zux%zu", shape.m, shape.n, shape.k);
        char time_buf[32];
        char flops_buf[32];
        std::snprintf(time_buf, sizeof(time_buf), "%.2f", ms / cfg.repeats);
        std::snprintf(flops_buf, sizeof(flops_buf), "%.1f", gflops);
        print_table_row({shape_buf, time_buf, flops_buf});

        HC_CHECK(hcFree(d_A));
        HC_CHECK(hcFree(d_B));
        HC_CHECK(hcFree(d_C));
    }
}

int main(int argc, char **argv) {
    GemmConfig cfg = parse_args(argc, argv);

    hcblasHandle_t handle;
    HCBLAS_CHECK(hcblasCreate(&handle));
    hcStream_t stream;
    HC_CHECK(hcStreamCreate(&stream));
    HCBLAS_CHECK(hcblasSetStream(handle, stream));

    for (const auto dtype : cfg.dtypes) {
        if (dtype == GemmDType::Float32) {
            printf("hcBLAS FP32 GEMM results\n");
            print_table_header({"MxNxK", "Time (ms)", "GFLOP/s"});
            run_dtype<float>(cfg, handle, stream);
        } else if (dtype == GemmDType::Float64) {
            printf("hcBLAS FP64 GEMM results\n");
            print_table_header({"MxNxK", "Time (ms)", "GFLOP/s"});
            run_dtype<double>(cfg, handle, stream);
        } else {
            printf("hcBLAS FP16 GEMM results\n");
            print_table_header({"MxNxK", "Time (ms)", "GFLOP/s"});
            run_dtype<hcHalf_t>(cfg, handle, stream);
        }
    }

    HC_CHECK(hcStreamDestroy(stream));
    HCBLAS_CHECK(hcblasDestroy(handle));
    return 0;
}
