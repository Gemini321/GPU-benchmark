#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(cmd)                                                                    \
    do {                                                                                   \
        cudaError_t _status = cmd;                                                         \
        if (_status != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %d at %s:%d -> %s\n", static_cast<int>(_status),     \
                    __FILE__, __LINE__, cudaGetErrorString(_status));                      \
            throw std::runtime_error("CUDA runtime call failed");                         \
        }                                                                                  \
    } while (0)

inline float cuda_elapsed_ms(std::function<void()> fn, cudaStream_t stream = nullptr) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    if (stream) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        fn();
        CUDA_CHECK(cudaEventRecord(stop, stream));
    } else {
        CUDA_CHECK(cudaEventRecord(start, nullptr));
        fn();
        CUDA_CHECK(cudaEventRecord(stop, nullptr));
    }
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

inline void cuda_warmup_device() { CUDA_CHECK(cudaDeviceSynchronize()); }

inline void print_table_header(const std::vector<std::string> &cols) {
    for (const auto &c : cols) {
        printf("%20s", c.c_str());
    }
    printf("\n");
}

inline void print_table_row(const std::vector<std::string> &cols) {
    for (const auto &c : cols) {
        printf("%20s", c.c_str());
    }
    printf("\n");
}

inline const char *cublas_get_error_string(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "alloc failed";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "arch mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "internal error";
    default:
        return "unknown";
    }
}

#define CUBLAS_CHECK(cmd)                                                                 \
    do {                                                                                  \
        cublasStatus_t _status = cmd;                                                     \
        if (_status != CUBLAS_STATUS_SUCCESS) {                                           \
            fprintf(stderr, "cuBLAS error %d at %s:%d -> %s\n", static_cast<int>(_status),\
                    __FILE__, __LINE__, cublas_get_error_string(_status));                \
            throw std::runtime_error("cuBLAS call failed");                               \
        }                                                                                 \
    } while (0)
