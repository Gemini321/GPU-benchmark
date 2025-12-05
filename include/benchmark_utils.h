#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "hc_compat.h"

#define HC_CHECK(cmd)                                                                       \
    do {                                                                                    \
        hcError_t _status = cmd;                                                            \
        if (_status != HC_SUCCESS) {                                                        \
            fprintf(stderr, "HC error %d at %s:%d -> %s\n", static_cast<int>(_status),      \
                    __FILE__, __LINE__, hcGetErrorString(_status));                         \
            throw std::runtime_error("hc runtime call failed");                             \
        }                                                                                   \
    } while (0)

#define HCBLAS_CHECK(cmd)                                                                   \
    do {                                                                                    \
        hcblasStatus_t _status = cmd;                                                       \
        if (_status != HCBLAS_STATUS_SUCCESS) {                                             \
            fprintf(stderr, "hcBLAS error %d at %s:%d\n", static_cast<int>(_status),        \
                    __FILE__, __LINE__);                                                    \
            throw std::runtime_error("hcBLAS call failed");                                 \
        }                                                                                   \
    } while (0)

inline float elapsed_ms(std::function<void()> fn, hcStream_t stream = nullptr) {
    hcEvent_t start, stop;
    HC_CHECK(hcEventCreate(&start));
    HC_CHECK(hcEventCreate(&stop));

    if (stream) {
        HC_CHECK(hcEventRecord(start, stream));
        fn();
        HC_CHECK(hcEventRecord(stop, stream));
    } else {
        HC_CHECK(hcEventRecord(start, nullptr));
        fn();
        HC_CHECK(hcEventRecord(stop, nullptr));
    }
    HC_CHECK(hcEventSynchronize(stop));
    float ms = 0.0f;
    HC_CHECK(hcEventElapsedTime(&ms, start, stop));
    HC_CHECK(hcEventDestroy(start));
    HC_CHECK(hcEventDestroy(stop));
    return ms;
}

inline void warmup_device() {
    HC_CHECK(hcDeviceSynchronize());
}

struct SizeSweepConfig {
    std::vector<size_t> sizes;
    int repeats{10};
};

inline SizeSweepConfig default_size_sweep() {
    return SizeSweepConfig{std::vector<size_t>{1 << 20, 1 << 22, 1 << 24, 1 << 26}, 10};
}

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
