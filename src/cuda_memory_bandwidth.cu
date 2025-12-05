#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "benchmark_utils_cuda.h"

struct alignas(16) VecFloat4 {
    float x;
    float y;
    float z;
    float w;
};

__global__ void cuda_device_copy_kernel(const VecFloat4 *__restrict__ src, VecFloat4 *__restrict__ dst, size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

struct CudaBandwidthConfig {
    size_t min_bytes = 8ull << 20;
    size_t max_bytes = 512ull << 20;
    double scale = 2.0;
    int iterations = 20;
    std::vector<size_t> explicit_sizes;
};

CudaBandwidthConfig parse_args(int argc, char **argv) {
    CudaBandwidthConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--min-bytes") == 0 && i + 1 < argc) {
            cfg.min_bytes = std::strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--max-bytes") == 0 && i + 1 < argc) {
            cfg.max_bytes = std::strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--scale") == 0 && i + 1 < argc) {
            cfg.scale = std::atof(argv[++i]);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            cfg.iterations = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            cfg.explicit_sizes.push_back(std::strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: cuda_memory_bandwidth [--min-bytes N --max-bytes N --scale F --iterations N --size N]\n");
            std::exit(0);
        }
    }
    return cfg;
}

std::vector<size_t> build_sizes(const CudaBandwidthConfig &cfg) {
    if (!cfg.explicit_sizes.empty()) {
        return cfg.explicit_sizes;
    }
    std::vector<size_t> sizes;
    double current = static_cast<double>(cfg.min_bytes);
    while (current <= static_cast<double>(cfg.max_bytes)) {
        sizes.push_back(static_cast<size_t>(current));
        current *= cfg.scale;
    }
    return sizes;
}

double benchmark_direction(void *dst, const void *src, size_t bytes, int iterations, enum cudaMemcpyKind kind) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const double seconds = static_cast<double>(ms) / 1e3;
    const double total_bytes = static_cast<double>(bytes) * static_cast<double>(iterations);
    return (total_bytes / seconds) / 1e9;
}

double benchmark_device_copy(uint8_t *dst, const uint8_t *src, size_t bytes, int iterations, double d2d_hint = -1.0) {
    const size_t vec_bytes = bytes - (bytes % sizeof(VecFloat4));
    const size_t vec_elems = vec_bytes / sizeof(VecFloat4);
    if (vec_elems == 0) {
        return 0.0;
    }
    static bool kernel_supported = true;
    static bool warned = false;
    if (!kernel_supported) {
        if (d2d_hint >= 0.0) {
            return d2d_hint * 2.0;
        }
        return benchmark_direction(dst, src, bytes, iterations, cudaMemcpyDeviceToDevice) * 2.0;
    }
    const VecFloat4 *src_vec = reinterpret_cast<const VecFloat4 *>(src);
    VecFloat4 *dst_vec = reinterpret_cast<VecFloat4 *>(dst);
    const unsigned int threads = 256;
    size_t raw_blocks = (vec_elems + threads - 1) / threads;
    if (raw_blocks == 0) {
        raw_blocks = 1;
    }
    const unsigned int blocks = static_cast<unsigned int>(std::min(raw_blocks, static_cast<size_t>(65535)));

    cuda_device_copy_kernel<<<blocks, threads>>>(src_vec, dst_vec, vec_elems);
    cudaError_t launch_status = cudaGetLastError();
    if (launch_status == cudaErrorUnsupportedPtxVersion) {
        kernel_supported = false;
        if (!warned) {
            fprintf(stderr,
                    "Warning: skipping RW Copy kernel in cuda_memory_bandwidth (unsupported PTX version on this driver). "
                    "Upgrade the driver or rebuild with a matching toolkit to enable this test.\n");
            warned = true;
        }
        cudaGetLastError(); // clear sticky error
        if (d2d_hint >= 0.0) {
            return d2d_hint * 2.0;
        }
        return benchmark_direction(dst, src, bytes, iterations, cudaMemcpyDeviceToDevice) * 2.0;
    }
    CUDA_CHECK(launch_status);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < iterations; ++i) {
        cuda_device_copy_kernel<<<blocks, threads>>>(src_vec, dst_vec, vec_elems);
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const double seconds = static_cast<double>(ms) / 1e3;
    const double processed_bytes = static_cast<double>(vec_bytes) * 2.0 * static_cast<double>(iterations);
    return (processed_bytes / seconds) / 1e9;
}

int main(int argc, char **argv) {
    CudaBandwidthConfig cfg = parse_args(argc, argv);
    std::vector<size_t> sizes = build_sizes(cfg);
    if (sizes.empty()) {
        fprintf(stderr, "No sizes to benchmark\n");
        return 1;
    }

    const size_t max_size = *std::max_element(sizes.begin(), sizes.end());

    void *h_src = nullptr;
    void *h_dst = nullptr;
    uint8_t *d_src = nullptr;
    uint8_t *d_dst = nullptr;

    CUDA_CHECK(cudaHostAlloc(&h_src, max_size, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_dst, max_size, cudaHostAllocDefault));
    std::memset(h_src, 0xAB, max_size);
    std::memset(h_dst, 0, max_size);

    CUDA_CHECK(cudaMalloc(&d_src, max_size));
    CUDA_CHECK(cudaMalloc(&d_dst, max_size));

    printf("CUDA memory bandwidth benchmark\n");
    print_table_header({"Size(MB)", "H2D GB/s", "D2H GB/s", "D2D GB/s", "RW Copy GB/s"});

    for (size_t bytes : sizes) {
        const double mb = static_cast<double>(bytes) / (1 << 20);
        const double h2d = benchmark_direction(d_src, h_src, bytes, cfg.iterations, cudaMemcpyHostToDevice);
        const double d2h = benchmark_direction(h_dst, d_src, bytes, cfg.iterations, cudaMemcpyDeviceToHost);
        const double d2d = benchmark_direction(d_dst, d_src, bytes, cfg.iterations, cudaMemcpyDeviceToDevice);
        const double rw = benchmark_device_copy(d_dst, d_src, bytes, cfg.iterations, d2d);

        char size_buf[32];
        char h2d_buf[32];
        char d2h_buf[32];
        char d2d_buf[32];
        char rw_buf[32];
        std::snprintf(size_buf, sizeof(size_buf), "%.1f", mb);
        std::snprintf(h2d_buf, sizeof(h2d_buf), "%.1f", h2d);
        std::snprintf(d2h_buf, sizeof(d2h_buf), "%.1f", d2h);
        std::snprintf(d2d_buf, sizeof(d2d_buf), "%.1f", d2d);
        std::snprintf(rw_buf, sizeof(rw_buf), "%.1f", rw);
        print_table_row({size_buf, h2d_buf, d2h_buf, d2d_buf, rw_buf});
    }

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFreeHost(h_src));
    CUDA_CHECK(cudaFreeHost(h_dst));
    return 0;
}
