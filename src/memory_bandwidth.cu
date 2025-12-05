#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "benchmark_utils.h"

struct BandwidthConfig {
    size_t min_bytes = 1ull << 20;   // 1 MB
    size_t max_bytes = 1536ull << 20; // 1.5 GB
    double scale = 2.0;
    int iterations = 20;
    std::vector<size_t> explicit_sizes;
};

BandwidthConfig parse_args(int argc, char **argv) {
    BandwidthConfig cfg;
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
            printf("Usage: memory_bandwidth [--min-bytes N --max-bytes N --scale F --iterations N --size N]\n");
            std::exit(0);
        }
    }
    return cfg;
}

std::vector<size_t> build_sizes(const BandwidthConfig &cfg) {
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

double benchmark_direction(void *dst, const void *src, size_t bytes, int iterations, hcMemcpyKind kind) {
    hcEvent_t start, stop;
    HC_CHECK(hcEventCreate(&start));
    HC_CHECK(hcEventCreate(&stop));

    HC_CHECK(hcEventRecord(start, 0));
    for (int i = 0; i < iterations; ++i) {
        HC_CHECK(hcMemcpy(dst, src, bytes, kind));
    }
    HC_CHECK(hcEventRecord(stop, 0));
    HC_CHECK(hcEventSynchronize(stop));

    float ms = 0.0f;
    HC_CHECK(hcEventElapsedTime(&ms, start, stop));

    HC_CHECK(hcEventDestroy(start));
    HC_CHECK(hcEventDestroy(stop));

    const double seconds = static_cast<double>(ms) / 1e3;
    const double total_bytes = static_cast<double>(bytes) * static_cast<double>(iterations);
    return (total_bytes / seconds) / 1e9;
}

int main(int argc, char **argv) {
    BandwidthConfig cfg = parse_args(argc, argv);
    std::vector<size_t> sizes = build_sizes(cfg);
    if (sizes.empty()) {
        fprintf(stderr, "No sizes to benchmark\n");
        return 1;
    }

    const size_t max_size = *std::max_element(sizes.begin(), sizes.end());

    void *h_src = nullptr;
    void *h_dst = nullptr;
    double *d_src = nullptr;
    double *d_dst = nullptr;

    HC_CHECK(hcMallocHost(&h_src, max_size, 0));
    HC_CHECK(hcMallocHost(&h_dst, max_size, 0));
    std::memset(h_src, 0xAB, max_size);
    std::memset(h_dst, 0, max_size);

    HC_CHECK(hcMalloc(reinterpret_cast<void **>(&d_src), max_size));
    HC_CHECK(hcMalloc(reinterpret_cast<void **>(&d_dst), max_size));

    printf("Memory bandwidth benchmark (bytes per transfer)\n");
    print_table_header({"Size(MB)", "H2D GB/s", "D2H GB/s", "D2D GB/s"});

    for (size_t bytes : sizes) {
        const double mb = static_cast<double>(bytes) / (1 << 20);
        const double h2d = benchmark_direction(d_src, h_src, bytes, cfg.iterations, hcMemcpyHostToDevice);
        const double d2h = benchmark_direction(h_dst, d_src, bytes, cfg.iterations, hcMemcpyDeviceToHost);
        const double d2d = benchmark_direction(d_dst, d_src, bytes, cfg.iterations, hcMemcpyDeviceToDevice);

        char size_buf[32];
        char h2d_buf[32];
        char d2h_buf[32];
        char d2d_buf[32];
        std::snprintf(size_buf, sizeof(size_buf), "%.1f", mb);
        std::snprintf(h2d_buf, sizeof(h2d_buf), "%.1f", h2d);
        std::snprintf(d2h_buf, sizeof(d2h_buf), "%.1f", d2h);
        std::snprintf(d2d_buf, sizeof(d2d_buf), "%.1f", d2d);
        print_table_row({size_buf, h2d_buf, d2h_buf, d2d_buf});
    }

    HC_CHECK(hcFree(d_src));
    HC_CHECK(hcFree(d_dst));
    HC_CHECK(hcFreeHost(h_src));
    HC_CHECK(hcFreeHost(h_dst));
    return 0;
}
