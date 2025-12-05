#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "benchmark_utils_cuda.h"

struct InfoOptions {
    int device = -1;
};

InfoOptions parse_args(int argc, char **argv) {
    InfoOptions opts;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            opts.device = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: cuda_device_info [--device N]\n");
            std::exit(0);
        }
    }
    return opts;
}

std::string format_bytes(double bytes) {
    static const char *kUnits[] = {"B", "KB", "MB", "GB", "TB"};
    int idx = 0;
    while (bytes >= 1024.0 && idx < 4) {
        bytes /= 1024.0;
        ++idx;
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f %s", bytes, kUnits[idx]);
    return std::string(buf);
}

bool get_attribute(int device, cudaDeviceAttr attr, int *value) {
    return cudaDeviceGetAttribute(value, attr, device) == cudaSuccess;
}

void print_device_info(int device_id) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    char pci_bus[64] = "N/A";
    if (cudaDeviceGetPCIBusId(pci_bus, sizeof(pci_bus), device_id) != cudaSuccess) {
        std::strncpy(pci_bus, "N/A", sizeof(pci_bus));
        pci_bus[sizeof(pci_bus) - 1] = '\0';
    }

    int sm_count = prop.multiProcessorCount;
    get_attribute(device_id, cudaDevAttrMultiProcessorCount, &sm_count);

    int l1_bytes = prop.sharedMemPerMultiprocessor;
    get_attribute(device_id, cudaDevAttrMaxSharedMemoryPerMultiprocessor, &l1_bytes);
#ifdef cudaDevAttrL1CacheSizePerMultiprocessor
    int l1_cache = 0;
    if (get_attribute(device_id, cudaDevAttrL1CacheSizePerMultiprocessor, &l1_cache)) {
        l1_bytes = l1_cache;
    }
#endif

    int l2_bytes = prop.l2CacheSize;
    get_attribute(device_id, cudaDevAttrL2CacheSize, &l2_bytes);

    int warp_size = prop.warpSize;
    get_attribute(device_id, cudaDevAttrWarpSize, &warp_size);

    int clock_khz = prop.clockRate;
    get_attribute(device_id, cudaDevAttrClockRate, &clock_khz);

    int memory_clock_khz = 0;
    get_attribute(device_id, cudaDevAttrMemoryClockRate, &memory_clock_khz);

    int memory_bus_width = prop.memoryBusWidth;
    get_attribute(device_id, cudaDevAttrGlobalMemoryBusWidth, &memory_bus_width);

    printf("CUDA Device %d: %s\n", device_id, prop.name);
    printf("  PCI Bus ID           : %s\n", pci_bus);
    printf("  Compute Capability   : %d.%d\n", prop.major, prop.minor);
    printf("  Streaming Multiprocessors : %d\n", sm_count);
    printf("  Warp Size            : %d\n", warp_size);
    printf("  Max Threads/Block    : %d\n", prop.maxThreadsPerBlock);
    printf("  Shared Mem per SM (L1): %s\n", format_bytes(static_cast<double>(l1_bytes)).c_str());
    printf("  L2 Cache Size        : %s\n", format_bytes(static_cast<double>(l2_bytes)).c_str());
    printf("  Global Memory        : %s\n", format_bytes(static_cast<double>(prop.totalGlobalMem)).c_str());
    printf("  Core Clock           : %.2f MHz\n", static_cast<double>(clock_khz) / 1000.0);
    if (memory_clock_khz > 0) {
        printf("  Memory Clock         : %.2f MHz\n", static_cast<double>(memory_clock_khz) / 1000.0);
    }
    if (memory_bus_width > 0) {
        printf("  Memory Bus Width     : %d-bit\n", memory_bus_width);
    }
    printf("  Max Threads Dim      : %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1],
           prop.maxThreadsDim[2]);
    printf("  Max Grid Size        : %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("\n");
}

int main(int argc, char **argv) {
    InfoOptions opts = parse_args(argc, argv);
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("No CUDA-capable devices detected.\n");
        return 0;
    }

    std::vector<int> devices;
    if (opts.device >= 0) {
        if (opts.device >= device_count) {
            fprintf(stderr, "Requested device %d out of range (0-%d)\n", opts.device, device_count - 1);
            return 1;
        }
        devices.push_back(opts.device);
    } else {
        for (int i = 0; i < device_count; ++i) {
            devices.push_back(i);
        }
    }

    for (int dev : devices) {
        print_device_info(dev);
    }
    return 0;
}

