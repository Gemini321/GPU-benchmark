#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "benchmark_utils.h"

struct InfoOptions {
    int device = -1;
};

InfoOptions parse_args(int argc, char **argv) {
    InfoOptions opts;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            opts.device = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: device_info [--device N]\n");
            std::exit(0);
        }
    }
    return opts;
}

bool get_attribute(int device, hcDeviceAttribute_t attr, int *value) {
    const hcError_t status = hcDeviceGetAttribute(value, attr, device);
    return status == hcSuccess;
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

void print_device_info(int device_id) {
    hcDeviceProp_t prop;
    HC_CHECK(hcGetDeviceProperties(&prop, device_id));

    char pci_bus[64] = "N/A";
    if (hcDeviceGetPCIBusId(pci_bus, sizeof(pci_bus), device_id) != hcSuccess) {
        std::strncpy(pci_bus, "N/A", sizeof(pci_bus));
        pci_bus[sizeof(pci_bus) - 1] = '\0';
    }

    int sm_count = prop.multiProcessorCount;
#ifdef hcDevAttrMultiprocessorCount
    int attr_val = 0;
    if (get_attribute(device_id, hcDevAttrMultiprocessorCount, &attr_val)) {
        sm_count = attr_val;
    }
#endif

    size_t l1_bytes = prop.sharedMemPerMultiprocessor;
#ifdef hcDevAttrMaxSharedMemoryPerMultiprocessor
    int l1_attr = 0;
    if (get_attribute(device_id, hcDevAttrMaxSharedMemoryPerMultiprocessor, &l1_attr)) {
        l1_bytes = static_cast<size_t>(l1_attr);
    }
#endif

    size_t l2_bytes = prop.l2CacheSize;
#ifdef hcDevAttrL2CacheSize
    int l2_attr = 0;
    if (get_attribute(device_id, hcDevAttrL2CacheSize, &l2_attr)) {
        l2_bytes = static_cast<size_t>(l2_attr);
    }
#endif

    int warp_size = prop.warpSize;
#ifdef hcDevAttrWarpSize
    int warp_attr = 0;
    if (get_attribute(device_id, hcDevAttrWarpSize, &warp_attr)) {
        warp_size = warp_attr;
    }
#endif

    int clock_khz = prop.clockRate;
#ifdef hcDevAttrClockRate
    int core_clock = 0;
    if (get_attribute(device_id, hcDevAttrClockRate, &core_clock)) {
        clock_khz = core_clock;
    }
#endif

    int memory_clock_khz = 0;
#ifdef hcDevAttrMemoryClockRate
    int mem_clock = 0;
    if (get_attribute(device_id, hcDevAttrMemoryClockRate, &mem_clock)) {
        memory_clock_khz = mem_clock;
    }
#endif

    int memory_bus_width = 0;
#ifdef hcDevAttrGlobalMemoryBusWidth
    int bus_attr = 0;
    if (get_attribute(device_id, hcDevAttrGlobalMemoryBusWidth, &bus_attr)) {
        memory_bus_width = bus_attr;
    }
#endif

    printf("Device %d: %s\n", device_id, prop.name);
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
    HC_CHECK(hcGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("No hc-capable devices detected.\n");
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

