#pragma once

// Wrapper header to make it easy to flip between the native hc runtime
// and CUDA when the hc SDK is not available on the development machine.

#if defined(USE_CUDA_FALLBACK)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

typedef cudaError_t hcError_t;
typedef cudaStream_t hcStream_t;
typedef cudaEvent_t hcEvent_t;
typedef cudaMemcpyKind hcMemcpyKind;
typedef cublasOperation_t hcblasOperation_t;
typedef cublasHandle_t hcblasHandle_t;
typedef cublasStatus_t hcblasStatus_t;
typedef __half hcHalf_t;
typedef cublasComputeType_t hcblasComputeType_t;
typedef cublasMath_t hcblasMath_t;
typedef cudaDeviceProp hcDeviceProp_t;
typedef cudaDeviceAttr hcDeviceAttribute_t;

#define hcSuccess cudaSuccess
#define hcErrorInvalidValue cudaErrorInvalidValue
#define hcInvalidDevice cudaErrorInvalidDevice
#define hcblasCreate cublasCreate
#define hcblasDestroy cublasDestroy
#define hcblasSetStream cublasSetStream
#define hcblasSetMathMode cublasSetMathMode
#define hcblasDgemm cublasDgemm
#define hcblasSgemm cublasSgemm
#define hcblasGemmEx cublasGemmEx
#define hcblasOperation_t cublasOperation_t
#define HCBLAS_OP_N CUBLAS_OP_N
#define HCBLAS_OP_T CUBLAS_OP_T
#define HCBLAS_COMPUTE_32F CUBLAS_COMPUTE_32F
#define HCBLAS_COMPUTE_32F_FAST_TF32 CUBLAS_COMPUTE_32F_FAST_TF32
#define HCBLAS_GEMM_DEFAULT CUBLAS_GEMM_DEFAULT
#define HCBLAS_GEMM_DEFAULT_TENSOR_OP CUBLAS_GEMM_DEFAULT_TENSOR_OP
#define HCBLAS_DEFAULT_MATH CUBLAS_DEFAULT_MATH
#define HCBLAS_TF32_TENSOR_OP_MATH CUBLAS_TF32_TENSOR_OP_MATH

#define HPCC_R_16F CUDA_R_16F
#define HPCC_R_32F CUDA_R_32F
#define HPCC_R_64F CUDA_R_64F

#define hcGetErrorString cudaGetErrorString
#define hcMalloc cudaMalloc
#define hcFree cudaFree
#define hcMemcpy cudaMemcpy
#define hcMemcpyPeer cudaMemcpyPeer
#define hcMemset cudaMemset
#define hcMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hcMemcpyHostToDevice cudaMemcpyHostToDevice
#define hcMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hcDeviceSynchronize cudaDeviceSynchronize
#define hcPeekAtLastError cudaPeekAtLastError
#define hcEventCreate cudaEventCreate
#define hcEventDestroy cudaEventDestroy
#define hcEventRecord cudaEventRecord
#define hcEventSynchronize cudaEventSynchronize
#define hcEventElapsedTime cudaEventElapsedTime
#define hcMallocHost cudaMallocHost
#define hcFreeHost cudaFreeHost
#define hcStreamCreate cudaStreamCreate
#define hcStreamDestroy cudaStreamDestroy
#define hcStreamSynchronize cudaStreamSynchronize
#define hcGetDeviceCount cudaGetDeviceCount
#define hcGetDeviceProperties cudaGetDeviceProperties
#define hcDeviceGetAttribute cudaDeviceGetAttribute
#define hcDeviceGetPCIBusId cudaDeviceGetPCIBusId
#define hcDeviceGet cudaDeviceGet
#define hcSetDevice cudaSetDevice
#define hcGetDevice cudaGetDevice
#define HCBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS

#define hcDevAttrMultiprocessorCount cudaDevAttrMultiProcessorCount
#define hcDevAttrMaxSharedMemoryPerMultiprocessor cudaDevAttrMaxSharedMemoryPerMultiprocessor
#define hcDevAttrMaxSharedMemoryPerBlock cudaDevAttrMaxSharedMemoryPerBlock
#define hcDevAttrL2CacheSize cudaDevAttrL2CacheSize
#define hcDevAttrWarpSize cudaDevAttrWarpSize
#define hcDevAttrClockRate cudaDevAttrClockRate
#define hcDevAttrMemoryClockRate cudaDevAttrMemoryClockRate
#define hcDevAttrGlobalMemoryBusWidth cudaDevAttrGlobalMemoryBusWidth
#ifdef cudaDevAttrL1CacheSizePerMultiprocessor
#define hcDevAttrL1CacheSizePerMultiprocessor cudaDevAttrL1CacheSizePerMultiprocessor
#endif
#else
#include <hc_runtime_api.h>
#include "hcblas/hcblas.h"

typedef hcblas_half hcHalf_t;
#endif
