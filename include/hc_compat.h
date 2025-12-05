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

#define hcSuccess cudaSuccess
#define hcErrorInvalidValue cudaErrorInvalidValue
#define hcInvalidDevice cudaErrorInvalidDevice
#define hcblasCreate cublasCreate
#define hcblasDestroy cublasDestroy
#define hcblasSetStream cublasSetStream
#define hcblasDgemm cublasDgemm
#define hcblasSgemm cublasSgemm
#define hcblasGemmEx cublasGemmEx
#define hcblasOperation_t cublasOperation_t
#define HCBLAS_OP_N CUBLAS_OP_N
#define HCBLAS_OP_T CUBLAS_OP_T
#define HCBLAS_COMPUTE_32F CUBLAS_COMPUTE_32F
#define HCBLAS_GEMM_DEFAULT CUBLAS_GEMM_DEFAULT

#define HC_R_16F CUDA_R_16F
#define HC_R_32F CUDA_R_32F
#define HC_R_64F CUDA_R_64F

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
#define hcHostAlloc cudaHostAlloc
#define hcHostFree cudaFreeHost
#define hcStreamCreate cudaStreamCreate
#define hcStreamDestroy cudaStreamDestroy
#define hcStreamSynchronize cudaStreamSynchronize
#define HCBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS
#else
#include <hc_runtime_api.h>
#include "hcblas/hcblas.h"
#endif
