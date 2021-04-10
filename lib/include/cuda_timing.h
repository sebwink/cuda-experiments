#ifndef _TIMING_UTILS_HPP_
#define _TIMING_UTILS_HPP_

#define timeit__(message, ...) { \
    cudaEvent_t start, stop; \
    checkCudaErrors(cudaEventCreate(&start)); \
    checkCudaErrors(cudaEventCreate(&stop)); \
    checkCudaErrors(cudaEventRecord(start)); \
    __VA_ARGS__ \
    checkCudaErrors(cudaEventRecord(stop)); \
    checkCudaErrors(cudaEventSynchronize(stop)); \
    float time_ms = 0; \
    checkCudaErrors(cudaEventElapsedTime(&time_ms, start, stop)); \
    printf(message "%f (ms) \n", time_ms); \
    checkCudaErrors(cudaEventDestroy(start)); \
    checkCudaErrors(cudaEventDestroy(stop)); \
}

#endif
