#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <helper_cuda.h>
#include <cublas_v2.h>

#include "cuda_timing.h"

const int NB_THREADS_PER_BLOC = 256;

__global__
void add(int size, double *d_C, double *d_A, double *d_B) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        d_C[tid] = d_A[tid] + d_B[tid];
    }
}

__global__
void inv(int size, double *d_x) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size) {
		d_x[tid] = 1. / d_x[tid];
	}
}

__global__
void inv_dot(int size, double *d_x, double *d_y, double *dot) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size) {
		*dot += (1. / d_x[tid]) * (1. / d_y[tid]);	
	}
}


int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: add nb components\n");
        exit(0);
    }
    int size = atoi(argv[1]);
    int i;
	cublasStatus_t cublas_status;
	cublasHandle_t cublas;
	cublas_status = cublasCreate(&cublas);
    // CPU memory
    double *h_arrayA = (double*)malloc(size * sizeof(double));
    double *h_arrayB = (double*)malloc(size * sizeof(double));
    double *h_arrayC = (double*)malloc(size * sizeof(double));
    double *h_arrayCgpu = (double*)malloc(size * sizeof(double));
    // GPU memory
    double *d_arrayA, *d_arrayB, *d_arrayC;
    checkCudaErrors(
        cudaMalloc((void**)&d_arrayA, size * sizeof(double))
    );
    checkCudaErrors(
        cudaMalloc((void**)&d_arrayB, size * sizeof(double))
    );
    checkCudaErrors(
        cudaMalloc((void**)&d_arrayC, size * sizeof(double))
    );
    for (i = 0; i < size; i++) {
        h_arrayA[i] = i + 1;
        h_arrayB[i] = 2 * (i + 1);
    }
    // CPU loop
    double cpu_dot = 0.0;
	timeit__("CPU processing time: ", {
		for (i = 0; i < size; i++) {
			h_arrayC[i] = h_arrayA[i] + h_arrayB[i];
		    cpu_dot += (1. / h_arrayC[i]) * (1. / h_arrayA[i]);
		}
	})
	// GPU kernel loop
	int nb_blocs = (size + NB_THREADS_PER_BLOC - 1) / NB_THREADS_PER_BLOC;
	double gpu_dot_1 = 0.0;
    timeit__("GPU processing time (inv + cublasDdot): ", {
        cublas_status = cublasSetVector(size, sizeof(double), h_arrayA, 1, d_arrayA, 1);
        cublas_status = cublasSetVector(size, sizeof(double), h_arrayB, 1, d_arrayB, 1);
	    add<<<nb_blocs, NB_THREADS_PER_BLOC>>>(size, d_arrayC, d_arrayA, d_arrayB);
		inv<<<nb_blocs, NB_THREADS_PER_BLOC>>>(size, d_arrayA);
		inv<<<nb_blocs, NB_THREADS_PER_BLOC>>>(size, d_arrayC);
		cublas_status = cublasDdot(cublas, size, d_arrayC, 1, d_arrayA, 1, &gpu_dot_1);
	})	
	/*
    double *gpu_dot_2 = (double*)malloc(sizeof(double));
	double *gpu_dot_2_gpu;
    checkCudaErrors(
        cudaMalloc((void**)&gpu_dot_2_gpu, sizeof(double))
    );
	*gpu_dot_2_gpu = 0.0;
    timeit__("GPU processing time (inv_dot): ", {
        cublas_status = cublasSetVector(size, sizeof(double), h_arrayA, 1, d_arrayA, 1);
        cublas_status = cublasSetVector(size, sizeof(double), h_arrayB, 1, d_arrayB, 1);
	    add<<<nb_blocs, NB_THREADS_PER_BLOC>>>(size, d_arrayC, d_arrayA, d_arrayB);
		inv_dot<<<nb_blocs, NB_THREADS_PER_BLOC>>>(size, d_arrayA, d_arrayC, gpu_dot_2_gpu);
        checkCudaErrors(
		    cudaMemcpy(
		        gpu_dot_2_gpu, gpu_dot_2, sizeof(double),
			    cudaMemcpyDeviceToHost
		    )
        );
    })
	*/
    // Check equivalence
	// assert(cpu_dot == gpu_dot);
	// Clean up
	checkCudaErrors(cudaFree(d_arrayA));
	checkCudaErrors(cudaFree(d_arrayB));
	checkCudaErrors(cudaFree(d_arrayC));
	free(h_arrayA);
	free(h_arrayB);
	free(h_arrayC);
	free(h_arrayCgpu);
	cublasDestroy(cublas);
	return 0;
}
