#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <helper_cuda.h>

#include "cuda_timing.h"

const int NB_THREADS_PER_BLOC = 256;

__global__
void add(int size, int *d_C, int *d_A, int *d_B) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        d_C[tid] = d_A[tid] + d_B[tid];
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: add nb components\n");
        exit(0);
    }
    int size = atoi(argv[1]);
    int i;
    // CPU memory
    int *h_arrayA = (int*)malloc(size * sizeof(int));
    int *h_arrayB = (int*)malloc(size * sizeof(int));
    int *h_arrayC = (int*)malloc(size * sizeof(int));
    int *h_arrayCgpu = (int*)malloc(size * sizeof(int));
    // GPU memory
    int *d_arrayA, *d_arrayB, *d_arrayC;
    checkCudaErrors(
        cudaMalloc((void**)&d_arrayA, size * sizeof(int))
    );
    checkCudaErrors(
        cudaMalloc((void**)&d_arrayB, size * sizeof(int))
    );
    checkCudaErrors(
        cudaMalloc((void**)&d_arrayC, size * sizeof(int))
    );
    // CPU loop
    for (i = 0; i < size; i++) {
        h_arrayA[i] = i;
        h_arrayB[i] = 2*i;
    }
	timeit__("CPU processing time: ", {
		for (i = 0; i < size; i++) {
			h_arrayC[i] = h_arrayA[i] + h_arrayB[i];
		}
	})
	// GPU kernel loop
	int nb_blocs = (size + NB_THREADS_PER_BLOC - 1) / NB_THREADS_PER_BLOC;
    timeit__("GPU processing time: ", {
		checkCudaErrors(
		    cudaMemcpy(
		        d_arrayA, h_arrayA, size * sizeof(int),
				cudaMemcpyHostToDevice
		    )
        );
        checkCudaErrors(
		    cudaMemcpy(
		        d_arrayB, h_arrayB, size * sizeof(int),
			    cudaMemcpyHostToDevice
		    )
        );
	    add<<<nb_blocs, NB_THREADS_PER_BLOC>>>(size, d_arrayC, d_arrayA, d_arrayB);
        checkCudaErrors(
		    cudaMemcpy(
		        h_arrayCgpu, d_arrayC, size * sizeof(int),
			    cudaMemcpyDeviceToHost
		    )
        );
	})	
    // Check equivalence
	for (i = 0; i < size; i++) {
		assert(h_arrayC[i] == h_arrayCgpu[i]);
	}
	checkCudaErrors(cudaFree(d_arrayA));
	checkCudaErrors(cudaFree(d_arrayB));
	checkCudaErrors(cudaFree(d_arrayC));
	free(h_arrayA);
	free(h_arrayB);
	free(h_arrayC);
	free(h_arrayCgpu);
	return 0;
}
