
// #pragma once

#include "myutils.cpp"
#include "constants.h"

/**
 * Matrix multiply diagonal, the API is similar to cublasGemmBatched. 
 * Should be launched with 3D block (__, TILE_DIM, TILE_DIM) and 1D grid.
 * Matrix stored in column major.
*/
__global__ void gpu_mmd_batched(float *A, float *D, float *res, size_t batchSize){
    size_t tile_id = threadIdx.x + blockIdx.x * blockDim.x;
    for(; tile_id < batchSize; tile_id += blockDim.x){
        size_t offset = threadIdx.y + threadIdx.z * TILE_DIM + tile_id * TILE_DIM * TILE_DIM;
        res[offset] = A[offset] * D[threadIdx.y + tile_id * TILE_DIM];
    }
}

void mmd_batched_a100_best_param(float *A, float *D, float *res, size_t batchSize, cudaStream_t stream=0){
    dim3 dimGrid = dim3(512);
    dim3 dimBlock = dim3(32, TILE_DIM, TILE_DIM);
    gpu_mmd_batched<<<dimGrid, dimBlock, 0, stream>>>(A, D, res, batchSize);
}

int main(){

    float *A, *D, *res;
    int N = 8;
    cudaMallocManaged(&A, sizeof(float) * N * N);
    cudaMallocManaged(&res, sizeof(float) * N * N);
    cudaMallocManaged(&D, sizeof(float) * 16);

    for(int i = 0; i < N * N; ++i){
        A[(i % N) * N + i / N] = i + 1;
    }

    D[0] = 0.001;
    for(int i = 1; i < 16; ++i){
        D[i] = D[i - 1] * 10;
    }

    cudaMemPrefetchAsync(A, sizeof(float) * N * N, 0);
    cudaMemPrefetchAsync(res, sizeof(float) * N * N, 0);
    cudaMemPrefetchAsync(D, sizeof(float) * 16, 0);
    cudaDeviceSynchronize();

    __TIMER_START__(compute_time);

    mmd_batched_a100_best_param(A, D, res, 4);
    cudaDeviceSynchronize();

    __TIMER_STOP__(compute_time);
    std::cout << "Compute " << compute_time << " ms\n";

    print_matrix_colmaj(A, N, N, N);
    print_matrix_colmaj(res, N, N, N);

}

