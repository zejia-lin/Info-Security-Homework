
#include <cuda_runtime.h>
#include <cublas.h>
#include <cusolverDn.h>

#include <iostream>
#include "myutils.cpp"
#include "mmd.cu"
#include "mydct.cu"
#include "constants.h"


void tiled_add_wm(size_t rows, size_t cols, float *A, size_t lda, float *res, size_t ldres, float *workspace, cudaStream_t stream=0){

    dct_a100_best_param(rows, cols, A, lda, workspace, ldres, stream);



    idct_a100_best_param(rows, cols, workspace, lda, res, ldres, stream);

}


__global__ void gpu_trans_and_pack_continguous(size_t rows, size_t cols, float *A, size_t lda, float *C, size_t ldc){

    // shared memory size equals to blockDim
    extern __shared__ float sA[];

    int tile_id = threadIdx.x + blockIdx.x * blockDim.x;
    int tile_per_row = cols / TILE_DIM;
    int num_tiles = (rows / TILE_DIM) * (cols / TILE_DIM);
    
    // grid stride loop
#pragma unroll
    for(; tile_id < num_tiles; tile_id += gridDim.x){

        // compute the starting address of current tile in A
        int tile_x = tile_id / tile_per_row;
        int tile_y = tile_id % tile_per_row;
        const float *tile_ptr_to_A = &A[tile_x * TILE_DIM * lda + tile_y * TILE_DIM];
        float *tile_ptr_to_shared = &sA[threadIdx.x * TILE_DIM * TILE_DIM];
        float *tile_ptr_to_res = &C[tile_id * TILE_DIM * TILE_DIM];
        
        // copy to shared memory
        tile_ptr_to_shared[threadIdx.y + threadIdx.z * TILE_DIM] = 
                 tile_ptr_to_A[IDX(threadIdx.y, threadIdx.z, lda)]; // note that leading dimension is cols
        __syncthreads();

        tile_ptr_to_res[threadIdx.y * TILE_DIM + threadIdx.z] = tile_ptr_to_shared[threadIdx.y * TILE_DIM + threadIdx.z];
    }
}

__global__ void gpu_unpack_and_trans(size_t rows, size_t cols, const float *A, size_t lda, float *C, size_t ldc){
    // shared memory size equals to blockDim
    extern __shared__ float sA[];

    int tile_id = threadIdx.x + blockIdx.x * blockDim.x;
    int tile_per_row = cols / TILE_DIM;
    int num_tiles = (rows / TILE_DIM) * (cols / TILE_DIM);
    
    // grid stride loop
#pragma unroll
    for(; tile_id < num_tiles; tile_id += gridDim.x){

        // compute the starting address of current tile in A
        int tile_x = tile_id / tile_per_row;
        int tile_y = tile_id % tile_per_row;
        float *tile_ptr_to_A = &C[tile_x * TILE_DIM * lda + tile_y * TILE_DIM];
        float *tile_ptr_to_shared = &sA[threadIdx.x * TILE_DIM * TILE_DIM];
        const float *tile_ptr_to_res = &A[tile_id * TILE_DIM * TILE_DIM];
        
        tile_ptr_to_shared[threadIdx.y * TILE_DIM + threadIdx.z] = tile_ptr_to_res[threadIdx.y * TILE_DIM + threadIdx.z];
        // copy to shared memory
        tile_ptr_to_A[IDX(threadIdx.y, threadIdx.z, lda)] = tile_ptr_to_shared[threadIdx.y + threadIdx.z * TILE_DIM];
        __syncthreads();

        // printf("(%d, %d, %d): %d\n", tile_id, threadIdx.y, threadIdx.z, tile_x * TILE_DIM * lda + tile_y * TILE_DIM);

    }
}


int mtxtp_a100_best_param(bool input, size_t rows, size_t cols, float *A, size_t lda, float *C, size_t ldc, cudaStream_t stream=0){
    dim3 dimGrid(1024);
    dim3 dimgBlock(8, TILE_DIM, TILE_DIM);
    size_t smemSize = TILE_DIM * TILE_DIM * sizeof(int);
    __TIMER_START__(dur);
    if (input) {
        gpu_trans_and_pack_continguous<<<dimGrid, dimgBlock, smemSize, stream>>>(rows, cols, A, lda, C, ldc);
    } else {
        gpu_unpack_and_trans<<<dimGrid, dimgBlock, smemSize, stream>>>(rows, cols, A, lda, C, ldc);
    }
    __TIMER_STOP__(dur);
    std::cout << "Trans: " << dur << std::endl;
}




int main(int argc, char **argv){

    __TIMER_START__(end2end);

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Using device " << device << " " << prop.name << std::endl;

    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);

    float *A, *U, *S, *V, *inv;
    int *info;

    cudaStream_t stream = NULL;
    cublasHandle_t blasHandle;
    cusolverDnHandle_t solverHandle;
    gesvdjInfo_t gesvdParams;
    int lwork;
    float *work;
    int batchSize = (rows / TILE_DIM) * (cols / TILE_DIM);
    int numTiles = (rows / TILE_DIM) * (cols / TILE_DIM);
    const float one = 1, zero = 0;


    CUDA_CHECK(cudaMallocManaged(&info, sizeof(int) * batchSize));
    CUDA_CHECK(cudaMallocManaged(&A, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&U, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&V, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&inv, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&S, sizeof(float) * numTiles * TILE_DIM));

    int bb = myreadbin("../out/A.bin", A);

    CUDA_CHECK(cudaMemPrefetchAsync(info, sizeof(int) * batchSize, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(A, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(U, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(V, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(S, sizeof(float) * numTiles * TILE_DIM, device, stream));

    CUSOLVER_CHECK(cusolverDnCreate(&solverHandle));
    CUSOLVER_CHECK(cusolverDnSetStream(solverHandle, stream));
    CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdParams));
    CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(gesvdParams, 1e-3));
    CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdParams, 1000));
    CUBLAS_CHECK(cublasCreate(&blasHandle));
    CUBLAS_CHECK(cublasSetStream(blasHandle, stream));
    CUSOLVER_CHECK(cusolverDnSgesvdjBatched_bufferSize(solverHandle, 
                                 CUSOLVER_EIG_MODE_VECTOR,
                                 TILE_DIM, TILE_DIM, 
                                 A, TILE_DIM, S, U, TILE_DIM, V, TILE_DIM,
                                 &lwork, gesvdParams, batchSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&work), sizeof(float) * lwork));

    std::cout << "Allocated " << lwork << " float buffer for gesvd\n";

    __TIMER_START__(computation);
    CUSOLVER_CHECK(cusolverDnSgesvdjBatched(solverHandle, CUSOLVER_EIG_MODE_VECTOR, 
                TILE_DIM, TILE_DIM, 
                A, TILE_DIM, S, U, TILE_DIM, V, TILE_DIM,
                work, lwork, info, gesvdParams, batchSize));
    CUDA_CHECK(cudaDeviceSynchronize());

    mmd_batched_a100_best_param(false, U, S, inv, batchSize);
    cublasGemmStridedBatchedEx(
        blasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
        TILE_DIM, TILE_DIM, TILE_DIM,
        &one,
        inv, CUDA_R_32F, TILE_DIM, TILE_DIM * TILE_DIM,
        V, CUDA_R_32F, TILE_DIM, TILE_DIM * TILE_DIM,
        &zero,
        inv, CUDA_R_32F, TILE_DIM, TILE_DIM * TILE_DIM,
        batchSize, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT
    );
    cudaDeviceSynchronize();
    __TIMER_STOP__(computation);


    for(int i = 0; i < batchSize; ++i){
        if (0 == info[i]) {
            // std::printf("matrix %d: gesvdj converges \n", i);
        } else if (0 > info[i]) {
            std::printf("Error: %d-th parameter is wrong \n", -info[i]);
            exit(1);
        } else {
            std::printf("WARNING: matrix %d, info = %d : gesvdj does not converge \n", i, info[i]);
        }
    }

    // std::cout << "====================\nGemm from GPU\n";
    // print_matrix_rowmaj(inv, 8, 8, 8);

    writebin("../out/inv.bin", inv, sizeof(float) * rows * cols);

    std::cout << "GPU computation " << computation << " ms\n";
    __TIMER_STOP__(end2end);
    std::cout << "GPU end to end " << end2end << " ms\n";

    std::cout << "Exit bwm with 0\n";

}