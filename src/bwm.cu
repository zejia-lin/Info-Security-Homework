
#include <cuda_runtime.h>
#include <cublas.h>
#include <cusolverDn.h>

#include <iostream>
#include "myutils.cpp"
#include "mydct.cu"
#include "constants.h"

struct myhandle {
    cusolverDnHandle_t solver;
    cublasHandle_t blas;
    int lwork;
    float *work;
};


void gesvd(size_t rows, size_t cols, float *A, size_t lda, float *S, float *U, size_t ldu, float *V, size_t ldv, cudaStream_t stream=0){

    cusolverDnHandle_t cusolverHandle;
    gesvdjInfo_t gesvdinfo;
    int lwork;
    // float *work;
    int batch_size = (rows / TILE_DIM) * (cols / TILE_DIM);

    CUSOLVER_CHECK(cusolverDnCreate(&cusolverHandle));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverHandle, stream));
    CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdinfo));

    cusolverDnSgesvdjBatched_bufferSize(cusolverHandle, 
                                 CUSOLVER_EIG_MODE_VECTOR,
                                 rows, cols, 
                                 A, lda, S, U, ldu, V, ldv,
                                 &lwork, gesvdinfo, batch_size);

}


void tiled_add_wm(size_t rows, size_t cols, float *A, size_t lda, float *res, size_t ldres, float *workspace, cudaStream_t stream=0){

    dct_a100_best_param(rows, cols, A, lda, workspace, ldres, stream);



    idct_a100_best_param(rows, cols, workspace, lda, res, ldres, stream);

}


__global__ void trans_and_pack_continguous(size_t rows, size_t cols, float *A, size_t lda, float *C, size_t ldc){

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
        int tile_offset_to_A = tile_x * TILE_DIM * lda + tile_y * TILE_DIM;
        const float *tile_ptr_to_A = &A[tile_offset_to_A];
        
        // copy to shared memory
        sA[threadIdx.x * TILE_DIM * TILE_DIM + threadIdx.y + threadIdx.z * TILE_DIM] = 
                 tile_ptr_to_A[IDX(threadIdx.y, threadIdx.z, lda)]; // note that leading dimension is cols
        __syncthreads();

        // compute the starting address of current tile in sA
        float *tile_ptr_to_shared = &sA[threadIdx.x * TILE_DIM * TILE_DIM];
        float *elm_ptr_to_res = &C[tile_offset_to_A + threadIdx.y * ldc + threadIdx.z];

        *elm_ptr_to_res = tile_ptr_to_shared[threadIdx.y * TILE_DIM + threadIdx.z];
    }
}


int main(){
    int N = 8;
    float *dA, *dRes;
    cudaMallocManaged(&dA, sizeof(float) * (N + 1) * N);
    cudaMallocManaged(&dRes, sizeof(float) * (N + 1) * N);
    for (size_t i = 0; i < N; ++i) {
        for(size_t j = 0; j < N; ++j){
            dA[i + j * (N + 1)] = i + j * N;
        }
    }
    print_matrix_rowmaj(dA, N, N + 1, N + 1);
    cudaMemPrefetchAsync(dA, sizeof(float) * (N + 1) * N, 0);
    cudaMemPrefetchAsync(dRes, sizeof(float) * (N + 1) * N, 0);
    cudaDeviceSynchronize();
    dim3 dimGrid(1024);
    dim3 dimgBlock(8, TILE_DIM, TILE_DIM);
    size_t smemSize = TILE_DIM * TILE_DIM * sizeof(int);
    for(int _iter = 0; _iter < 1; ++_iter){
        __TIMER_START__(duration)
        trans_and_pack_continguous<<<dimGrid, dimgBlock, smemSize>>>(N, N, dA, N + 1, dRes, N + 1);
        cudaDeviceSynchronize();
        __TIMER_STOP__(duration);
        std::cout << "Transpose in "<< duration / 1000 << " ms\n";
    }
    print_matrix_rowmaj(dRes, N + 1, N, N + 1);
}



int maind(int argc, char **argv){

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Using device " << device << " " << prop.name << std::endl;

    int N = atoi(argv[1]);
    int rows = N;
    int cols = N;

    float *A, *AT, *U, *S, *V;
    int *info;
    int lda = N;
    int ldu = N;
    int ldv = N;

    cudaStream_t stream = NULL;
    cublasHandle_t blasHandle;
    cusolverDnHandle_t solverHandle;
    gesvdjInfo_t gesvdParams;
    int lwork;
    float *work;
    int batchSize = (rows / TILE_DIM) * (cols / TILE_DIM);

    CUDA_CHECK(cudaMallocManaged(&info, sizeof(int) * batchSize));
    CUDA_CHECK(cudaMallocManaged(&AT, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&A, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&U, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&V, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&S, sizeof(float) * batchSize * TILE_DIM));

    int bb = myreadbin("../out/A.bin", AT);

    CUDA_CHECK(cudaMemPrefetchAsync(AT, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(A, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(U, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(V, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(S, sizeof(float) * N, device, stream));

    CUSOLVER_CHECK(cusolverDnCreate(&solverHandle));
    CUSOLVER_CHECK(cusolverDnSetStream(solverHandle, stream));
    CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdParams));
    CUBLAS_CHECK(cublasCreate(&blasHandle));
    CUBLAS_CHECK(cublasSetStream(blasHandle, stream));

    const float one = 1, zero = 0;
    CUBLAS_CHECK(cublasSgeam(blasHandle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, &one, AT, lda, &zero, A, lda, A, lda));
    

    CUSOLVER_CHECK(cusolverDnSgesvdjBatched_bufferSize(solverHandle, 
                                 CUSOLVER_EIG_MODE_VECTOR,
                                 TILE_DIM, TILE_DIM, 
                                 A, lda, S, U, ldu, V, ldv,
                                 &lwork, gesvdParams, batchSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&work), sizeof(float) * lwork));

    CUSOLVER_CHECK(cusolverDnSgesvdjBatched(solverHandle, CUSOLVER_EIG_MODE_VECTOR, 
                TILE_DIM, TILE_DIM, 
                A, lda, S, U, ldu, V, ldv,
                work, lwork, info, gesvdParams, batchSize));
    CUDA_CHECK(cudaDeviceSynchronize());

    writebin("../out/U.bin", U, sizeof(float) * rows * cols);
    writebin("../out/V.bin", V, sizeof(float) * rows * cols);
    writebin("../out/S.bin", S, sizeof(float) * batchSize * TILE_DIM);

    // print_matrix_colmaj(A, rows, cols, lda);
    // print_matrix_colmaj(U, rows, cols, lda);
    // print_matrix_colmaj(V, rows, cols, lda);
    // print_matrix_rowmaj(S, 1, N, lda);

}
