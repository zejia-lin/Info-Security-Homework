
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


void mtxtp_a100_best_param(bool input, size_t rows, size_t cols, float *A, size_t lda, float *C, size_t ldc, cudaStream_t stream=0){
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

void init_cudalib(cusolverDnHandle_t *solverHandle, cublasHandle_t *blasHandle, size_t batchSize, 
                    float *A, float *U, float *S, float *V,
                    float **work, int *lwork, gesvdjInfo_t *gesvdParams, cudaStream_t stream){
    CUSOLVER_CHECK(cusolverDnCreate(solverHandle));
    CUSOLVER_CHECK(cusolverDnSetStream(*solverHandle, stream));
    CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(gesvdParams));
    CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(*gesvdParams, 1e-3));
    CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(*gesvdParams, 1000));
    CUBLAS_CHECK(cublasCreate(blasHandle));
    CUBLAS_CHECK(cublasSetStream(*blasHandle, stream));
    CUSOLVER_CHECK(cusolverDnSgesvdjBatched_bufferSize(*solverHandle, 
                                 CUSOLVER_EIG_MODE_VECTOR,
                                 TILE_DIM, TILE_DIM, 
                                 A, TILE_DIM, S, U, TILE_DIM, V, TILE_DIM,
                                 lwork, *gesvdParams, batchSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(work), sizeof(float) * (*lwork)));
}


void gesvd_a100_best_param(cusolverDnHandle_t solverHandle, size_t batchSize, 
                            float *A, float *U, float *S, float *V, 
                            float *work, int lwork, int *info, gesvdjInfo_t gesvdParams){
    CUSOLVER_CHECK(cusolverDnSgesvdjBatched(solverHandle, CUSOLVER_EIG_MODE_VECTOR, 
                TILE_DIM, TILE_DIM, 
                A, TILE_DIM, S, U, TILE_DIM, V, TILE_DIM,
                work, lwork, info, gesvdParams, batchSize));
}

void invsvd_a100_best_param(cublasHandle_t blasHandle, size_t batchSize, float *inv, float *U, float *S, float *V){
    const float one = 1, zero = 0;
    mmd_batched_a100_best_param(false, U, S, inv, batchSize);
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        blasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
        TILE_DIM, TILE_DIM, TILE_DIM,
        &one,
        inv, CUDA_R_32F, TILE_DIM, TILE_DIM * TILE_DIM,
        V, CUDA_R_32F, TILE_DIM, TILE_DIM * TILE_DIM,
        &zero,
        inv, CUDA_R_32F, TILE_DIM, TILE_DIM * TILE_DIM,
        batchSize, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    ));
}


__global__ void gpu_tiled_add_wm(size_t batchSize, float *S, uint8_t *wm, size_t wmlen, size_t mod1){
    size_t tile_id = threadIdx.x + blockIdx.x * blockDim.x;
    for(; tile_id < batchSize; tile_id += blockDim.x * gridDim.x){
        int bbyt = wm[tile_id % wmlen];
        // printf("(%lld): %d, %d,%f\n", tile_id, bbyt, (1 << (tile_id % 8)), (S[tile_id * TILE_DIM] / mod1 + 0.25 + 0.5 * bbyt) * mod1);
        S[tile_id * TILE_DIM] = (int(S[tile_id * TILE_DIM] / mod1) + 0.25 + 0.5 * bbyt) * mod1;
    }
}


void tiled_add_wm_a100_bestparam(size_t batchSize, float *S, uint8_t *wm, 
                                        size_t wmlen, size_t mod1, cudaStream_t stream){
    dim3 dimGrid = dim3(512);
    dim3 dimBlock = dim3(512);
    gpu_tiled_add_wm<<<dimGrid, dimBlock, 0, stream>>>(batchSize, S, wm, wmlen, mod1);
}

__global__ void gpu_tiled_get_wm(size_t batchSize, const float *S, float *wm, size_t wmlen, size_t mod1){

    size_t basecnt = (batchSize + wmlen - 1) / wmlen;
    size_t numextra = batchSize - batchSize % wmlen;

    for(size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < wmlen; tid += blockDim.x * gridDim.x){
        float acc = 0;
        for(int i = tid; i < batchSize; i += wmlen){
            float elm = S[i * TILE_DIM];
            float xiaoshu = elm - int(elm);
            bool bbyt = bool((int(elm) % mod1 + xiaoshu) > (mod1 / 2.));
            acc += bbyt;
        }
        if(tid < numextra){
            acc /= float(basecnt);
        } else {
            acc /= float(basecnt - 1);
        }
        wm[tid] = acc;
    }
}


void tiled_get_wm_a100_bestparam(size_t batchSize, const float *S, float *wm, 
                                        size_t wmlen, size_t mod1, cudaStream_t stream){
    dim3 dimGrid = dim3(512);
    dim3 dimBlock = dim3(512);
    gpu_tiled_get_wm<<<dimGrid, dimBlock, 0, stream>>>(batchSize, S, wm, wmlen, mod1);
}


