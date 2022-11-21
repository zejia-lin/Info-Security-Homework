
#include <cuda_runtime.h>
#include <cublas.h>
#include <cusolverDn.h>

#include <iostream>
#include "svd_3x3.cu"

#define IDX(i, j, ld) (((i) * (ld)) + j)


/**
 * A = U * S * VT, the VT is transposed, all matrix has **continous** address space
 * 
 * @param A dense input
 * @param lda leading dimension of A
 * @param U dense matrix U
 * @param S dense vector V
 * @param VT dense matrix VT
*/
__device__ __forceinline__
void svd_continous(float *A, float *U, float *S, float *VT){
    svd(A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8],
        U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7], U[8],
        S[0], S[1], S[2],
        VT[0], VT[3], VT[6], VT[1], VT[4], VT[7], VT[2], VT[5], VT[8]);
}

/**
 * @param A CWH stored, row major
 * @param rows num of rows
 * @param cols num of cols
 * @param lda leading dimension of A
*/
__global__ void embed_gpu(float *A, size_t rows, size_t cols, size_t lda, float *U, float *S, float *VT){

    const int TILE_DIM = 3;
    int i_stride = TILE_DIM * blockDim.x * gridDim.x;
    int j_stride = TILE_DIM * blockDim.y * gridDim.y;

    __shared__ float sA[rows * cols];

    for(int cid = 0; cid < 3; ++ cid){
        float *channel = A + cid * rows * cols;
        for(int ti = threadIdx.x + blockIdx.x * blockDim.x; ti < rows / TILE_DIM; ti += i_stride){
            for(int tj = threadIdx.y + blockIdx.y * blockDim.y; tj < cols / TILE_DIM; tj += j_stride){

                // start address of the tile
                int i_tile = ti * TILE_DIM;
                int j_tile = tj * TILE_DIM;
                
                // copy to shared memory, remap address so each tile has **continous** address space
                sA[IDX(i_tile, j_tile, lda) + 0] = channel[IDX(i_tile + 0, j_tile + 0, lda)];
                sA[IDX(i_tile, j_tile, lda) + 1] = channel[IDX(i_tile + 0, j_tile + 1, lda)];
                sA[IDX(i_tile, j_tile, lda) + 2] = channel[IDX(i_tile + 0, j_tile + 2, lda)];
                sA[IDX(i_tile, j_tile, lda) + 3] = channel[IDX(i_tile + 1, j_tile + 0, lda)];
                sA[IDX(i_tile, j_tile, lda) + 4] = channel[IDX(i_tile + 1, j_tile + 1, lda)];
                sA[IDX(i_tile, j_tile, lda) + 5] = channel[IDX(i_tile + 1, j_tile + 2, lda)];
                sA[IDX(i_tile, j_tile, lda) + 6] = channel[IDX(i_tile + 2, j_tile + 0, lda)];
                sA[IDX(i_tile, j_tile, lda) + 7] = channel[IDX(i_tile + 2, j_tile + 1, lda)];
                sA[IDX(i_tile, j_tile, lda) + 8] = channel[IDX(i_tile + 2, j_tile + 2, lda)];

                // addr of the embedding 3x3 slice
                float *tile = &sA[IDX(i_tile, j_tile, lda)]; 
                float *tileU = &U[IDX(i_tile, j_tile, lda)];
                float *tileVT = &VT[IDX(i_tile, j_tile, lda)];
                float *tileS = &S[IDX(ti, j_tile, cols)]; // note that S is vector
                svd_continous(tile, tileU, tileS, tileVT);
                
            }
        }
    }
    
}


