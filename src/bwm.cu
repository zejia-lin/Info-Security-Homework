
#include <cuda_runtime.h>
#include <cublas.h>
#include <cusolverDn.h>

#include <iostream>
#include "myutils.cpp"
#include "mydct.cu"
#include "cusolver_utils.cpp"
#include "constants.h"

struct myhandle {

};


void gesvd(size_t rows, size_t cols, float *A, size_t lda, float *S, float *U, size_t ldu, float *V, size_t ldv, cudaStream_t stream=0){

    cusolverDnHandle_t cusolverHandle;
    gesvdjInfo_t gesvdinfo;
    int lwork;
    float *work;
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


int main(){

}
