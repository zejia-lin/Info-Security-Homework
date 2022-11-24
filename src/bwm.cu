
#include <cuda_runtime.h>
#include <cublas.h>
#include <cusolverDn.h>

#include <iostream>
#include "myutils.cpp"
#include "mmd.cu"
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

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Using device " << device << " " << prop.name << std::endl;

    // int N = atoi(argv[1]);
    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);

    float *A, *AT, *U, *S, *V;
    float *pyU, *pyV, *inv;
    int *info;
    int lda = rows;
    // int ldu = rows;
    // int ldv = rows;
    int lda_T = cols;
    // int ldu_T = cols;
    // int ldv_T = cols;

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
    CUDA_CHECK(cudaMallocManaged(&AT, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&A, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&U, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&pyU, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&V, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&pyV, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&inv, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&S, sizeof(float) * numTiles * TILE_DIM));

    int bb = myreadbin("../out/A.bin", A);

    CUDA_CHECK(cudaMemPrefetchAsync(info, sizeof(int) * batchSize, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(AT, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(A, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(U, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(pyU, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(V, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(pyV, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(S, sizeof(float) * numTiles * TILE_DIM, device, stream));

    CUSOLVER_CHECK(cusolverDnCreate(&solverHandle));
    CUSOLVER_CHECK(cusolverDnSetStream(solverHandle, stream));
    CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdParams));
    CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(gesvdParams, 1e-5));
    CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdParams, 1000));
    CUBLAS_CHECK(cublasCreate(&blasHandle));
    CUBLAS_CHECK(cublasSetStream(blasHandle, stream));

    // mtxtp_a100_best_param(true, rows, cols, AT, lda_T, A, lda, stream);
    // CUDA_CHECK(cudaDeviceSynchronize());
    
    for(int tile_id = 0; tile_id < (cols / TILE_DIM) * (rows / TILE_DIM); ++tile_id){
        for(int i = 0; i < TILE_DIM; ++i){
            for(int j = 0; j < TILE_DIM; ++j){
                std::cout << A[i + j * TILE_DIM + tile_id * TILE_DIM * TILE_DIM] << ", ";
            }
            std::cout << "\n";
        }
        std::cout << "===================\n";
    }
    // exit(0);

    // CUDA_CHECK(cudaMemcpy(A, AT, sizeof(float) * rows * cols, cudaMemcpyDefault));
    // CUBLAS_CHECK(cublasSgeam(blasHandle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, &one, AT, lda_T, &zero, A, lda, A, lda));

    for(int i = 0; i < rows * cols; ++i){
        std::cout << A[i] << ", ";
        if((i + 1) % cols == 0){
            std::cout << "\n";
        }
    }

    CUSOLVER_CHECK(cusolverDnSgesvdjBatched_bufferSize(solverHandle, 
                                 CUSOLVER_EIG_MODE_VECTOR,
                                 TILE_DIM, TILE_DIM, 
                                 A, TILE_DIM, S, U, TILE_DIM, V, TILE_DIM,
                                 &lwork, gesvdParams, batchSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&work), sizeof(float) * lwork));

    CUSOLVER_CHECK(cusolverDnSgesvdjBatched(solverHandle, CUSOLVER_EIG_MODE_VECTOR, 
                TILE_DIM, TILE_DIM, 
                A, TILE_DIM, S, U, TILE_DIM, V, TILE_DIM,
                work, lwork, info, gesvdParams, batchSize));
    CUDA_CHECK(cudaDeviceSynchronize());

    for(int i = 0; i < batchSize; ++i){
        if (0 == info[i]) {
            std::printf("matrix %d: gesvdj converges \n", i);
        } else if (0 > info[i]) {
            /* only info[0] shows if some input parameter is wrong.
             * If so, the error is CUSOLVER_STATUS_INVALID_VALUE.
             */
            std::printf("Error: %d-th parameter is wrong \n", -info[i]);
            exit(1);
        } else { /* info = m+1 */
                 /* if info[i] is not zero, Jacobi method does not converge at i-th matrix. */
            std::printf("WARNING: matrix %d, info = %d : gesvdj does not converge \n", i, info[i]);
        }
    }

    std::cout << "U\n";
    print_matrix_colmaj(U, 4, 4, 4);
    std::cout << "V\n";
    print_matrix_colmaj(V, 4, 4, 4);
    std::cout << "S\n";
    print_matrix_rowmaj(S, 1, 4, 4);

    mmd_batched_a100_best_param(false, U, S, inv, batchSize);
    cublasGemmStridedBatchedEx(
        blasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
        TILE_DIM, TILE_DIM, TILE_DIM,
        &one,
        inv, CUDA_R_32F, TILE_DIM, TILE_DIM * TILE_DIM,
        V, CUDA_R_32F, TILE_DIM, TILE_DIM * TILE_DIM,
        &zero,
        inv, CUDA_R_32F, TILE_DIM, TILE_DIM * TILE_DIM,
        batchSize, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );

    cudaDeviceSynchronize();


    std::cout << "====================\nGemm from GPU\n";
    print_matrix_rowmaj(inv, 4, 4, 4);

    // print_matrix_rowmaj(pyU, 4, 4, 4);
    // print_matrix_rowmaj(pyV, 4, 4, 4);
    // print_matrix_rowmaj(S, 1, 4, 4);

    mtxtp_a100_best_param(false, rows, cols, U, lda, pyU, lda, stream);
    mtxtp_a100_best_param(false, rows, cols, V, lda, pyV, lda, stream);
    cudaDeviceSynchronize();

    // CUBLAS_CHECK(cublasSgeam(blasHandle, CUBLAS_OP_T, CUBLAS_OP_N, cols, rows, &one, U, ldu, &zero, pyU, ldu_T, pyU, ldu_T));
    // CUBLAS_CHECK(cublasSgeam(blasHandle, CUBLAS_OP_T, CUBLAS_OP_N, cols, rows, &one, V, ldv, &zero, pyV, ldv_T, pyV, ldv_T));
    // CUDA_CHECK(cudaDeviceSynchronize());

    // std::cout << "======================\nU\n";
    // print_matrix_rowmaj(V, rows, cols, lda);
    // std::cout << "======================\npyU\n";
    // print_matrix_rowmaj(pyV, rows, cols, lda);

    writebin("../out/U.bin", U, sizeof(float) * rows * cols);
    writebin("../out/V.bin", V, sizeof(float) * rows * cols);
    writebin("../out/S.bin", S, sizeof(float) * numTiles * TILE_DIM);

    // print_matrix_colmaj(A, rows, cols, lda);
    // print_matrix_colmaj(U, rows, cols, lda);
    // print_matrix_colmaj(V, rows, cols, lda);
    // print_matrix_rowmaj(S, 1, N, lda);

    std::cout << "Exit bwm with 0\n";

}
