
#include "myutils.cpp"


#define TILE_DIM 4

using DTYPE = float;
using ACC_TYPE = double;

#define SQRT1 0.5 // sqrt(1 / 4)
#define SQRT2 0.7071067811865475727373109293694142252206802368164062  // sqrt(2 / 4)

void dct_cpu(float *A, float *res, int N){
    float tmp, alpha_u, alpha_v;
    for(int u = 0; u < N; ++u){
        for(int v = 0; v < N; ++v){
            tmp = 0;
            for(int x = 0; x < N; ++x){
                for(int y = 0; y < N; ++y){
                    tmp += A[IDX(x, y, N)] * cos((2 * x + 1) * u * M_PI / (2 * N))  
                                           * cos((2 * y + 1) * v * M_PI / (2 * N));
                }
            }
            if(u == 0) alpha_u = sqrt(1. / N);
            else alpha_u = sqrt(2. / N);
            if(v == 0) alpha_v = sqrt(1. / N);
            else alpha_v = sqrt(2. / N);
            res[IDX(u, v, N)] = alpha_u * alpha_v * tmp;
        }
    }
}

void cpu_dct_tile(const float *A, int lda, float *res, int u, int v){
    float tmp = 0;
    for(int x = 0; x < TILE_DIM; ++x){
        for(int y = 0; y < TILE_DIM; ++y){
            tmp += A[IDX(x, y, lda)] * cos((2 * x + 1) * u * M_PI / (2 * TILE_DIM))  
                                     * cos((2 * y + 1) * v * M_PI / (2 * TILE_DIM));
        }
    }
    float alpha_u = SQRT2;
    float alpha_v = SQRT2;
    if(u == 0) alpha_u = SQRT1;
    if(v == 0) alpha_v = SQRT1;
    *res = alpha_u * alpha_v * tmp;
}

void dct_cpu_tiled(const float *A, float *res, int N){
    for(int u = 0; u < N; ++u){
        for(int v = 0; v < N; ++v){
            cpu_dct_tile(A, N, &res[IDX(u, v, N)], u, v);
        }
    }
}

void idct_cpu(float *A, float *res, int N){
    float tmp, alpha_u, alpha_v;
    for(int u = 0; u < N; ++u){
        for(int v = 0; v < N; ++v){
            tmp = 0;
            for(int x = 0; x < N; ++x){
                for(int y = 0; y < N; ++y){
                    tmp += A[IDX(x, y, N)] * cos((2 * x + 1) * u * M_PI / (2 * N))  
                                           * cos((2 * y + 1) * v * M_PI / (2 * N));
                }
            }
            if(u == 0) alpha_u = sqrt(1. / N);
            else alpha_u = sqrt(2. / N);
            if(v == 0) alpha_v = sqrt(1. / N);
            else alpha_v = sqrt(2. / N);
            res[IDX(u, v, N)] = alpha_u * alpha_v * tmp;
        }
    }
}


__constant__ ACC_TYPE COSINES[16] = {
    1.0, 0.9238795325112867384831361050601117312908172607421875, 0.7071067811865475727373109293694142252206802368164062, 0.3826834323650898372903839117498137056827545166015625,
    1.0, 0.3826834323650898372903839117498137056827545166015625, -0.7071067811865474617150084668537601828575134277343750, -0.9238795325112868495054385675757657736539840698242188,
    1.0, -0.3826834323650897262680814492341596633195877075195312, -0.7071067811865476837596133918850682675838470458984375, 0.9238795325112865164385311800288036465644836425781250,
    1.0, -0.9238795325112867384831361050601117312908172607421875, 0.7071067811865473506927060043381061404943466186523438, -0.3826834323650898928015351430076407268643379211425781
};

__constant__ ACC_TYPE ALPHAS[16] = {
    SQRT1, SQRT2, SQRT2, SQRT2, 
    SQRT2, SQRT2, SQRT2, SQRT2, 
    SQRT2, SQRT2, SQRT2, SQRT2, 
    SQRT2, SQRT2, SQRT2, SQRT2
};


__device__ __forceinline__ void dct_tile(const float *A, int lda, float *res, int u, int v){
    ACC_TYPE tmp = 0;
#pragma unroll
    for(int x = 0; x < TILE_DIM; ++x){
#pragma unroll
        for(int y = 0; y < TILE_DIM; ++y){
            tmp += A[IDX(x, y, lda)] * COSINES[IDX(x, u, TILE_DIM)] * COSINES[IDX(y, v, TILE_DIM)];
        }
    }
    *res = ALPHAS[u] * ALPHAS[v] * tmp;
}


/**
 * Should be launched with 3D block: (__, TILE_DIM, TILE_DIM) and 1D grid, shared mem size equals to blockDim
*/
__global__ void dct_gpu(int rows, int cols, const float *A, int lda, float *res, int ldres){

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
        sA[threadIdx.x * TILE_DIM * TILE_DIM + threadIdx.y * TILE_DIM + threadIdx.z] = 
                 tile_ptr_to_A[IDX(threadIdx.y, threadIdx.z, lda)]; // note that leading dimension is cols
        __syncthreads();

        // compute the starting address of current tile in sA
        float *tile_ptr_to_shared = &sA[threadIdx.x * TILE_DIM * TILE_DIM];
        float *elm_ptr_to_res = &res[tile_offset_to_A + threadIdx.y * ldres + threadIdx.z];
        // printf("(%d, %d, %d): %d\n", tile_id, threadIdx.y, threadIdx.z, tile_offset_to_A + threadIdx.y * ldres + threadIdx.z);

        dct_tile(tile_ptr_to_shared, TILE_DIM, elm_ptr_to_res, threadIdx.y, threadIdx.z);
    }
}

__device__ __forceinline__ void idct_tile(const float *A, int lda, float *res, int u, int v){
    ACC_TYPE tmp = 0;
#pragma unroll
    for(int x = 0; x < TILE_DIM; ++x){
#pragma unroll
        for(int y = 0; y < TILE_DIM; ++y){
            tmp += ALPHAS[x] * ALPHAS[y] * A[IDX(x, y, lda)] 
                             * COSINES[IDX(u, x, TILE_DIM)] * COSINES[IDX(v, y, TILE_DIM)];
        }
    }
    *res = tmp;
}


/**
 * Should be launched with 3D block: (__, TILE_DIM, TILE_DIM) and 1D grid, shared mem size equals to blockDim
*/
__global__ void idct_gpu(int rows, int cols, const float *A, int lda, float *res, int ldres){

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
        sA[threadIdx.x * TILE_DIM * TILE_DIM + threadIdx.y * TILE_DIM + threadIdx.z] = 
                 tile_ptr_to_A[IDX(threadIdx.y, threadIdx.z, lda)]; // note that leading dimension is cols
        __syncthreads();

        // compute the starting address of current tile in sA
        float *tile_ptr_to_shared = &sA[threadIdx.x * TILE_DIM * TILE_DIM];
        float *elm_ptr_to_res = &res[tile_offset_to_A + threadIdx.y * ldres + threadIdx.z];
        // printf("(%d, %d, %d): %d\n", tile_id, threadIdx.y, threadIdx.z, tile_offset_to_A + threadIdx.y * ldres + threadIdx.z);

        idct_tile(tile_ptr_to_shared, TILE_DIM, elm_ptr_to_res, threadIdx.y, threadIdx.z);
    }

}


void dct_a100_best_param(int rows, int cols, const float *A, int lda, float *res, int ldres, cudaStream_t stream=0){
    dim3 dimGrid = dim3(512);
    dim3 dimBlock = dim3(8, TILE_DIM, TILE_DIM);
    size_t smemSize = dimBlock.x * dimBlock.y * dimBlock.z * sizeof(float);
    dct_gpu<<<dimGrid, dimBlock, smemSize, stream>>>(rows, cols, A, lda, res, ldres);
}

void idct_a100_best_param(int rows, int cols, const float *A, int lda, float *res, int ldres, cudaStream_t stream=0){
    dim3 dimGrid = dim3(512);
    dim3 dimBlock = dim3(8, TILE_DIM, TILE_DIM);
    size_t smemSize = dimBlock.x * dimBlock.y * dimBlock.z * sizeof(float);
    idct_gpu<<<dimGrid, dimBlock, smemSize, stream>>>(rows, cols, A, lda, res, ldres);
}


int main(int argc, char **argv) {

    uint64_t compute_time, prefetch_time;
    size_t N = atoll(argv[1]);
    // float A[N * N];//, res[N * N];
    // for (size_t i = 0; i < N * N; ++i) {
    //     std::cout << i << ',';
    //     A[i] = i;
    // }
    // cpu_dct_tile(A, N, res, 0, 1);
    // std::cout << "(0, 1): " << *res << std::endl;
    // dct_cpu_tiled(A, res, N);
    // print_matrix(res, N, N);
    // writebin("./out/cpu_9.bin", res, sizeof(float) * N * N);

    float *dA, *dRes;
    cudaMallocManaged(&dA, sizeof(float) * (N + 1) * N);
    cudaMallocManaged(&dRes, sizeof(float) * (N + 1) * N);
    for (size_t i = 0; i < N; ++i) {
        for(size_t j = 0; j < N; ++j){
            dA[i + j * (N + 1)] = i * N + j;
        }
    }

    // print_matrix_rowmaj(dA, N + 1, N, N + 1);

    __TIMER_START__
    cudaMemPrefetchAsync(dA, sizeof(float) * N * N, 0);
    cudaDeviceSynchronize();
    __TIMER_STOP__(prefetch_time)
    std::cout << "Prefetch in "<< prefetch_time / 1000 << " ms\n";
    
    // ====================================================================================================
    // ================================================DCT================================================
    // ====================================================================================================
    for(int _iter = 0; _iter < 1; ++_iter){
        __TIMER_START__
        dct_a100_best_param(N, N, dA, N + 1, dRes, N + 1);
        cudaDeviceSynchronize();
        __TIMER_STOP__(compute_time);
        auto err = cudaGetLastError();
        if(err != cudaSuccess){
            std::cout << cudaGetErrorString(err) << std::endl;
        }
        std::cout << "DCT GPU time " << double(compute_time) / 1000. << " ms\n";
    }

    writebin("./out/gpu_dct.bin", dRes, sizeof(float) * (N + 1) * N);

    // =====================================================================================================
    // ================================================IDCT================================================
    // =====================================================================================================
    for(int _iter = 0; _iter < 1; ++_iter){
        __TIMER_START__
        idct_a100_best_param(N, N, dRes, N + 1, dA, N + 1);
        cudaDeviceSynchronize();
        __TIMER_STOP__(compute_time);
        auto err = cudaGetLastError();
        if(err != cudaSuccess){
            std::cout << cudaGetErrorString(err) << std::endl;
        }
        std::cout << "IDCT GPU time " << double(compute_time) / 1000. << " ms\n";
    }
    writebin("./out/gpu_idct.bin", dA, sizeof(float) * (N + 1) * N);

    

}

#undef TILE_DIM
