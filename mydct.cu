
#include "myutils.cpp"


#define TILE_DIM 3

#define SQRT1 0.5773502691896257 // sqrt(1 / 3)
#define SQRT2 0.816496580927726  // sqrt(2 / 3)

// 𝐹(𝑢,𝑣)=𝑐(𝑢)𝑐(𝑣)∑𝑖=0𝑁−1∑𝑗=0𝑁−1𝑓(𝑖,𝑗)𝑐𝑜𝑠[(2𝑖+1)𝜋2𝑁𝑢]𝑐𝑜𝑠[(2𝑗+1)𝜋2𝑁𝑣]
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


__constant__ float COSINES[81] = {
    1.0, 0.8660254037844387, 0.5000000000000001, 6.123233995736766e-17, -0.4999999999999998, -0.8660254037844387, -1.0, -0.8660254037844388, -0.5000000000000004,
    1.0, 6.123233995736766e-17, -1.0, -1.8369701987210297e-16, 1.0, 1.1943401194869635e-15, -1.0, -4.286263797015736e-16, 1.0,
    1.0, -0.8660254037844387, 0.5000000000000001, 1.1943401194869635e-15, -0.49999999999999983, 0.8660254037844383, -1.0, 0.8660254037844386, -0.5000000000000003,
    1.0, -0.8660254037844388, 0.5000000000000006, -4.286263797015736e-16, -0.499999999999999, 0.8660254037844386, -1.0, 0.8660254037844395, -0.500000000000002,
    1.0, -1.8369701987210297e-16, -1.0, 5.51091059616309e-16, 1.0, 8.578717400397356e-16, -1.0, -4.904777002955296e-16, 1.0,
    1.0, 0.8660254037844384, 0.4999999999999991, 1.1028010998692062e-15, -0.5000000000000018, -0.8660254037844377, -1.0, -0.8660254037844382, -0.4999999999999964,
    1.0, 0.8660254037844386, 0.49999999999999994, 2.57237725884603e-15, -0.5000000000000001, -0.8660254037844383, -1.0, -0.8660254037844398, -0.4999999999999998,
    1.0, 1.1943401194869635e-15, -1.0, 8.578717400397356e-16, 1.0, -2.45548340466059e-16, -1.0, 3.185938619692883e-15, 1.0,
    1.0, -0.8660254037844388, 0.5000000000000004, 2.8173066186755008e-15, -0.49999999999999917, 0.866025403784438, -1.0, 0.8660254037844359, -0.5000000000000017
};

__constant__ float ALPHAS[9] = {
    SQRT1, SQRT2, SQRT2, SQRT2, SQRT2, SQRT2, SQRT2, SQRT2, SQRT2
};


__device__ __forceinline__ void dct_tile(const float *A, int lda, float *res, int u, int v){
    float tmp = 0;
#pragma unroll
    for(int x = 0; x < TILE_DIM; ++x){
#pragma unroll
        for(int y = 0; y < TILE_DIM; ++y){
            tmp += A[IDX(x, y, lda)] * COSINES[IDX(x, u, TILE_DIM * TILE_DIM)] * COSINES[IDX(y, v, TILE_DIM * TILE_DIM)];
        }
    }
    *res = ALPHAS[u] * ALPHAS[v] * tmp;
}


/**
 * Should be launched with 3D block: (__, 3, 3) and 1D grid, shared mem size equals to blockDim
*/
__global__ void dct_gpu(const float *A, float *res, int rows, int cols){

    // shared memory size equals to blockDim
    extern __shared__ float sA[];

    int tile_id = threadIdx.x + blockIdx.x * blockDim.x;
    int tile_per_row = rows / TILE_DIM;
    int num_tiles = (rows / TILE_DIM) * (cols / TILE_DIM);
    
    // grid stride loop
#pragma unroll
    for(; tile_id < num_tiles; tile_id += gridDim.x){

        // compute the starting address of current tile in A
        int tile_x = tile_id / tile_per_row;
        int tile_y = tile_id % tile_per_row;
        int tile_offset_to_A = tile_x * TILE_DIM * TILE_DIM * tile_per_row + tile_y * TILE_DIM;
        const float *tile_ptr_to_A = &A[tile_offset_to_A];
        
        // copy to shared memory
        sA[threadIdx.x * TILE_DIM * TILE_DIM + threadIdx.y * TILE_DIM + threadIdx.z] = 
                 tile_ptr_to_A[threadIdx.y * cols + threadIdx.z]; // note that leading dimension is cols
        __syncthreads();

        // compute the starting address of current tile in sA
        float *tile_ptr_to_shared = &sA[threadIdx.x * TILE_DIM * TILE_DIM];
        float *elm_ptr_to_res = &res[tile_offset_to_A + threadIdx.y * cols + threadIdx.z];

        dct_tile(tile_ptr_to_shared, TILE_DIM, elm_ptr_to_res, threadIdx.y, threadIdx.z);
    }

}

__device__ __forceinline__ void idct_tile(const float *A, int lda, float *res, int u, int v){
    float tmp = 0;
#pragma unroll
    for(int x = 0; x < TILE_DIM; ++x){
#pragma unroll
        for(int y = 0; y < TILE_DIM; ++y){
            tmp += ALPHAS[x] * ALPHAS[y] * A[IDX(x, y, lda)] * COSINES[IDX(u, x, TILE_DIM * TILE_DIM)] * COSINES[IDX(v, y, TILE_DIM * TILE_DIM)];
        }
    }
    *res = tmp;
}


/**
 * Should be launched with 3D block: (__, 3, 3) and 1D grid, shared mem size equals to blockDim
*/
__global__ void idct_gpu(const float *A, float *res, int rows, int cols){

    // shared memory size equals to blockDim
    extern __shared__ float sA[];

    int tile_id = threadIdx.x + blockIdx.x * blockDim.x;
    int tile_per_row = rows / TILE_DIM;
    int num_tiles = (rows / TILE_DIM) * (cols / TILE_DIM);
    
    // grid stride loop
#pragma unroll
    for(; tile_id < num_tiles; tile_id += gridDim.x){

        // compute the starting address of current tile in A
        int tile_x = tile_id / tile_per_row;
        int tile_y = tile_id % tile_per_row;
        int tile_offset_to_A = tile_x * TILE_DIM * TILE_DIM * tile_per_row + tile_y * TILE_DIM;
        const float *tile_ptr_to_A = &A[tile_offset_to_A];
        
        // copy to shared memory
        sA[threadIdx.x * TILE_DIM * TILE_DIM + threadIdx.y * TILE_DIM + threadIdx.z] = 
                 tile_ptr_to_A[threadIdx.y * cols + threadIdx.z]; // note that leading dimension is cols
        __syncthreads();

        // compute the starting address of current tile in sA
        float *tile_ptr_to_shared = &sA[threadIdx.x * TILE_DIM * TILE_DIM];
        float *elm_ptr_to_res = &res[tile_offset_to_A + threadIdx.y * cols + threadIdx.z];

        idct_tile(tile_ptr_to_shared, TILE_DIM, elm_ptr_to_res, threadIdx.y, threadIdx.z);
    }

}


int main(int argc, char **argv) {

    uint64_t compute_time;
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
    cudaMallocManaged(&dA, sizeof(float) * N * N);
    cudaMallocManaged(&dRes, sizeof(float) * N * N);
    for (size_t i = 0; i < N * N; ++i) {
        dA[i] = i;
    }

    cudaMemcpy(dA, dA, sizeof(float) * N * N, cudaMemcpyDefault);
    cudaDeviceSynchronize();

    dim3 dimGrid = dim3(512);
    dim3 dimBlock = dim3(8, TILE_DIM, TILE_DIM);
    size_t smemSize = dimBlock.x * dimBlock.y * dimBlock.z * sizeof(float);

    // ====================================================================================================
    // ================================================DCT================================================
    // ====================================================================================================
    for(int _iter = 0; _iter < 1; ++_iter){
        __TIMER_START__
        dct_gpu<<<dimGrid, dimBlock, smemSize>>>(dA, dRes, N, N);
        cudaDeviceSynchronize();
        __TIMER_STOP__(compute_time);
        auto err = cudaGetLastError();
        if(err != cudaSuccess){
            std::cout << cudaGetErrorString(err) << std::endl;
        }
        std::cout << "DCT GPU time " << double(compute_time) / 1000. << " ms\n";
    }
    writebin("./out/gpu_dct.bin", dRes, sizeof(float) * N * N);

    // =====================================================================================================
    // ================================================IDCT================================================
    // =====================================================================================================
    for(int _iter = 0; _iter < 1; ++_iter){
        __TIMER_START__
        idct_gpu<<<dimGrid, dimBlock, smemSize>>>(dRes, dA, N, N);
        cudaDeviceSynchronize();
        __TIMER_STOP__(compute_time);
        auto err = cudaGetLastError();
        if(err != cudaSuccess){
            std::cout << cudaGetErrorString(err) << std::endl;
        }
        std::cout << "IDCT GPU time " << double(compute_time) / 1000. << " ms\n";
    }
    writebin("./out/gpu_idct.bin", dA, sizeof(float) * N * N);

    

}

#undef TILE_DIM
