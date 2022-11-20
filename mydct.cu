
#include "myutils.cpp"


#define TILE_DIM 3

#define SQRT1 0.5773502691896257 // sqrt(1 / 3)
#define SQRT2 0.816496580927726  // sqrt(2 / 3)


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


__device__ void dct_tile(const float *A, int lda, float *res, int u, int v){
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


/**
 * Should be launched with 3D block: (__, 3, 3) and 1D grid, shared mem size equals to blockDim
*/
__global__ void dct_gpu(const float *A, float *res, int rows, int cols){

    // shared memory size equals to blockDim
    extern __shared__ float sA[];

    int tile_id = threadIdx.x + blockIdx.x * blockDim.x;
    int tile_per_row = rows / TILE_DIM;
    
    // grid stride loop
    for(; tile_id < (rows * cols) / (TILE_DIM * TILE_DIM); tile_id += gridDim.x){

        // compute the starting address of current tile in A
        int tile_x = tile_id / tile_per_row;
        int tile_y = tile_id % tile_per_row;
        int tile_offset_to_A = tile_x * TILE_DIM * TILE_DIM * tile_per_row + tile_y * TILE_DIM;
        const float *tile_ptr_to_A = &A[tile_offset_to_A];
        
        // copy to shared memory
        sA[threadIdx.x * TILE_DIM * TILE_DIM + threadIdx.y * TILE_DIM + threadIdx.z] = 
                 tile_ptr_to_A[threadIdx.y * cols + threadIdx.z]; // note that leading dimension is cols
        __syncthreads();
        // printf("(%d, %d, %d): %f\n", tile_id, threadIdx.y, threadIdx.z, tile_ptr_to_A[threadIdx.y * cols + threadIdx.z]);

        // compute the starting address of current tile in sA
        // int smem_id = threadIdx.x;
        // int smem_x = smem_id / tile_per_row;
        // int smem_y = smem_id % tile_per_row;
        // float *tile_ptr_to_shared = &sA[smem_x * TILE_DIM * TILE_DIM * tile_per_row + smem_y * TILE_DIM];
        float *elm_ptr_to_res = &res[tile_offset_to_A + threadIdx.y * cols + threadIdx.z];

        dct_tile(tile_ptr_to_A, cols, elm_ptr_to_res, threadIdx.y, threadIdx.z);
        __syncthreads();

        // printf("(%d, %d, %d): %d, %f\n", tile_id, threadIdx.y, threadIdx.z, 
        //             tile_offset_to_A + threadIdx.y * cols + threadIdx.z, *elm_ptr_to_res);
    }

}


int main(int argc, char **argv) {

    uint64_t compute_time;
    size_t N = atoll(argv[1]);
    std::cout << "Read " << N << std::endl;
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

    std::cout << "Created matrix\n";

    dim3 dimGrid = dim3(128);
    dim3 dimBlock = dim3(64, TILE_DIM, TILE_DIM);
    int smemSize = dimBlock.x * dimBlock.y * dimBlock.z * sizeof(float);

    for(int _iter = 0; _iter < 1; ++_iter){
        __TIMER_START__
        dct_gpu<<<dimGrid, dimBlock, smemSize>>>(dA, dRes, N, N);
        cudaDeviceSynchronize();
        __TIMER_STOP__(compute_time);
        std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
        std::cout << "GPU time " << double(compute_time) / 1000. << " ms\n";
    }



    // print_matrix(dRes, N, N);

    writebin("./out/gpu_9.bin", dRes, sizeof(float) * N * N);

}

#undef TILE_DIM
