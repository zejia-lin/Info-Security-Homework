
#include "myutils.cpp"

void dct_cpu(float *A, float *res, int N){
    float tmp, alpha_u, alpha_v;
    for(int u = 0; u < N; ++u){
        for(int v = 0; v < N; ++v){
            tmp = 0;
            for(int x = 0; x < N; ++x){
                for(int y = 0; y < N; ++y){
                    tmp += A[IDX(x, y, N)] * cos((2 * x + 1) * u * M_PI / 2 / N)  
                                           * cos((2 * y + 1) * v * M_PI / 2 / N);
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

#define TILE_DIM 3

#define SQRT1 0.5773502691896257 // sqrt(1 / 3)
#define SQRT2 0.816496580927726  // sqrt(2 / 3)


__device__ void dct_tile(float *A, int lda, float *res, int u, int v){
    float tmp = 0;
    for(int x = 0; x < TILE_DIM; ++x){
        for(int y = 0; y < TILE_DIM; ++y){
            tmp += A[IDX(x, y, lda)] * cos((2 * x + 1) * u * M_PI / (2 * TILE_DIM))  
                                    * cos((2 * y + 1) * v * M_PI / 2 / (2 * TILE_DIM));
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
__global__ void dct_gpu(float *A, float *res, int rows, int cols){

    int tile_per_row = rows / TILE_DIM;

    // shared memory size equals to blockDim
    extern __shared__ float sA[];

    int tile_id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride_x = TILE_DIM * TILE_DIM * gridDim.x;
    
    // grid stride loop
    for(; tile_id < rows * cols - TILE_DIM * TILE_DIM; tile_id += stride_x){

        // compute the starting address of current tile in A
        int tile_x = tile_id / tile_per_row;
        int tile_y = tile_id % tile_per_row;
        int origin_ptr_offset = tile_x * TILE_DIM * TILE_DIM * tile_per_row + tile_y * TILE_DIM;
        float *tile_ptr_to_A = &A[origin_ptr_offset];
        
        // copy to shared memory
        sA[threadIdx.x * TILE_DIM * TILE_DIM + threadIdx.y * TILE_DIM + threadIdx.z] = 
                 tile_ptr_to_A[threadIdx.y * cols + threadIdx.z]; // note that leading dimension is cols
        __syncthreads();

        // compute the starting address of current tile in sA
        int smem_id = threadIdx.x;
        int smem_x = smem_id / tile_per_row;
        int smem_y = smem_id % tile_per_row;
        float *tile_ptr_to_shared = &sA[smem_x * TILE_DIM * TILE_DIM * tile_per_row + smem_y * TILE_DIM];
        float *ptr_to_res = &res[origin_ptr_offset];

        dct_tile(tile_ptr_to_shared, TILE_DIM, ptr_to_res, threadIdx.y, threadIdx.z);
        
        __syncthreads();
    }

}


int main() {

    int N = 3;
    float A[N * N], res[N * N];
    for (int i = 0; i < N * N; ++i) {
        A[i] = i;
    }

    dct_cpu(A, res, N);
    print_matrix(res, N, N);

    float *dA, *dRes;
    cudaMallocManaged(&dA, sizeof(float) * N * N);
    cudaMallocManaged(&dRes, sizeof(float) * N * N);
    for (int i = 0; i < N * N; ++i) {
        dA[i] = i;
    }

    dim3 dimGrid = dim3(1, 1);
    dim3 dimBlock = dim3(1, 3, 3);
    dct_gpu<<<dimGrid, dimBlock>>>(dA, dRes, N, N);
    cudaDeviceSynchronize();

    print_matrix(dRes, N, N);
    

}

#undef TILE_DIM
