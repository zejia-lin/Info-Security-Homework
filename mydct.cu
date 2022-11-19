
#include "myutils.cpp"

// void DCT() {
//   freopen("dct.raw", "wb", stdout);
//   int r, c, i, j, x, y;
//   for (r = 0; r < 64; r++)
//     for (c = 0; c < 64; c++)
//       for (i = 0; i < 8; i++)
//         for (j = 0; j < 8; j++) {
//           double sum = 0;
//           for (x = 0; x < 8; x++)
//             for (y = 0; y < 8; y++)
//               sum += (pic[r * 8 + x][c * 8 + y] - 128) * COS[x][i] * COS[y][j];
//           sum *= C[i] * C[j] * 0.25;
//           dct[r * 8 + i][c * 8 + j] = sum;
//       }
//   for (r = 0; r < N; r++)
//     for (c = 0; c < N; c++)
//       putchar(dct[r][c]);
// }

#ifndef TILE_DIM
#define TILE_DIM 3
#endif

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


__device__ void dct_cell(float *A, float *res){

}


/**
 * Should be launched with 3D block: (__, 3, 3) and 1D grid
*/
__global__ void dct_gpu(float *A, float *res, int rows, int cols){

    float tmp, alpha_u, alpha_v;

    int tx = threadIdx.x + blockIdx.x;
    int ty = threadIdx.y + blockIdx.y;
    int tz = threadIdx.z + blockIdx.z;

    int stride_x = TILE_DIM * TILE_DIM * gridDim.x;
    
    // grid stride loop
    for(; tx < rows * cols - TILE_DIM * TILE_DIM; tx += stride_x){

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
}

#undef TILE_DIM
