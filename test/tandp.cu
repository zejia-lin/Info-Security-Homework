
#include "../src/bwm.cu"

int main(){
    int N = 8;
    float *dA, *dRes;
    cudaMallocManaged(&dA, sizeof(float) * (N + 1) * N);
    cudaMallocManaged(&dRes, sizeof(float) * (N + 1) * N);
    for (size_t i = 0; i < N; ++i) {
        for(size_t j = 0; j < N; ++j){
            dA[i + j * N] = i + j * N;
        }
    }
    print_matrix_rowmaj(dA, N, N, N);
    cudaMemPrefetchAsync(dA, sizeof(float) * N * N, 0);
    cudaMemPrefetchAsync(dRes, sizeof(float) * N * N, 0);
    cudaDeviceSynchronize();
    mtxtp_a100_best_param(true, N, N, dA, N, dRes, N);
    cudaDeviceSynchronize();
    for(int i = 0; i < N * N; ++i){
        std::cout << dRes[i] << ", ";
        if((i + 1) % N == 0){
            std::cout << "\n";
        }
    }
    mtxtp_a100_best_param(false, N, N, dRes, N, dA, N);
    cudaDeviceSynchronize();
    for(int i = 0; i < N * N; ++i){
        std::cout << dA[i] << ", ";
        if((i + 1) % N == 0){
            std::cout << "\n";
        }
    }
}
