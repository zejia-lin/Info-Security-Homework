
#include "../src/dwt.cu"


int main(){

    size_t rows = 4898;
    size_t cols = 6660;
    size_t filesize;
    float *cpu, *gpu, *coefs[4];
    float *out;

    cpu = (float *)malloc(sizeof(float) * rows * cols);
    cudaMallocManaged(&gpu, sizeof(float) * rows * cols);
    cudaMallocManaged(&out, sizeof(float) * rows * cols);
    cudaMallocManaged(&coefs[0], sizeof(float) * rows * cols);
    cudaMallocManaged(&coefs[1], sizeof(float) * rows * cols);
    cudaMallocManaged(&coefs[2], sizeof(float) * rows * cols);
    cudaMallocManaged(&coefs[3], sizeof(float) * rows * cols);

    readbin("../out/5.bin", &filesize, gpu, rows * cols * sizeof(float));
    readbin("../out/5.bin", &filesize, cpu, rows * cols * sizeof(float));

    cudaMemPrefetchAsync(gpu, sizeof(float) * rows * cols, 0);
    cudaMemPrefetchAsync(out, sizeof(float) * rows * cols, 0);
    cudaMemPrefetchAsync(coefs[0], sizeof(float) * rows * cols, 0);
    cudaMemPrefetchAsync(coefs[1], sizeof(float) * rows * cols, 0);
    cudaMemPrefetchAsync(coefs[2], sizeof(float) * rows * cols, 0);
    cudaMemPrefetchAsync(coefs[3], sizeof(float) * rows * cols, 0);
    
    __TIMER_START__(computation);
    haar_forward2d(gpu, coefs, rows, cols);
    haar_inverse2d(gpu, coefs, rows, cols);
    cudaDeviceSynchronize();
    __TIMER_STOP__(computation);
    
    std::cout << "Computation " << computation << " ms\n";

    writebin("../out/gpuhar.bin", gpu, sizeof(float) * rows * cols);

    // print_matrix_rowmaj(gpu, rows, cols, cols);

    // print_matrix_rowmaj(coefs[0], rows / 2, cols / 2, cols / 2);
    // print_matrix_rowmaj(coefs[1], rows / 2, cols / 2, cols / 2);
    // print_matrix_rowmaj(coefs[2], rows / 2, cols / 2, cols / 2);
    // print_matrix_rowmaj(coefs[3], rows / 2, cols / 2, cols / 2);

    

    std::cout << "Finished " << cudaGetErrorString(cudaGetLastError()) << std::endl;
}

