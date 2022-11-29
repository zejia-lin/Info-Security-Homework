
#include "../src/svd.cu"


int main(int argc, char **argv){

    __TIMER_START__(end2end);

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Using device " << device << " " << prop.name << std::endl;

    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    int wmlen = atoi(argv[3]);

    float *A, *U, *S, *V, *inv;
    int mod1 = 10;
    uint8_t *wm, *wmget;
    int *info;

    cudaStream_t stream = NULL;
    cublasHandle_t blasHandle;
    cusolverDnHandle_t solverHandle;
    gesvdjInfo_t gesvdParams;
    int lwork;
    float *work;
    size_t numTiles = (rows / TILE_DIM) * (cols / TILE_DIM);

    CUDA_CHECK(cudaMallocManaged(&wm, sizeof(uint8_t) * wmlen));
    CUDA_CHECK(cudaMallocManaged(&wmget, sizeof(uint8_t) * wmlen));
    CUDA_CHECK(cudaMallocManaged(&info, sizeof(int) * numTiles));
    CUDA_CHECK(cudaMallocManaged(&A, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&U, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&V, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&inv, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&S, sizeof(float) * numTiles * TILE_DIM));

    int bb = myreadbin("../out/A.bin", A);
    bb = myreadbin("../out/wm.bin", wm);
    std::cout << "Read watermark\n";
    for(int i = 0; i < wmlen; ++i){
        std::cout << int(wm[i]) << ", ";
    }
    std::cout << "\n";

    CUDA_CHECK(cudaMemPrefetchAsync(wm, sizeof(uint8_t) * wmlen, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(wmget, sizeof(uint8_t) * wmlen, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(info, sizeof(int) * numTiles, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(A, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(U, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(V, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(S, sizeof(float) * numTiles * TILE_DIM, device, stream));

    init_cudalib(&solverHandle, &blasHandle, numTiles, A, U, S, V, &work, &lwork, &gesvdParams, stream);
    std::cout << "Allocated " << lwork << " float buffer for gesvd\n";

    __TIMER_START__(computation);
    gesvd_a100_best_param(solverHandle, numTiles, A, U, S, V, work, lwork, info, gesvdParams);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Before add wm\n";
    print_matrix_rowmaj(S, 2, TILE_DIM, TILE_DIM);
    tiled_add_wm_a100_bestparam(numTiles, S, wm, wmlen, mod1, stream);
    cudaDeviceSynchronize();
    std::cout << "After add wm\n";
    print_matrix_rowmaj(S, 2, TILE_DIM, TILE_DIM);
    tiled_get_wm_a100_bestparam(numTiles, S, wmget, wmlen, mod1, stream);

    mmd_batched_a100_best_param(false, U, S, inv, numTiles);
    invsvd_a100_best_param(blasHandle, numTiles, inv, U, S, V);
    cudaDeviceSynchronize();
    __TIMER_STOP__(computation);


    for(int i = 0; i < numTiles; ++i){
        if (0 == info[i]) {
            // std::printf("matrix %d: gesvdj converges \n", i);
        } else if (0 > info[i]) {
            std::printf("Error: %d-th parameter is wrong \n", -info[i]);
            exit(1);
        } else {
            std::printf("WARNING: matrix %d, info = %d : gesvdj does not converge \n", i, info[i]);
        }
    }

    // std::cout << "====================\nGemm from GPU\n";
    // print_matrix_rowmaj(inv, 8, 8, 8);

    writebin("../out/inv.bin", inv, sizeof(float) * rows * cols);
    writebin("../out/wmget.bin", wmget, sizeof(uint8_t) * wmlen);

    // for(int i = 0; i < wmlen; ++i){
    //     std::cout << int(wmget[i]) << ", ";
    // }
    // std::cout << "\n";

    std::cout << "GPU computation " << computation << " ms\n";
    __TIMER_STOP__(end2end);
    std::cout << "GPU end to end " << end2end << " ms\n";

    std::cout << "Exit bwm with 0\n";

}

