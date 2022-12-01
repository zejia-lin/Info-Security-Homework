
#include <opencv2/opencv.hpp>

#include "../src/svd.cu"
#include "../src/mydct.cu"
#include "../src/dwt.cu"


int main(int argc, char **argv){

    __TIMER_START__(end2end);

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Using device " << device << " " << prop.name << std::endl;

    size_t orirows = atoll(argv[1]);
    size_t oricols = atoll(argv[2]);
    size_t wmlen = atoll(argv[3]);

    size_t rows = 16 * (orirows / 32);
    size_t cols = 16 * (oricols / 32);

    printf("Ori: (%d, %d), Rnd:(%d, %d)\n", orirows, oricols, rows, cols);

    float *U, *S, *V, *inv, *dct, *wmget;
    float *Coefs, *Img;
    float *coefs[4];
    int mod1 = 37, mod2 = 11;
    uint8_t *wm;
    int *info;

    cudaStream_t stream = NULL;
    cublasHandle_t blasHandle;
    cusolverDnHandle_t solverHandle;
    gesvdjInfo_t gesvdParams;
    int lwork;
    float *work;
    size_t numTiles = (rows / TILE_DIM) * (cols / TILE_DIM);

    CUDA_CHECK(cudaMallocManaged(&wm, sizeof(uint8_t) * wmlen));
    CUDA_CHECK(cudaMallocManaged(&wmget, sizeof(float) * wmlen));
    CUDA_CHECK(cudaMallocManaged(&info, sizeof(int) * numTiles));
    CUDA_CHECK(cudaMallocManaged(&Img, sizeof(float) * orirows * oricols));
    CUDA_CHECK(cudaMallocManaged(&Coefs, sizeof(float) * orirows * oricols));
    CUDA_CHECK(cudaMallocManaged(&dct, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&U, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&V, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&inv, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMallocManaged(&S, sizeof(float) * numTiles * TILE_DIM));
    coefs[0] = &Coefs[0 * rows * cols];
    coefs[1] = &Coefs[1 * rows * cols];
    coefs[2] = &Coefs[2 * rows * cols];
    coefs[3] = &Coefs[3 * rows * cols];
    init_cudalib(&solverHandle, &blasHandle, numTiles, coefs[0], U, S, V, &work, &lwork, &gesvdParams, stream);
    std::cout << "Finnish allocation\n";

    CHECK_READ(myreadbin("../out/gpuout.bin", Img));
    CHECK_READ(myreadbin("../out/wm.bin", wm));

    CUDA_CHECK(cudaMemPrefetchAsync(wm, sizeof(uint8_t) * wmlen, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(wmget, sizeof(float) * wmlen, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(info, sizeof(int) * numTiles, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(Img, sizeof(float) * orirows * oricols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(Coefs, sizeof(float) * orirows * oricols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(dct, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(U, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(V, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(S, sizeof(float) * numTiles * TILE_DIM, device, stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    __TIMER_START__(computation);

    haar_forward2d(rows * 2, cols * 2, Img, oricols, coefs);
    CUDA_CHECK(cudaDeviceSynchronize());

    dct_a100_best_param(rows, cols, coefs[0], cols, dct, cols, stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    writebin("../out/dct.bin", dct, sizeof(float) * rows * cols);

    gesvd_a100_best_param(solverHandle, numTiles, dct, U, S, V, work, lwork, info, gesvdParams);
    CUDA_CHECK(cudaDeviceSynchronize());

    tiled_get_wm_a100_bestparam(numTiles, S, wmget, wmlen, mod1, mod2, stream);

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

    writebin("../out/wmget.bin", wmget, sizeof(float) * wmlen);

    std::cout << "GPU computation " << computation << " ms\n";
    __TIMER_STOP__(end2end);
    std::cout << "GPU end to end " << end2end << " ms\n";

    std::cout << "Exit bwm with 0\n";

}

