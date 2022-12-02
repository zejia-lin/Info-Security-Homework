
#include <vector>

#include <opencv2/opencv.hpp>

#include "../src/svd.cu"
#include "../src/mydct.cu"
#include "../src/dwt.cu"


int main(int argc, char **argv){

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Using device " << device << " " << prop.name << std::endl;

    __TIMER_START__(end2end);

    // Read image
    __TIMER_START__(readimg);
    cv::Mat matImg = cv::imread(argv[1]);
    cv::Mat matWm = cv::imread(argv[2]);
    std::vector<cv::Mat> imgYUV(3);
    // imgYUV[0].reserveBuffer(sizeof(uint8_t) * MAX_ROWS * MAX_COLS);
    // imgYUV[1].reserveBuffer(sizeof(uint8_t) * MAX_ROWS * MAX_COLS);
    // imgYUV[2].reserveBuffer(sizeof(uint8_t) * MAX_ROWS * MAX_COLS);

    size_t orirows = matImg.rows;
    size_t oricols = matImg.cols;
    size_t wmlen = matWm.rows * matWm.cols;

    size_t rows = 16 * (orirows / 32);
    size_t cols = 16 * (oricols / 32);
    printf("Ori: (%d, %d), Rnd:(%d, %d)\n", orirows, oricols, rows, cols);
    __TIMER_STOP__(readimg);

    // Preprocess
    __TIMER_START__(preprocess);
    cv::threshold(matImg, matImg, 245, 245, cv::THRESH_TRUNC);
    cv::cvtColor(matImg, matImg, cv::COLOR_BGR2YUV);
    cv::split(matImg, imgYUV);
    cv::cvtColor(matWm, matWm, cv::COLOR_BGR2GRAY);
    cv::threshold(matWm, matWm, 127, 1, cv::THRESH_BINARY);
    __TIMER_STOP__(preprocess);



    // Allocation
    float *U, *S, *V, *inv, *dct;
    float *Coefs;
    float *coefs[4];
    int mod1 = 29, mod2 = 5;
    uint8_t *wm, *wmget, *Img;
    int *info;

    cudaStream_t stream = NULL;
    cublasHandle_t blasHandle;
    cusolverDnHandle_t solverHandle;
    gesvdjInfo_t gesvdParams;
    int lwork;
    float *work;
    size_t numTiles = (rows / TILE_DIM) * (cols / TILE_DIM);

    __TIMER_START__(alloc_time);
    CUDA_CHECK(cudaMallocManaged(&wm, sizeof(uint8_t) * wmlen));
    CUDA_CHECK(cudaMallocManaged(&wmget, sizeof(uint8_t) * wmlen));
    CUDA_CHECK(cudaMallocManaged(&info, sizeof(int) * numTiles));
    CUDA_CHECK(cudaMallocManaged(&Img, sizeof(uint8_t) * orirows * oricols));
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
    __TIMER_STOP__(alloc_time);

    // Copy
    __TIMER_START__(copy_time);
    CUDA_CHECK(cudaMemcpy(Img, imgYUV[0].ptr<uint8_t>(), sizeof(uint8_t) * orirows * oricols, cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(wm, matWm.ptr<uint8_t>(), sizeof(uint8_t) * wmlen, cudaMemcpyDefault));

    CUDA_CHECK(cudaMemPrefetchAsync(wm, sizeof(uint8_t) * wmlen, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(wmget, sizeof(uint8_t) * wmlen, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(info, sizeof(int) * numTiles, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(Img, sizeof(uint8_t) * orirows * oricols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(Coefs, sizeof(float) * orirows * oricols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(dct, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(U, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(V, sizeof(float) * rows * cols, device, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(S, sizeof(float) * numTiles * TILE_DIM, device, stream));
    CUDA_CHECK(cudaDeviceSynchronize());
    __TIMER_STOP__(copy_time);

    // Computation
    __TIMER_START__(computation);
    haar_forward2d(rows * 2, cols * 2, Img, oricols, coefs);
    CUDA_CHECK(cudaDeviceSynchronize());

    dct_a100_best_param(rows, cols, coefs[0], cols, dct, cols, stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    // writebin("../out/dct.bin", dct, sizeof(float) * rows * cols);

    gesvd_a100_best_param(solverHandle, numTiles, dct, U, S, V, work, lwork, info, gesvdParams);
    CUDA_CHECK(cudaDeviceSynchronize());

    tiled_add_wm_a100_bestparam(numTiles, S, wm, wmlen, mod1, mod2, stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    tiled_get_wm_a100_bestparam(numTiles, S, wmget, wmlen, mod1, mod2, stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    mmd_batched_a100_best_param(false, U, S, inv, numTiles);
    CUDA_CHECK(cudaDeviceSynchronize());
    invsvd_a100_best_param(blasHandle, numTiles, inv, U, S, V);
    CUDA_CHECK(cudaDeviceSynchronize());
    idct_a100_best_param(rows, cols, inv, cols, coefs[0], cols, stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    haar_inverse2d(rows * 2, cols * 2, Img, oricols, coefs);

    CUDA_CHECK(cudaDeviceSynchronize());
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

    // Copy
    __TIMER_START__(postproccess);
    cudaMemcpy(imgYUV[0].ptr<uint8_t>(), Img, sizeof(uint8_t) * orirows * oricols, cudaMemcpyDeviceToHost);
    cv::merge(imgYUV, matImg);
    cv::cvtColor(matImg, matImg, cv::COLOR_YUV2BGR);
    cv::imwrite("../out/gpucv.png", matImg);
    cudaMemcpy(matWm.ptr<uint8_t>(), wmget, sizeof(uint8_t) * wmlen, cudaMemcpyDeviceToHost);
    cv::imwrite("../out/gpuwm.png", matWm);
    __TIMER_STOP__(postproccess);

    // std::cout << "====================\nGemm from GPU\n";
    // print_matrix_rowmaj(inv, 8, 8, 8);

    writebin("../out/embeded.bin", coefs[0], sizeof(float) * rows * cols);
    writebin("../out/gpuout.bin", Img, sizeof(float) * orirows * oricols);
    writebin("../out/wmget.bin", wmget, sizeof(uint8_t) * wmlen);
    writebin("../out/wmread.bin", wm, sizeof(uint8_t) * wmlen);

    // for(int i = 0; i < wmlen; ++i){
    //     std::cout << int(wmget[i]) << ", ";
    // }
    // std::cout << "\n";

    std::cout << "GPU computation " << computation << " ms\n";
    __TIMER_STOP__(end2end);
    std::cout << "GPU end to end " << end2end << " ms\n";

    std::cout << "Exit bwm with 0\n";

}

