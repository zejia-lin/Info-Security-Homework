
#include <iostream>
#include <fstream>
#include <string>

#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

#include "dwt.cu"
#include "mmd.cu"
#include "mydct.cu"
#include "svd.cu"
#include "myutils.cpp"


class LzjWatermark{
public:
    const int mod1 = 29, mod2 = 5;
    float *U, *S, *V, *inv, *dct;
    float *Coefs;
    float *coefs[4];
    uint8_t *wm, *Img;
    int *info;

    int device = 0;
    cudaStream_t stream;
    cublasHandle_t blasHandle;
    cusolverDnHandle_t solverHandle;
    gesvdjInfo_t gesvdParams;
    int lwork;
    float *work;

    LzjWatermark(cudaStream_t ss=0, size_t max_rows=MAX_ROWS, size_t max_cols=MAX_COLS){
        stream = ss;
        size_t rows = max_rows / 2;
        size_t cols = max_cols / 2;
        size_t numTiles = (rows / TILE_DIM) * (cols / TILE_DIM);

        CUDA_CHECK(cudaMallocManaged(&wm, sizeof(uint8_t) * numTiles));
        CUDA_CHECK(cudaMallocManaged(&info, sizeof(int) * numTiles));
        CUDA_CHECK(cudaMallocManaged(&Img, sizeof(uint8_t) * max_rows * max_cols));
        CUDA_CHECK(cudaMallocManaged(&Coefs, sizeof(float) * max_rows * max_cols));
        CUDA_CHECK(cudaMallocManaged(&dct, sizeof(float) * rows * cols));
        CUDA_CHECK(cudaMallocManaged(&U, sizeof(float) * rows * cols));
        CUDA_CHECK(cudaMallocManaged(&V, sizeof(float) * rows * cols));
        CUDA_CHECK(cudaMallocManaged(&inv, sizeof(float) * rows * cols));
        CUDA_CHECK(cudaMallocManaged(&S, sizeof(float) * numTiles * TILE_DIM));

        CUDA_CHECK(cudaMemPrefetchAsync(wm, sizeof(uint8_t) * numTiles, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(info, sizeof(int) * numTiles, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(Img, sizeof(uint8_t) * max_rows * max_cols, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(Coefs, sizeof(float) * max_rows * max_cols, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(dct, sizeof(float) * rows * cols, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(U, sizeof(float) * rows * cols, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(V, sizeof(float) * rows * cols, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(S, sizeof(float) * numTiles * TILE_DIM, device, stream));

        coefs[0] = &Coefs[0 * rows * cols];
        coefs[1] = &Coefs[1 * rows * cols];
        coefs[2] = &Coefs[2 * rows * cols];
        coefs[3] = &Coefs[3 * rows * cols];

        init_cudalib(&solverHandle, &blasHandle, numTiles, coefs[0], U, S, V, &work, &lwork, &gesvdParams, stream);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "Initiated Watermark at " << this << "\n";
    }

    void embed(cv::Mat matImg, cv::Mat matWm){
        
        // shapes
        std::vector<cv::Mat> imgYUV(3);
        size_t orirows = matImg.rows;
        size_t oricols = matImg.cols;
        size_t wmlen = matWm.rows * matWm.cols;
        size_t rows = 16 * (orirows / 32);
        size_t cols = 16 * (oricols / 32);
        size_t numTiles = (rows / TILE_DIM) * (cols / TILE_DIM);

        // preproccess
        cv::threshold(matImg, matImg, 245, 245, cv::THRESH_TRUNC);
        cv::cvtColor(matImg, matImg, cv::COLOR_BGR2YUV);
        cv::split(matImg, imgYUV);
        cv::cvtColor(matWm, matWm, cv::COLOR_BGR2GRAY);
        cv::threshold(matWm, matWm, 127, 1, cv::THRESH_BINARY);

        // copy
        CUDA_CHECK(cudaMemcpyAsync(Img, imgYUV[0].ptr<uint8_t>(), 
                                    sizeof(uint8_t) * orirows * oricols, cudaMemcpyDefault, stream));
        CUDA_CHECK(cudaMemcpyAsync(wm, matWm.ptr<uint8_t>(), 
                                    sizeof(uint8_t) * wmlen, cudaMemcpyDefault, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(wm, sizeof(uint8_t) * wmlen, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(info, sizeof(int) * numTiles, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(Img, sizeof(uint8_t) * orirows * oricols, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(Coefs, sizeof(float) * orirows * oricols, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(dct, sizeof(float) * rows * cols, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(U, sizeof(float) * rows * cols, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(V, sizeof(float) * rows * cols, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(S, sizeof(float) * numTiles * TILE_DIM, device, stream));
        
        // computation
        haar_forward2d(rows * 2, cols * 2, Img, oricols, coefs);
        dct_a100_best_param(rows, cols, coefs[0], cols, dct, cols, stream);
        gesvd_a100_best_param(solverHandle, numTiles, dct, U, S, V, work, lwork, info, gesvdParams);
        tiled_add_wm_a100_bestparam(numTiles, S, wm, wmlen, mod1, mod2, stream);
        mmd_batched_a100_best_param(false, U, S, inv, numTiles);
        invsvd_a100_best_param(blasHandle, numTiles, inv, U, S, V);
        idct_a100_best_param(rows, cols, inv, cols, coefs[0], cols, stream);
        haar_inverse2d(rows * 2, cols * 2, Img, oricols, coefs);

        // validation
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

        // postproccess
        CUDA_CHECK(cudaMemcpyAsync(imgYUV[0].ptr<uint8_t>(), Img, 
                        sizeof(uint8_t) * orirows * oricols, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cv::merge(imgYUV, matImg);
        cv::cvtColor(matImg, matImg, cv::COLOR_YUV2BGR);

    }

    cv::Mat extract(cv::Mat matImg, size_t wmRows, size_t wmCols){
                
        // shapes
        std::vector<cv::Mat> imgYUV(3);
        cv::Mat matWm(cv::Size(wmCols, wmRows), CV_8UC1);
        size_t orirows = matImg.rows;
        size_t oricols = matImg.cols;
        size_t wmlen = matWm.rows * matWm.cols;
        size_t rows = 16 * (orirows / 32);
        size_t cols = 16 * (oricols / 32);
        size_t numTiles = (rows / TILE_DIM) * (cols / TILE_DIM);

        // preproccess
        cv::threshold(matImg, matImg, 245, 245, cv::THRESH_TRUNC);
        cv::cvtColor(matImg, matImg, cv::COLOR_BGR2YUV);
        cv::split(matImg, imgYUV);

        // copy
        CUDA_CHECK(cudaMemcpyAsync(Img, imgYUV[0].ptr<uint8_t>(), 
                                    sizeof(uint8_t) * orirows * oricols, cudaMemcpyDefault, stream));
        CUDA_CHECK(cudaMemcpyAsync(wm, matWm.ptr<uint8_t>(), 
                                    sizeof(uint8_t) * wmlen, cudaMemcpyDefault, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(wm, sizeof(uint8_t) * wmlen, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(info, sizeof(int) * numTiles, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(Img, sizeof(uint8_t) * orirows * oricols, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(Coefs, sizeof(float) * orirows * oricols, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(dct, sizeof(float) * rows * cols, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(U, sizeof(float) * rows * cols, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(V, sizeof(float) * rows * cols, device, stream));
        CUDA_CHECK(cudaMemPrefetchAsync(S, sizeof(float) * numTiles * TILE_DIM, device, stream));
        
        // computation
        haar_forward2d(rows * 2, cols * 2, Img, oricols, coefs);
        dct_a100_best_param(rows, cols, coefs[0], cols, dct, cols, stream);
        gesvd_a100_best_param(solverHandle, numTiles, dct, U, S, V, work, lwork, info, gesvdParams);
        tiled_get_wm_a100_bestparam(numTiles, S, wm, wmlen, mod1, mod2, stream);

        // validation
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

        // postproccess
        CUDA_CHECK(cudaMemcpy(matWm.ptr<uint8_t>(), wm, sizeof(uint8_t) * wmlen, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        return matWm;
    }

};


void lzjCLI(bool prompt=true){
#define PROMPT if(prompt) std::cout
    std::cout << "Starting...\n";
    std::string line, filename;
    LzjWatermark wmobj;
    PROMPT << "> ";
    while(std::getline(std::cin, line)){
        try{
            auto cmds = split(line);
            if(cmds[0] == "embed"){
                cv::Mat matImg = cv::imread(cmds[1]);
                cv::Mat matWm = cv::imread(cmds[2]);
                std::string outpath = cmds[3];
                wmobj.embed(matImg, matWm);
                if(!cv::imwrite(outpath, matImg)){
                    PROMPT << "Fail to write " << outpath << "\n";
                } else {
                    PROMPT << "Save " << outpath << "\n";
                }
            } else if(cmds[0] == "extract"){
                int rows = std::stoi(cmds[1]);
                int cols = std::stoi(cmds[2]);
                cv::Mat matImg = cv::imread(cmds[3]);
                std::string outpath = cmds[4];
                cv::Mat matWm = wmobj.extract(matImg, rows, cols);
                if(!cv::imwrite(outpath, matWm)){
                    PROMPT << "Fail to write " << outpath << "\n";
                } else {
                    PROMPT << "Save " << outpath << "\n";
                }
            } else if(cmds[0] == "quit"){
                break;
            } else {
                PROMPT << "[embed|extract]\n";
            }
            PROMPT << "> ";
        } catch (std::exception e){
            std::cout << e.what() << "\n";
        }
    }
    std::cout << "Shutdown normally\n";
#undef PROMPT
}


