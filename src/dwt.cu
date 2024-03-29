
#pragma once

// https://github.com/pierrepaleo/pypwt/blob/d225e097c8761dcab30d34e2ea4cd20bf11374e9/pdwt/src/haar.cu

#include <cuda_runtime.h>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>

#include "myutils.cpp"
#include "constants.h"

#define HAAR_AVG(a, b) ((a+b))
#define HAAR_DIF(a, b) ((a-b))

int w_iDivUp(int a, int b) {
    return (a + b - 1) / b;
}


__global__ void light_cutoff(size_t rows, size_t cols, uint8_t *img, size_t lda, uint8_t thresh){
    for(size_t tid = threadIdx.x + blockIdx.x * blockDim.x; 
        tid < rows * cols; 
        tid += blockDim.x * gridDim.x){
        img[tid] = min(img[tid], thresh);
    }
}


// must be run with grid size = (Nc/2, Nr/2)  where Nr = numrows of input
__global__ void kern_haar2d_fwd(size_t Nr, size_t Nc, uint8_t* img, size_t lda, 
                                float* c_a, float* c_h, float* c_v, float* c_d) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    int Nr2 = Nr / 2;
    int Nc2 = Nc / 2;
    if (gidy < Nr2 && gidx < Nc2) {

        int posx0 = 2*gidx;
        int posx1 = 2*gidx+1;
        int posy0 = 2*gidy;
        int posy1 = 2*gidy+1;

        int16_t a = img[posy0*lda + posx0];
        int16_t b = img[posy0*lda + posx1];
        int16_t c = img[posy1*lda + posx0];
        int16_t d = img[posy1*lda + posx1];

        c_a[gidy* Nc2 + gidx] = 0.5*HAAR_AVG(HAAR_AVG(a, c), HAAR_AVG(b, d)); // A
        c_v[gidy* Nc2 + gidx] = 0.5*HAAR_DIF(HAAR_AVG(a, c), HAAR_AVG(b, d)); // V
        c_h[gidy* Nc2 + gidx] = 0.5*HAAR_AVG(HAAR_DIF(a, c), HAAR_DIF(b, d)); // H
        c_d[gidy* Nc2 + gidx] = 0.5*HAAR_DIF(HAAR_DIF(a, c), HAAR_DIF(b, d)); // D
    }
}


int haar_forward2d(size_t Nr, size_t Nc, uint8_t* d_image, size_t lda, float** d_coeffs) {

    int tpb = 32;
    dim3 dimGrid = dim3(w_iDivUp((Nc + 1) / 2, tpb), w_iDivUp((Nr + 1) / 2, tpb));
    dim3 dimBlock = dim3(tpb, tpb);
    kern_haar2d_fwd<<<dimGrid, dimBlock>>>(Nr, Nc, d_image, lda, d_coeffs[0], d_coeffs[1], d_coeffs[2], d_coeffs[3]);

    return 0;
}

// must be run with grid size = (2*Nr, 2*Nc) ; Nr = numrows of input
__global__ void kern_haar2d_inv(uint8_t* img, size_t lda, 
                                float* c_a, float* c_h, float* c_v, float* c_d, 
                                int Nr, int Nc, int Nr2, int Nc2) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    if (gidy < Nr2 && gidx < Nc2) {
        int16_t a = c_a[(gidy/2)*Nc + (gidx/2)];
        int16_t b = c_v[(gidy/2)*Nc + (gidx/2)];
        int16_t c = c_h[(gidy/2)*Nc + (gidx/2)];
        int16_t d = c_d[(gidy/2)*Nc + (gidx/2)];
        float res = 0.0f;
        int gx1 = (gidx & 1), gy1 = (gidy & 1);
        if (gx1 == 0 && gy1 == 0) res = 0.5*HAAR_AVG(HAAR_AVG(a, c), HAAR_AVG(b, d));
        if (gx1 == 1 && gy1 == 0) res = 0.5*HAAR_DIF(HAAR_AVG(a, c), HAAR_AVG(b, d));
        if (gx1 == 0 && gy1 == 1) res = 0.5*HAAR_AVG(HAAR_DIF(a, c), HAAR_DIF(b, d));
        if (gx1 == 1 && gy1 == 1) res = 0.5*HAAR_DIF(HAAR_DIF(a, c), HAAR_DIF(b, d));
        img[gidy*lda + gidx] = res;
    }

}

int haar_inverse2d(size_t Nr, size_t Nc, uint8_t* d_image, size_t lda, float** d_coeffs) {
    int tpb = 32;
    dim3 dimGrid = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb));
    dim3 dimBlock = dim3(tpb, tpb);
    kern_haar2d_inv<<<dimGrid, dimBlock>>>(d_image, lda, d_coeffs[0], d_coeffs[1], d_coeffs[2], d_coeffs[3], 
            (Nr + 1) / 2, (Nc + 1) / 2, Nr, Nc);

    return 0;
}


