
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


// must be run with grid size = (Nc/2, Nr/2)  where Nr = numrows of input
template<typename T>
__global__ void kern_haar2d_fwd(T* img, T* c_a, T* c_h, T* c_v, T* c_d, int Nr, int Nc) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    int Nr_is_odd = (Nr & 1);
    int Nc_is_odd = (Nc & 1);
    int Nr2 = (Nr + Nr_is_odd) / 2;
    int Nc2 = (Nc + Nc_is_odd) / 2;
    if (gidy < Nr2 && gidx < Nc2) {

        // for odd N, image is virtually extended by repeating the last element
        int posx0 = 2*gidx;
        int posx1 = 2*gidx+1;
        if ((Nc_is_odd) && (posx1 == Nc)) posx1--;
        int posy0 = 2*gidy;
        int posy1 = 2*gidy+1;
        if ((Nr_is_odd) && (posy1 == Nr)) posy1--;

        T a = img[posy0*Nc + posx0];
        T b = img[posy0*Nc + posx1];
        T c = img[posy1*Nc + posx0];
        T d = img[posy1*Nc + posx1];

        c_a[gidy* Nc2 + gidx] = 0.5*HAAR_AVG(HAAR_AVG(a, c), HAAR_AVG(b, d)); // A
        c_v[gidy* Nc2 + gidx] = 0.5*HAAR_DIF(HAAR_AVG(a, c), HAAR_AVG(b, d)); // V
        c_h[gidy* Nc2 + gidx] = 0.5*HAAR_AVG(HAAR_DIF(a, c), HAAR_DIF(b, d)); // H
        c_d[gidy* Nc2 + gidx] = 0.5*HAAR_DIF(HAAR_DIF(a, c), HAAR_DIF(b, d)); // D
    }
}

template<typename T>
int haar_forward2d(T* d_image, T** d_coeffs, int Nr, int Nc) {

    int tpb = 32;
    dim3 dimGrid = dim3(w_iDivUp((Nc + 1) / 2, tpb), w_iDivUp((Nr + 1) / 2, tpb));
    dim3 dimBlock = dim3(tpb, tpb);
    kern_haar2d_fwd<<<dimGrid, dimBlock>>>(d_image, d_coeffs[0], d_coeffs[1], d_coeffs[2], d_coeffs[3], Nr, Nc);

    return 0;
}

// must be run with grid size = (2*Nr, 2*Nc) ; Nr = numrows of input
__global__ void kern_haar2d_inv(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int Nr2, int Nc2) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    if (gidy < Nr2 && gidx < Nc2) {
        DTYPE a = c_a[(gidy/2)*Nc + (gidx/2)];
        DTYPE b = c_v[(gidy/2)*Nc + (gidx/2)];
        DTYPE c = c_h[(gidy/2)*Nc + (gidx/2)];
        DTYPE d = c_d[(gidy/2)*Nc + (gidx/2)];
        DTYPE res = 0.0f;
        int gx1 = (gidx & 1), gy1 = (gidy & 1);
        if (gx1 == 0 && gy1 == 0) res = 0.5*HAAR_AVG(HAAR_AVG(a, c), HAAR_AVG(b, d));
        if (gx1 == 1 && gy1 == 0) res = 0.5*HAAR_DIF(HAAR_AVG(a, c), HAAR_AVG(b, d));
        if (gx1 == 0 && gy1 == 1) res = 0.5*HAAR_AVG(HAAR_DIF(a, c), HAAR_DIF(b, d));
        if (gx1 == 1 && gy1 == 1) res = 0.5*HAAR_DIF(HAAR_DIF(a, c), HAAR_DIF(b, d));
        img[gidy*Nc2 + gidx] = res;
    }

}

int haar_inverse2d(DTYPE* d_image, DTYPE** d_coeffs, size_t Nr, size_t Nc) {
    int tpb = 32;
    dim3 dimGrid = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb));
    dim3 dimBlock = dim3(tpb, tpb);
    kern_haar2d_inv<<<dimGrid, dimBlock>>>(d_image, d_coeffs[0], d_coeffs[1], d_coeffs[2], d_coeffs[3], 
            (Nr + 1) / 2, (Nc + 1) / 2, Nr, Nc);

    return 0;
}


