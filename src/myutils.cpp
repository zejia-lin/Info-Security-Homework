
#pragma once

#include <iostream>
#include <sys/stat.h>
#include <fstream>
#include <chrono>
#include <vector>


#define IDX(i, j, ld) (((i) * (ld)) + (j))

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

#define ENABLE_TIMER

#ifdef ENABLE_TIMER
#define __TIMER_START__(_ms) double _ms; { \
    auto _start_timer = clock_type::now();

#define __TIMER_STOP__(_ms) \
    auto _end_timer = clock_type::now(); \
    _ms = chrono::duration_cast<chrono::microseconds>(_end_timer - _start_timer).count() / 1000.; \
}
#else
#define __TIMER_START__
#define __TIMER_STOP__(_duration)
#endif

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__);      \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                        \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUSPARSE_CHECK(err)                                                                        \
    do {                                                                                           \
        cusparseStatus_t err_ = (err);                                                             \
        if (err_ != CUSPARSE_STATUS_SUCCESS) {                                                     \
            printf("cusparse error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusparse error");                                            \
        }                                                                                          \
    } while (0)



void print_matrix_colmaj(float *A, int rows, int cols, int lda){
    printf("[");
    for(int i = 0; i < rows; ++i){
        printf("[");
        for(int j = 0; j < cols; ++j){
            printf("%.3f, ", A[IDX(j, i, lda)]);
        }
        printf("]");
        if(i != rows - 1){
            printf(",\n");
        }
    }
    printf("]\n\n");
}

template<typename T>
void print_matrix_rowmaj(T *A, int rows, int cols, int lda){
    printf("[");
    for(int i = 0; i < rows; ++i){
        printf("[");
        for(int j = 0; j < cols; ++j){
            printf("%.7f, ", A[IDX(i, j, lda)]);
        }
        printf("]");
        if(i != rows - 1){
            printf(",\n");
        }
    }
    printf("]\n\n");
}

void print_vector(float *A, int n){
    for(int i = 0; i < n; ++i){
        std::cout << A[i] << ", ";
    }
    std::cout << std::endl;
}


int myreadbin(const std::string &filepath, void *buffer){
    std::ifstream fin(filepath, std::ios::binary);
    std::vector<unsigned char> bb(std::istreambuf_iterator<char>(fin), {});
    memcpy(buffer, bb.data(), sizeof(char) * bb.size());
    return bb.size();
}


char *readbin(const std::string &filePath, size_t *fileSize, void *buffer, size_t bufferSize) {
    std::cout << "Begin read " << filePath << std::endl;
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        std::cout << "failed to get file" << std::endl;
        return nullptr;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        std::cout << filePath.c_str() << " is not a file, please enter a file" << std::endl;
        return nullptr;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Open file failed. path " << filePath.c_str() << std::endl;
        return nullptr;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        std::cout << "file size is 0" << std::endl;
        file.close();
        return nullptr;
    }
    std::cout << "File size = " << size << ", buffer size = " << bufferSize << std::endl;
    if (size > bufferSize) {
        std::cout << "file size = " << size << " is larger than buffer size = " << bufferSize << std::endl;
        file.close();
        return nullptr;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    *fileSize = size;
    file.close();
    std::cout << "Finish read file" << std::endl;
    return nullptr;
}

bool writebin(const std::string &filePath, const void *buffer, size_t size) {
    if (buffer == nullptr) {
        // ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    FILE *outputFile = fopen(filePath.c_str(), "wb");
    if (outputFile == nullptr) {
        // ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    fwrite(buffer, size, sizeof(char), outputFile);
    fclose(outputFile);

    return true;
}

