
#include <iostream>
#include <sys/stat.h>
#include <fstream>
#include <chrono>

#ifndef IDX
#define IDX(i, j, ld) (((i) * (ld)) + j)
#endif

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

#define ENABLE_TIMER

#ifdef ENABLE_TIMER
#define __TIMER_START__ { \
    auto _start_timer = clock_type::now();

#define __TIMER_STOP__(_duration) \
    auto _end_timer = clock_type::now(); \
    _duration = chrono::duration_cast<chrono::microseconds>(_end_timer - _start_timer).count(); \
}
#else
#define __TIMER_START__
#define __TIMER_STOP__(_duration)
#endif



void print_matrix(float *A, int rows, int cols){
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            printf("%.3f, ", A[IDX(i, j, cols)]);
        }
        std::cout << std::endl;
    }
}

void print_vector(float *A, int n){
    for(int i = 0; i < n; ++i){
        std::cout << A[i] << ", ";
    }
    std::cout << std::endl;
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
    return static_cast<char *>(buffer);
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

