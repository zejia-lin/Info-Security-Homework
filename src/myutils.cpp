
#include <iostream>
#include <sys/stat.h>
#include <fstream>
#include <chrono>

#define IDX(i, j, ld) (((i) * (ld)) + (j))

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

#define TILE_DIM 4

using DTYPE = float;
using ACC_TYPE = double;

#define SQRT1 0.5 // sqrt(1 / 4)
#define SQRT2 0.7071067811865475727373109293694142252206802368164062  // sqrt(2 / 4)


__constant__ ACC_TYPE COSINES[16] = {
    1.0, 0.9238795325112867384831361050601117312908172607421875, 0.7071067811865475727373109293694142252206802368164062, 0.3826834323650898372903839117498137056827545166015625,
    1.0, 0.3826834323650898372903839117498137056827545166015625, -0.7071067811865474617150084668537601828575134277343750, -0.9238795325112868495054385675757657736539840698242188,
    1.0, -0.3826834323650897262680814492341596633195877075195312, -0.7071067811865476837596133918850682675838470458984375, 0.9238795325112865164385311800288036465644836425781250,
    1.0, -0.9238795325112867384831361050601117312908172607421875, 0.7071067811865473506927060043381061404943466186523438, -0.3826834323650898928015351430076407268643379211425781
};

__constant__ ACC_TYPE ALPHAS[16] = {
    SQRT1, SQRT2, SQRT2, SQRT2, 
    SQRT2, SQRT2, SQRT2, SQRT2, 
    SQRT2, SQRT2, SQRT2, SQRT2, 
    SQRT2, SQRT2, SQRT2, SQRT2
};



void print_matrix_colmaj(float *A, int rows, int cols, int lda){
    for(int i = 0; i < rows; ++i){
        printf("[");
        for(int j = 0; j < cols; ++j){
            printf("%.3f, ", A[IDX(j, i, lda)]);
        }
        printf("]\n");
    }
    printf("\n");
}

void print_matrix_rowmaj(float *A, int rows, int cols, int lda){
    for(int i = 0; i < rows; ++i){
        printf("[");
        for(int j = 0; j < cols; ++j){
            printf("%.3f, ", A[IDX(i, j, lda)]);
        }
        printf("]\n");
    }
    printf("\n");
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

