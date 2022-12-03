
#include <thread>

#include "../src/wm_core.cpp"

int main(int argc, char **argv) {

    std::cout << cv::getBuildInformation();

    if (argc < 3) {
        std::cout << "Usage: ./demo_video [videoPath] [wmPath] [batchSize]\n";
        return -1;
    }

    int batchSize = 8;
    std::string videoPath = argv[1];
    std::string wmPath = argv[2];
    if (argc > 3) {
        batchSize = atoi(argv[3]);
    }

    std::vector<cv::Mat> frames(batchSize);
    std::vector<LzjWatermark> wmMakers(batchSize);
    for(int i = 0; i < batchSize; ++i){
        CUDA_CHECK(cudaStreamCreate(&wmMakers[i].stream));
        std::cout << "Created stream " << wmMakers[i].stream << "\n";
    }

    cv::Mat matWm = cv::imread(wmPath);
    cv::VideoCapture cap;
    if (!cap.open(videoPath)) {
        std::cout << "Fail to open " << videoPath << "\n";
        return -1;
    }
    while (true) {
        for(int i = 0; i < batchSize; ++i){
            if (!cap.read(frames[i])) {
                std::cout << "Read end of " << videoPath << "\n";
                break;
            }
            std::thread(&LzjWatermark::embed, &wmMakers[i], frames[i], matWm);
        }
    }

    cap.release();
}
