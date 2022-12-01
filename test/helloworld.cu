
#include <opencv2/opencv.hpp>

int main(){
    cv::Mat img = cv::imread("../pic/lena.png");
    cv::resize(img, img, {128, 64});
    std::cout << int(img.at<uchar>(0, 1));
    cv::imwrite("../out/helloworld.png", img);
}
