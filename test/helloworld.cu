
#include <opencv2/opencv.hpp>

int main(){
    auto img = cv::imread("../pic/wm.png");
    cv::resize(img, img, {128, 64});
    cv::imwrite("../out/helloworld.png", img);
}
