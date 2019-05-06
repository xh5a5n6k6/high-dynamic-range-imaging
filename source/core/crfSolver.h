#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

namespace shdr {

class CRFSolver {
public:
    virtual void solve(std::vector<cv::Mat> images, std::vector<float> shutterSpeeds, cv::Mat &hdri) = 0;
};

} // namespace shdr