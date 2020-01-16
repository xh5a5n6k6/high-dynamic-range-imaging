#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace shdr {

/*
    CrfSolver: Camera Response Function Solver


*/
class CrfSolver {
public:
    virtual void solve(const std::vector<cv::Mat>& images, 
                       const std::vector<float>&   shutterSpeeds, 
                       cv::Mat* const              out_hdri) const = 0;
};

} // namespace shdr