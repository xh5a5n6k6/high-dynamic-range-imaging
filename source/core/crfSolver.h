#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace shdr {

/*
    CrfSolver: Camera Response Function Solver

    Because camera can not capture the full dynamic 
    range, we want to use multiple images with different 
    shutter speeds to reconstruct the radiance map.

    CrfSolver's goal is to recover response curve of 
    the camera, and then we can use these information
    to reconstruct the radiance map (hdr image).
*/
class CrfSolver {
public:
    virtual void solve(const std::vector<cv::Mat>& images, 
                       const std::vector<float>&   shutterSpeeds, 
                       cv::Mat* const              out_hdri) const = 0;
};

} // namespace shdr