#pragma once

#include "config.h"

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
    void solve(const std::vector<cv::Mat>& images, 
               const std::vector<float>&   shutterSpeeds, 
               cv::Mat* const              out_hdri) const;

private:
    virtual void _solveImpl(const std::vector<cv::Mat>& images,
                            const std::vector<float>&   shutterSpeeds,
                            cv::Mat* const              out_hdri) const = 0;

    void _writeHdrImage(const cv::Mat& hdri) const;
};

// header implementation

inline void CrfSolver::solve(const std::vector<cv::Mat>& images,
                             const std::vector<float>&   shutterSpeeds,
                             cv::Mat* const              out_hdri) const {

    _solveImpl(images, shutterSpeeds, out_hdri);

#ifdef DRAW_RADIANCE_MAP
    _writeHdrImage(*out_hdri);

#endif
}

inline void CrfSolver::_writeHdrImage(const cv::Mat& hdri) const {
    cv::imwrite("./hdr_radiance_map.hdr", hdri);
}

} // namespace shdr