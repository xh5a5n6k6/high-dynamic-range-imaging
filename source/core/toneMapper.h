#pragma once

#include <opencv2/opencv.hpp>

namespace shdr {

class ToneMapper {
public:
    virtual void solve(const cv::Mat& hdri, 
                       cv::Mat* const out_ldri) const = 0;
};

} // namespace shdr