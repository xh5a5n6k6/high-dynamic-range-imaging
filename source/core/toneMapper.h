#pragma once

#include <opencv2/opencv.hpp>

namespace shdr {

/*
    ToneMapper is used for tone mapping for hdr image.

    Because it eventually needs to display image on the
    monitor, which is often the low dynamic range display 
    system, we need to compress the full dynamic range 
    to ldr (0-255) but preserving the contrast so that 
    it gives a similar visual match.
*/
class ToneMapper {
public:
    virtual void map(const cv::Mat& hdri, 
                     cv::Mat* const out_ldri) const = 0;
};

} // namespace shdr