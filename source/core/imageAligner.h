#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace shdr {

/*
    ImageAligner if used for image alignment.

    It is suggested to run image alignment before
    radiance map reconstruction.
*/
class ImageAligner {
public:
    virtual void align(const std::vector<cv::Mat>& images,
                       std::vector<cv::Mat>* const out_alignImages) const = 0;
};

} // namespace shdr