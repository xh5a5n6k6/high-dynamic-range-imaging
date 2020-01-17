#pragma once

#include "core/imageAligner.h"

namespace shdr {

/*
    MtbImageAligner: Median Threshold Bitmap Image Aligner
*/
class MtbImageAligner : public ImageAligner {
public:
    MtbImageAligner();

    void align(const std::vector<cv::Mat>& images,
               std::vector<cv::Mat>* const out_alignImages) const override;

private:
    void _calculateBitmap(const cv::Mat&              image,
                          std::vector<cv::Mat>* const out_vecMtb,
                          std::vector<cv::Mat>* const out_vecEb) const;

    int  _findMedian(const cv::Mat& image) const;

    static const int MAX_MTB_LEVEL = 5;
};

} // namespace shdr