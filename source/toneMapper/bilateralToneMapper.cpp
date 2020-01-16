#include "toneMapper/bilateralToneMapper.h"

#include <cmath>
#include <iostream>

namespace shdr {

BilateralToneMapper::BilateralToneMapper() :
    BilateralToneMapper(0.000001f) {
}

BilateralToneMapper::BilateralToneMapper(const float delta) :
    _delta(delta) {
}

void BilateralToneMapper::solve(const cv::Mat& hdri, 
                                cv::Mat* const out_ldri) const {

    std::cout << "# Begin to implement tone mapping using bilateral method"
              << std::endl;

    cv::Mat ldri = hdri.clone();
    cv::Mat intensity; 
    cv::Mat logIntensity;
    cv::Mat lowFrequency;
    cv::Mat highFrequency;
    cv::Mat newIntensity;

    /*
        We need to separate intensity & color,
        first calculate its intensity
    */
    cv::cvtColor(hdri, intensity, cv::COLOR_BGR2GRAY);
    cv::log(intensity + _delta, logIntensity);

    /*
        Split to low frequency image & high frequency image
    */
    cv::bilateralFilter(logIntensity, lowFrequency, 5, 30, 30);
    highFrequency = logIntensity - lowFrequency;

    /*
        Now we need to reduce contrast in low frequency image
    */
    double min;
    double max;
    cv::minMaxLoc(lowFrequency, &min, &max);

    const float compressionFactor = static_cast<float>(std::log(6.0) / (max - min));
    lowFrequency *= compressionFactor;

    /*
        Now we combine reduced contrast low frequency image
        and high frequency image to new intensity image
    */
    newIntensity = lowFrequency + highFrequency;
    cv::exp(newIntensity, newIntensity);

    /*
        Recalculate color for each channel
    */
    const float logScale = 1.0f / static_cast<float>(std::exp(compressionFactor * max));
    std::vector<cv::Mat> vecMat;
    cv::split(ldri, vecMat);
    for (int c = 0; c < 3; ++c) {
        cv::divide(vecMat[c], intensity, vecMat[c]);
        vecMat[c] = vecMat[c].mul(newIntensity) * logScale;
    }
    cv::merge(vecMat, ldri);
    ldri *= 255.0f;
    ldri.convertTo(ldri, CV_8UC3);

    *out_ldri = ldri;

    std::cout << "# Finish implementing tone mapping"
              << std::endl;
}

} // namespace shdr