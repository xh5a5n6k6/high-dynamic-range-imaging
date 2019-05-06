#include "toneMapper/bilateral.h"

namespace shdr {

BilateralToneMapper::BilateralToneMapper() :
    _delta(0.000001f) {
}

void BilateralToneMapper::solve(cv::Mat hdri, cv::Mat &ldri) {
    fprintf(stderr, "# Begin to implement tone mapping using bilateral method\n");

    ldri = hdri.clone();
    cv::Mat intensity, logIntensity, lowFrequency, highFrequency, newIntensity;

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
    double min, max;
    cv::minMaxLoc(lowFrequency, &min, &max);
    float compressionFactor = float(log(6.0) / (max - min));
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
    float logScale = 1.0f / float(exp(compressionFactor * max));
    std::vector<cv::Mat> vecMat;
    cv::split(ldri, vecMat);
    for (int c = 0; c < 3; c++) {
        cv::divide(vecMat.at(c), intensity, vecMat.at(c));
        vecMat.at(c) = vecMat.at(c).mul(newIntensity) * logScale;
    }
    cv::merge(vecMat, ldri);
    ldri *= 255.0f;
    ldri.convertTo(ldri, CV_8UC3);

    fprintf(stderr, "# Finish to implement tone mapping\n");
}

} // namespace shdr