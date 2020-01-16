#include "toneMapper/photographicGlobal.h"

#include <cmath>
#include <iostream>

namespace shdr {

PhotographicGlobalToneMapper::PhotographicGlobalToneMapper() :
    PhotographicGlobalToneMapper(0.7f, 0.000001f) {
}

PhotographicGlobalToneMapper::PhotographicGlobalToneMapper(const float alpha, const float delta) :
    _alpha(alpha),
    _delta(delta) {
}

void PhotographicGlobalToneMapper::solve(const cv::Mat& hdri, 
                                         cv::Mat* const out_ldri) const {

    std::cout << "# Begin to implement tone mapping using photographic global method"
              << std::endl;

    cv::Mat ldri = hdri.clone();
    cv::Mat lw;
    cv::Mat logLw;
    cv::Mat lm; 
    cv::Mat ld;

    cv::cvtColor(hdri, lw, cv::COLOR_BGR2GRAY);
    cv::log(lw + _delta, logLw);

    const float meanLogLw = static_cast<float>(cv::mean(logLw)[0]);
    const float meanLw    = std::exp(meanLogLw);
    const float invMeanLw = 1.0f / meanLw;
    lm = _alpha * invMeanLw * lw;

    double min;
    double max;
    cv::minMaxLoc(lm, &min, &max);

    const float lWhite     = static_cast<float>(max);
    const float invLWhite2 = 1.0f / (lWhite * lWhite);

    const cv::Mat up   = 1.0f + lm * invLWhite2;
    const cv::Mat down = 1.0f + lm;
    cv::divide(up, down, ld);
    ld = ld.mul(lm);

    /*
        calculate each channel
    */
    std::vector<cv::Mat> vecMat;
    cv::split(ldri, vecMat);
    for (int c = 0; c < 3; ++c) {
        cv::divide(vecMat[c], lw, vecMat[c]);
        vecMat[c] = vecMat[c].mul(ld);
    }
    cv::merge(vecMat, ldri);
    ldri *= 255.0f;
    ldri.convertTo(ldri, CV_8UC3);

    *out_ldri = ldri;

    std::cout << "# Finish implementing tone mapping"
              << std::endl;
}

} // namespace shdr