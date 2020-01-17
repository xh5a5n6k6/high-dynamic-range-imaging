#include "toneMapper/photographicLocalToneMapper.h"

#include <cmath>
#include <iostream>

namespace shdr {

PhotographicLocalToneMapper::PhotographicLocalToneMapper() :
    PhotographicLocalToneMapper(0.3f, 0.000001f, 8.0f, 0.05f, 35) {
}

PhotographicLocalToneMapper::PhotographicLocalToneMapper(const float alpha,
                                                         const float delta,
                                                         const float phi,
                                                         const float epsilon,
                                                         const int   maxKernelSize) :
    _alpha(alpha),
    _delta(delta),
    _phi(phi),
    _epsilon(epsilon),
    _maxKernelSize(maxKernelSize) {
}

void PhotographicLocalToneMapper::map(const cv::Mat& hdri, 
                                      cv::Mat* const out_ldri) const {

    std::cout << "# Begin to implement tone mapping using photographic local method"
              << std::endl;

    cv::Mat ldri = hdri.clone();
    cv::Mat lw;
    cv::Mat logLw;
    cv::Mat lm;
    cv::Mat ld;
    cv::Mat lsmax;

    cv::cvtColor(hdri, lw, cv::COLOR_BGR2GRAY);
    cv::log(lw + _delta, logLw);
    
    const float meanLogLw = static_cast<float>(cv::mean(logLw)[0]);
    const float meanLw    = std::exp(meanLogLw);
    const float invMeanLw = 1.0f / meanLw;
    lm = _alpha * invMeanLw * lw;

    _localOperator(lm, &lsmax);
    cv::divide(lm, 1.0f + lsmax, ld);

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

void PhotographicLocalToneMapper::_localOperator(const cv::Mat& lm, cv::Mat* const out_lsmax) const {
    cv::Mat lsmax = lm.clone();

    const int width      = lm.cols;
    const int height     = lm.rows;
    const int numKernels = (_maxKernelSize-1) / 2 + 1;
	
    /*
        Create all blur images
    */
    std::vector<cv::Mat> lblur;
    for (int size = 1; size <= _maxKernelSize; size += 2) {
        cv::Mat blur;
        cv::GaussianBlur(lm, blur, cv::Size(size, size), 0);
        lblur.push_back(blur);
    }

    /*
        For each pixel, find its lsmax
    */
    cv::Mat index = cv::Mat::zeros(lm.size(), CV_8UC1);
    cv::Mat vs;
    // for each kernel size (blur image)
    for (int n = 0; n < numKernels - 1; n++) {
        const int s = 1 + 2 * n;
        
        const cv::Mat up   = lblur[n] - lblur[n + 1];
        const cv::Mat down = std::pow(2.0f, _phi) * _alpha / (s * s) + lblur[n];
        cv::divide(up, down, vs);
        vs = cv::abs(vs);

        // check if vs < epsilon
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                if (vs.at<float>(j, i) < _epsilon) {
                    if (index.at<uchar>(j, i) == 0) {
                        const cv::Mat& ls = lblur[n];
                        lsmax.at<float>(j, i) = ls.at<float>(j, i);
                    }
                }
                else {
                    index.at<uchar>(j, i) = 1;
                }
            }
        }
    }

    *out_lsmax = lsmax;
}

} // namespace shdr