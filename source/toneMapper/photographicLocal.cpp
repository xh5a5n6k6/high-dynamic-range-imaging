#include "toneMapper/photographicLocal.h"

namespace shdr {

PhotographicLocalToneMapper::PhotographicLocalToneMapper() :
    _alpha(0.3f), _delta(0.000001f), _phi(8.0f), _epsilon(0.05f), _maxKernelSize(35) {
}

void PhotographicLocalToneMapper::solve(cv::Mat hdri, cv::Mat &ldri) {
    fprintf(stderr, "# Begin to implement tone mapping using photographic local method\n");

    ldri = hdri.clone();
    cv::Mat lw, logLw, lm, ld, lsmax;
    cv::cvtColor(hdri, lw, cv::COLOR_BGR2GRAY);
    cv::log(lw + _delta, logLw);
    float meanLogLw = float(cv::mean(logLw)[0]);
    float meanLw = exp(meanLogLw);
    float invMeanLw = 1.0f / meanLw;
    lm = _alpha * invMeanLw * lw;

    _localOperator(lm, lsmax);
    cv::divide(lm, 1.0f + lsmax, ld);

    /*
        calculate each channel
    */
    std::vector<cv::Mat> vecMat;
    cv::split(ldri, vecMat);
    for (int c = 0; c < 3; c++) {
        cv::divide(vecMat.at(c), lw, vecMat.at(c));
        vecMat.at(c) = vecMat.at(c).mul(ld);
    }
    cv::merge(vecMat, ldri);
    ldri *= 255.0f;
    ldri.convertTo(ldri, CV_8UC3);

    fprintf(stderr, "# Finish to implement tone mapping\n");
}

void PhotographicLocalToneMapper::_localOperator(cv::Mat lm, cv::Mat &lsmax) {
    lsmax = lm.clone();
    int width = lm.cols;
    int height = lm.rows;
    int kernelNumber = (_maxKernelSize-1) / 2 + 1;
	
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
    for (int n = 0; n < kernelNumber - 1; n++) {
        int s = 1 + 2 * n;
        cv::Mat up = lblur.at(n) - lblur.at(n + 1);
        cv::Mat down = powf(2.0f, _phi) * _alpha / powf(float(s), 2.0f) + lblur.at(n);
        cv::divide(up, down, vs);
        vs = cv::abs(vs);

        // check if vs < epsilon
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                if (vs.at<float>(j, i) < _epsilon) {
                    if (index.at<uchar>(j, i) == 0) {
                        cv::Mat ls = lblur.at(n);
                        lsmax.at<float>(j, i) = ls.at<float>(j, i);
                    }
                }
                else {
                    index.at<uchar>(j, i) = 1;
                }
            }
        }
    }
}

} // namespace shdr