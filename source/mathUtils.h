#pragma once

/*
    It stores some math utilities including 
    constants and functions.
*/

#include <cmath>
#include <opencv2/opencv.hpp>
#include <random>

namespace shdr::mathUtils {

inline constexpr float PI
    = 3.14159265358979323846f;

inline constexpr float TWO_PI
    = 6.28318530717958647692f;

inline constexpr float SQRT_TWO_PI
    = 2.50662827463100050241f;

// 1 / sqrt(2 * pi)
inline constexpr float INV_SQRT_TWO_PI
    = 0.39894228040143267793f;

inline float gaussian(const int x, const float mean, const float sigma) {
    const float invSigma = 1.0f / sigma;
    const float expoBase = (x - mean) * invSigma;
    const float exponent = expoBase * expoBase * -0.5f;
    const float a        = invSigma * INV_SQRT_TWO_PI;

    return a * std::exp(exponent);
}


static std::random_device rd;

inline int nextInt(const int min, const int max) {
    std::default_random_engine generator = std::default_random_engine(rd());

    std::uniform_int_distribution<int> distribution(min, max - 1);

    return distribution(generator);
}

inline void getTranslationMatrix(const int      tx,
                                 const int      ty,
                                 cv::Mat* const out_mat) {

    cv::Mat mat = cv::Mat::zeros(cv::Size(3, 2), CV_32FC1);
    mat.at<float>(0, 0) = 1.0f;
    mat.at<float>(0, 2) = static_cast<float>(tx);
    mat.at<float>(1, 1) = 1.0f;
    mat.at<float>(1, 2) = static_cast<float>(ty);

    *out_mat = mat;
}

} // namespace shdr::mathUtils