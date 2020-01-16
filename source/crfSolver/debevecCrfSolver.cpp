#include "crfSolver/debevecCrfSolver.h"

#include "mathUtils.h"

#include <iostream>

#define _DRAW_HDR_IMAGE_

namespace shdr {

DebevecCrfSolver::DebevecCrfSolver() :
    DebevecCrfSolver(DwfType::D_GAUSSIAN, 50, 40.0f) {
} 

DebevecCrfSolver::DebevecCrfSolver(const DwfType& type, 
                                   const int      numSamples, 
                                   const float    lambda) :
    _weight(),
    _numSamples(numSamples), 
    _lambda(lambda) {

    _weight.reset(new float[256]);
    if (type == DwfType::D_GAUSSIAN) {
        for (int x = 0; x < 256; ++x) {
            _weight[x] = mathUtils::gaussian(x, 128.0f, 128.0f);
        }
    }
    else if (type == DwfType::D_UNIFORM) {
        for (int x = 0; x < 256; ++x) {
            _weight[x] = 1.0f / 256.0f;
        }
    }
    else {
        std::cout << "Unknown weighted type, use gaussian instead"
                  << std::endl;

        for (int x = 0; x < 256; ++x) {
            _weight[x] = mathUtils::gaussian(x, 128.0f, 128.0f);
        }
    }
}

void DebevecCrfSolver::solve(const std::vector<cv::Mat>& images, 
                             const std::vector<float>&   shutterSpeeds, 
                             cv::Mat* const              out_hdri) const {

    std::cout << "# Begin to reconstruct CRF using Debevec's method"
              << std::endl;

    const int width     = images.at(0).cols;
    const int height    = images.at(0).rows;
    const int numImages = static_cast<int>(images.size());
    
    cv::Mat hdri = cv::Mat::zeros(height, width, CV_32FC3);

    /*
        First, generate sample points
    */
    std::unique_ptr<int[]> sampleX(new int[_numSamples]);
    std::unique_ptr<int[]> sampleY(new int[_numSamples]);

    for (int i = 0; i < _numSamples; ++i) {
        sampleX[i] = mathUtils::nextInt(0, width);
        sampleY[i] = mathUtils::nextInt(0, height);
    }

    /*
        For each channel, use SVD to solve x, i.e. argmin_x (|Ax-b|^2),
        first 256 elements are what we want, i.e. g(0) ~ g(255)

        logG means camera response function
    */
    cv::Mat g = cv::Mat::zeros(256, 1, CV_32FC3);
    for (int c = 0; c < 3; ++c) {
        cv::Mat A = cv::Mat::zeros(_numSamples * numImages + 1 + 254, 
                                   256 + _numSamples, 
                                   CV_32FC1);
        
        cv::Mat b = cv::Mat::zeros(_numSamples * numImages + 1 + 254, 
                                   1, 
                                   CV_32FC1);

        /*
            Start to fill in value in A and b
        */
        int line = 0;
        for (int n = 0; n < numImages; ++n) {
            const cv::Mat& nowImage = images[n];

            for (int sample = 0; sample < _numSamples; ++sample, ++line) {
                const int z = static_cast<int>(
                    nowImage.at<cv::Vec3b>(sampleY[sample], sampleX[sample])[c]);

                A.at<float>(line, z)            = 1.0f * _weight[z];
                A.at<float>(line, 256 + sample) = -1.0f * _weight[z];
				
                b.at<float>(line, 0)            = std::log(shutterSpeeds[n]) * _weight[z];
            }
        }
        A.at<float>(line, 127) = 1.0f;
        ++line;
        for (int i = 1; i < 255; ++i, ++line) {
            A.at<float>(line, i - 1) = _lambda * _weight[i];
            A.at<float>(line, i)     = -2.0f * _lambda * _weight[i];
            A.at<float>(line, i + 1) = _lambda * _weight[i];
        }

        /*
            Use SVD to find invA, then x = invA * x
        */
        cv::Mat invA;
        cv::invert(A, invA, cv::DECOMP_SVD);
        const cv::Mat x = invA * b;

        /*
            First 256 values are what we want, i.e. g(0) ~ g(255)
        */
        for (int j = 0; j < 256; ++j) {
            g.at<cv::Vec3f>(j, 0)[c] = x.at<float>(j, 0);
        }
    }

    /*
        Clear sample point buffer
    */
    sampleX.reset();
    sampleY.reset();

    /*
        Begin to construct HDR radiance map (hdri),
        for each channel,  each pixel,
        calculate its weighted radiance sum
    */
    cv::Mat weightSum = cv::Mat::zeros(height, width, CV_32FC3);
    // number of images
    for (int n = 0; n < numImages; ++n) {
        const cv::Mat& nowImage = images[n];
        // image y
        for (int j = 0; j < height; ++j) {
            // image x
            for (int i = 0; i < width; ++i) {
                // three color channel
                for (int c = 0; c < 3; ++c) {
                    const int   z   = static_cast<int>(nowImage.at<cv::Vec3b>(j, i)[c]);
                    const float lnE = g.at<cv::Vec3f>(z, 0)[c] - std::log(shutterSpeeds[n]);

                    hdri.at<cv::Vec3f>(j, i)[c]      += _weight[z] * lnE;
                    weightSum.at<cv::Vec3f>(j, i)[c] += _weight[z];
                }
            }
        }

        std::cout << "    Finish CRF reconstruction of image " << (n + 1)
                  << std::endl;
    }

    cv::divide(hdri, weightSum, hdri);
    cv::exp(hdri, hdri);

    *out_hdri = hdri;

    std::cout << "# Finish reconstructing CRF"
              << std::endl;

#ifdef _DRAW_HDR_IMAGE_

    _writeHDRImage(hdri);
#endif
}

void DebevecCrfSolver::_writeHDRImage(const cv::Mat& hdri) const {
    cv::imwrite("./hdr.hdr", hdri);
}

} // namespace shdr