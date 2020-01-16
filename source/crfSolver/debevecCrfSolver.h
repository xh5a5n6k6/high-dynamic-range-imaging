#pragma once

#include "core/crfSolver.h"

#include <memory>

namespace shdr {

/*
    DwfType: Debevec Weighted Function Type

    It supports two types of weighted function
    using in Debevec's algorithm.
*/
enum class DwfType {
    D_GAUSSIAN,
    D_UNIFORM,
};

class DebevecCrfSolver : public CrfSolver {
public:
    DebevecCrfSolver();
    DebevecCrfSolver(const DwfType& type, 
                     const int      numSamples, 
                     const float    lambda);

    void solve(const std::vector<cv::Mat>& images, 
               const std::vector<float>&   shutterSpeeds, 
               cv::Mat* const              out_hdri) const override;

private:
    void _writeHDRImage(const cv::Mat& hdri) const;

    std::unique_ptr<float[]> _weight;
    int                      _numSamples;
    float                    _lambda;
};

} // namespace shdr