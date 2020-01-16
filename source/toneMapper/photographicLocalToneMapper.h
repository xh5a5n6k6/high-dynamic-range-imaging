#pragma once

#include "core/toneMapper.h"

namespace shdr {

class PhotographicLocalToneMapper : public ToneMapper {
public:
    PhotographicLocalToneMapper();
    PhotographicLocalToneMapper(const float alpha,
                                const float delta,
                                const float phi,
                                const float epsilon,
                                const int   maxKernelSize);

    void solve(const cv::Mat& hdri, 
               cv::Mat* const out_ldri) const override;

private:
    void _localOperator(const cv::Mat& lm, cv::Mat* const out_lsmax) const;

    float _alpha;
    float _delta; 
    float _phi; 
    float _epsilon;
    int   _maxKernelSize;
};

} // namespace shdr