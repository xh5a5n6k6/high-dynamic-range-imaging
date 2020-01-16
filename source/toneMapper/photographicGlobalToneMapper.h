#pragma once

#include "core/toneMapper.h"

namespace shdr {

class PhotographicGlobalToneMapper : public ToneMapper {
public:
    PhotographicGlobalToneMapper();
    PhotographicGlobalToneMapper(const float alpha, const float delta);

    void solve(const cv::Mat& hdri, 
               cv::Mat* const out_ldri) const override;

private:
    float _alpha;
    float _delta;
};

} // namespace shdr