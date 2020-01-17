#pragma once

#include "core/toneMapper.h"

namespace shdr {

class BilateralToneMapper : public ToneMapper {
public:
    BilateralToneMapper();
    BilateralToneMapper(const float delta);

    void map(const cv::Mat& hdri, 
             cv::Mat* const out_ldri) const override;

private:
    float _delta;
};

} // namespace shdr