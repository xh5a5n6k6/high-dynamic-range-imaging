#pragma once

#include "core/toneMapper.h"

namespace shdr {

class PhotographicGlobalToneMapper : public ToneMapper {
public:
	PhotographicGlobalToneMapper();
	void solve(cv::Mat hdri, cv::Mat &ldri);

	float _delta, _alpha;
};

} // namespace shdr