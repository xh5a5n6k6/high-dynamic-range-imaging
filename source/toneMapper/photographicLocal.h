#pragma once

#include "core/toneMapper.h"

namespace shdr {

class PhotographicLocalToneMapper : public ToneMapper {
public:
	PhotographicLocalToneMapper();
	void solve(cv::Mat hdri, cv::Mat &ldri);

private:
	void _localOperator(cv::Mat lm, cv::Mat &lsmax);

	float _delta, _alpha, _phi, _epsilon;
	int _maxKernelSize;
};

} // namespace shdr