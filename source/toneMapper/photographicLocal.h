#pragma once

#include "core/toneMapper.h"

namespace shdr {

class PhotographicLocalToneMapper : public ToneMapper {
public:
	PhotographicLocalToneMapper();
	void solve(cv::Mat hdri, cv::Mat &ldri);

	float _delta, _alpha;
	float _phi, _epsilon;
	int _maxKernelSize;

private:
	void _localOperator(cv::Mat lm, cv::Mat &lsmax);
};

} // namespace shdr