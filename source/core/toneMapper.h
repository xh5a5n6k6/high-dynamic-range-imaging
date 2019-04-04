#pragma once

#include "basicHeader.h"

#include <opencv2/opencv.hpp>

namespace shdr {

class ToneMapper {
public:
	virtual void solve(cv::Mat hdri, cv::Mat &ldri) = 0;
};

} // namespace shdr