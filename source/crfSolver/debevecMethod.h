#pragma once

#include "core/crfSolver.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif // M_PI

namespace shdr {

enum DebevecWeightedFunctionType {
	D_GAUSSIAN,
	D_UNIFORM
};

class DebevecMethod : public CRFSolver {
public:
	DebevecMethod();
	DebevecMethod(DebevecWeightedFunctionType type, int sampleNumber, float lambda);
	~DebevecMethod();

	void solve(std::vector<cv::Mat> images, std::vector<float> shutterSpeeds, cv::Mat &hdri) override;

private:
	void _writeHDRImage(cv::Mat hdri);

	float* _weight;
	int _sampleNumber;
	float _lambda;
};

inline float Gaussian(int x, float mean, float sigma) {
	float invA = 1.0f / (sigma * sqrtf(2 * M_PI));
	float expo = pow((x - mean) / sigma, 2) * -1.0f / 2.0f;

	return invA * exp(expo);
}

} // namespace shdr