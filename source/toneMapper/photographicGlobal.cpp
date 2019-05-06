#include "toneMapper/photographicGlobal.h"

namespace shdr {

PhotographicGlobalToneMapper::PhotographicGlobalToneMapper() :
	_alpha(0.7f), _delta(0.000001f) {
}

void PhotographicGlobalToneMapper::solve(cv::Mat hdri, cv::Mat &ldri) {
	fprintf(stderr, "# Begin to implement tone mapping using photographic global method\n");

	ldri = hdri.clone();
	cv::Mat lw, logLw, lm, ld;
	cv::cvtColor(hdri, lw, cv::COLOR_BGR2GRAY);
	cv::log(lw + _delta, logLw);
	float meanLogLw = float(cv::mean(logLw)[0]);
	float meanLw = exp(meanLogLw);
	float invMeanLw = 1.0f / meanLw;
	lm = _alpha * invMeanLw * lw;

	double min, max;
	cv::minMaxLoc(lm, &min, &max);

	float lWhite = float(max);
	cv::Mat up = 1.0f + lm / (lWhite * lWhite);
	cv::Mat down = 1.0f + lm;
	cv::divide(up, down, ld);
	ld = ld.mul(lm);

	/*
		calculate each channel
	*/
	std::vector<cv::Mat> vecMat;
	cv::split(ldri, vecMat);
	for (int c = 0; c < 3; c++) {
		cv::divide(vecMat.at(c), lw, vecMat.at(c));
		vecMat.at(c) = vecMat.at(c).mul(ld);
	}
	cv::merge(vecMat, ldri);
	ldri *= 255.0f;
	ldri.convertTo(ldri, CV_8UC3);

	fprintf(stderr, "# Finish to implement tone mapping\n");
}

} // namespace shdr