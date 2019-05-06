#pragma once

#include "crfSolver/debevecMethod.h"
#include "toneMapper/bilateral.h"
#include "toneMapper/photographicGlobal.h"
#include "toneMapper/photographicLocal.h"

namespace shdr {

class HDRSolver {
public:
	HDRSolver(std::string imageDirectory, std::string shutterFilename, 
              std::string crfSolver = "debevecMethod", std::string toneMapper = "photographicLocal");

	void solve(cv::Mat &dst);

private:
	void _readData(std::string imageDirectory, std::string shutterFilename);
	void _alignMTB();
	void _calculateBitmap(cv::Mat image, std::vector<cv::Mat> &mtb, std::vector<cv::Mat> &eb);
	int _findMedian(cv::Mat image);

	std::unique_ptr<CRFSolver> _crfSolver;
	std::unique_ptr<ToneMapper> _toneMapper;

	std::vector<cv::Mat> _images;
	std::vector<float> _shutterSpeeds;

	int _maxMTBLevel;
};

inline cv::Mat GetTranslationMatrix(int tx, int ty) {
	cv::Mat m = cv::Mat::zeros(cv::Size(3, 2), CV_32FC1);
	m.at<float>(0, 0) = 1.0f; 
	m.at<float>(0, 2) = float(tx);
	m.at<float>(1, 1) = 1.0f;
	m.at<float>(1, 2) = float(ty);

	return m;
}

} // namespace shdr