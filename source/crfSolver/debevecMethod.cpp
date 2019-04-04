#include "debevecMethod.h"

#include <random>

#define _DRAW_HDR_IMAGE_

namespace shdr {

DebevecMethod::DebevecMethod() :
	_sampleNumber(50), _lambda(40.0f) {

	_weight = new float[256];
	for (int x = 0; x < 256; x++) 
		_weight[x] = Gaussian(x, 128.0f, 128.0f);
}

DebevecMethod::DebevecMethod(DebevecWeightedFunctionType type, int sampleNumber, float lambda) :
	_sampleNumber(sampleNumber), _lambda(lambda) {

	_weight = new float[256];
	if (type == D_GAUSSIAN)
		for (int x = 0; x < 256; x++)
			_weight[x] = Gaussian(x, 128.0f, 128.0f);
	else
		for (int x = 0; x < 256; x++)
			_weight[x] = 1.0f / 256.0f;
}

DebevecMethod::~DebevecMethod() {
	delete[] _weight;
}

void DebevecMethod::solve(std::vector<cv::Mat> images, std::vector<float> shutterSpeeds, cv::Mat &hdri) {
	fprintf(stderr, "# Begin to reconstruct CRF using Debevec's method\n");

	int width = images.at(0).cols;
	int height = images.at(0).rows;
	int imageNumber = images.size();
	hdri = cv::Mat::zeros(height, width, CV_32FC3);

	/*
		First, generate sample points
	*/
	int* sampleX = new int[_sampleNumber];
	int* sampleY = new int[_sampleNumber];

	std::random_device rd;
	std::default_random_engine gen = std::default_random_engine(rd());
	std::uniform_int_distribution<int> disX(0, width-1);
	std::uniform_int_distribution<int> disY(0, height-1);

	for (int i = 0; i < _sampleNumber; i++) {
		sampleX[i] = disX(gen);
		sampleY[i] = disY(gen);
	}

	/*
		For each channel, use SVD to solve x, i.e. argmin_x (|Ax-b|^2),
		first 256 elements are what we want, i.e. g(0) ~ g(255)

		logG means camera response function
	*/
	cv::Mat g = cv::Mat::zeros(256, 1, CV_32FC3);
	for (int c = 0; c < 3; c++) {
		cv::Mat A = cv::Mat::zeros(_sampleNumber*imageNumber + 1 + 254, 256 + _sampleNumber, CV_32FC1);
		cv::Mat b = cv::Mat::zeros(_sampleNumber*imageNumber + 1 + 254, 1, CV_32FC1);

		/*
			Start to fill in value in A and b
		*/
		int line = 0;
		for (int n = 0; n < imageNumber; n++) {
			cv::Mat nowImage = images.at(n);
			for (int sample = 0; sample < _sampleNumber; sample++, line++) {
				int z = int(nowImage.at<cv::Vec3b>(sampleY[sample], sampleX[sample])[c]);
				A.at<float>(line, z) = 1 * _weight[z];
				A.at<float>(line, 256 + sample) = -1 * _weight[z];
				
				b.at<float>(line, 0) = log(shutterSpeeds.at(n)) * _weight[z];
			}
		}
		A.at<float>(line, 127) = 1;
		line += 1;
		for (int i = 1; i < 255; i++, line++) {
			A.at<float>(line, i - 1) = _lambda * _weight[i];
			A.at<float>(line, i) = -2 * _lambda * _weight[i];
			A.at<float>(line, i + 1) = _lambda * _weight[i];
		}

		/*
			Use SVD to find invA, then x = invA * x
		*/
		cv::Mat invA;
		cv::invert(A, invA, cv::DECOMP_SVD);
		cv::Mat x = invA * b;

		/*
			First 256 values are what we want, i.e. g(0) ~ g(255)
		*/
		for (int j = 0; j < 256; j++)
			g.at<cv::Vec3f>(j, 0)[c] = x.at<float>(j, 0);
	}

	/*
		Clear sample point buffer
	*/
	delete[] sampleX;
	delete[] sampleY;

	/*
		Begin to construct HDR radiance map (hdri),
		for each channel,  each pixel,
		calculate its weighted radiance sum
	*/
	cv::Mat weightSum = cv::Mat::zeros(height, width, CV_32FC3);
	// number of images
	for (int n = 0; n < imageNumber; n++) {
		cv::Mat nowImage = images.at(n);
		// image y
		for (int j = 0; j < height; j++) {
			// image x
			for (int i = 0; i < width; i++) {
				// three color channel
				for (int c = 0; c < 3; c++) {
					int z = int(nowImage.at<cv::Vec3b>(j, i)[c]);
					float lnE = g.at<cv::Vec3f>(z, 0)[c] - log(shutterSpeeds.at(n));
					hdri.at<cv::Vec3f>(j, i)[c] += _weight[z] * lnE;
					weightSum.at<cv::Vec3f>(j, i)[c] += _weight[z];
				}
			}
		}
		fprintf(stderr, "\tFinish CRF recontruction of image %d\n", n+1);
	}

	cv::divide(hdri, weightSum, hdri);
	cv::exp(hdri, hdri);

	fprintf(stderr, "# Finish to reconstruct CRF\n");

#ifdef _DRAW_HDR_IMAGE_

	_writeHDRImage(hdri);
#endif
}

void DebevecMethod::_writeHDRImage(cv::Mat hdri) {
	cv::imwrite("hdri.hdr", hdri);
}

} // namespace shdr