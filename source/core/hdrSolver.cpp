#include "hdrSolver.h"

#include <filesystem>

namespace shdr {

HDRSolver::HDRSolver(std::string imageDirectory, std::string shutterFilename, std::string crfSolver, std::string toneMapper) : 
	_maxMTBLevel(5) {

	_readData(imageDirectory, shutterFilename);

	// decide which crfSolver to use
	if (crfSolver.compare("debevecMethod") == 0)
		_crfSolver = std::make_unique<DebevecMethod>(D_GAUSSIAN, 50, 40.0f);
	else
		fprintf(stderr, "Unsupported crf solver\n");


	// decide which toneMapper to use
	if (toneMapper.compare("photographicGlobal") == 0)
		_toneMapper = new PhotographicGlobalToneMapper();
	else if (toneMapper.compare("photographicLocal") == 0)
		_toneMapper = new PhotographicLocalToneMapper();
	else if (toneMapper.compare("bilateral") == 0)
		_toneMapper = new BilateralToneMapper();
	else
		_toneMapper = new PhotographicLocalToneMapper();
}

HDRSolver::~HDRSolver() {
	delete _toneMapper;
}

void HDRSolver::solve(cv::Mat &dst) {
	cv::Mat hdr;
	_alignMTB();
	_crfSolver->solve(_images, _shutterSpeeds, hdr);
	_toneMapper->solve(hdr, dst);

	cv::Mat tmp;
	std::unique_ptr<ToneMapper> global = std::make_unique<PhotographicGlobalToneMapper>();
	global->solve(hdr, tmp);
	std::unique_ptr<ToneMapper> bilateral = std::make_unique<BilateralToneMapper>();
	bilateral->solve(hdr, tmp);
}

void HDRSolver::_readData(std::string imageDirectory, std::string shutterFilename) {
	/*
		First, we read shutter times from a file,
		and calculate its size
	*/
	FILE *f;
	errno_t err;

	if ((err = fopen_s(&f, shutterFilename.c_str(), "r")) != 0) {
		fprintf(stderr, "Shutter times file can't open !\n");
		exit(0);
	}
	char line[1024];
	while (fgets(line, 1024, f)) {
		if (line[strlen(line)] == '\n')
			line[strlen(line) - 1] = '\0';

		float time = std::stof(line);
		_shutterSpeeds.push_back(time);
	}

	/*
		Second, we read image data from files
	*/
	for (const auto & entry : std::filesystem::directory_iterator(imageDirectory)) {
		cv::Mat img = cv::imread(entry.path().string());
		_images.push_back(img);
	}
}

void HDRSolver::_alignMTB() {
	fprintf(stderr, "# Begin to align images using MTB method\n");

	int imageNumber = _images.size();
	int middle = imageNumber / 2;

	// mainMTB means main median threshold bitmap
	// mainEB means main exclusive bitmap
	std::vector<cv::Mat> mainVecMTB, mainVecEB;
	_calculateBitmap(_images.at(middle), mainVecMTB, mainVecEB);

	/*
		for each image find its best offset that is the closest offset to main image
	*/
	int dx[9] = { -1, 0, 1, -1, 0, 1, -1,  0,  1 };
	int dy[9] = {  1, 1, 1,  0, 0, 0, -1, -1, -1 };
	for (int n = 0; n < imageNumber; n++) {
		if (n != middle) {
			std::vector<cv::Mat> tmpVecMTB, tmpVecEB;
			_calculateBitmap(_images.at(n), tmpVecMTB, tmpVecEB);

			/*
				trace each level of MTB & EB
			*/
			int offsetX = 0, offsetY = 0;
			for (int level = 0; level < _maxMTBLevel; level++) {
				cv::Mat nowMTB = tmpVecMTB.at(_maxMTBLevel - level - 1);
				cv::Mat nowEB = tmpVecEB.at(_maxMTBLevel - level - 1);
				int width = nowMTB.cols;
				int height = nowMTB.rows;

				/*
					for each level, offset needs to be multiplied by 2
					because image size is also twice than previous one
				*/
				offsetX *= 2;
				offsetY *= 2;

				/*
					test 9 directions,
					find which one has the lowest error
				 */
				float maxError = FLT_MAX;
				int dir;
				for (int idx = 0; idx < 9; idx++) {
					cv::Mat tmpMTB, tmpEB;
					cv::Mat translationMatrix = GetTranslationMatrix(offsetX + dx[idx], offsetY + dy[idx]);
					cv::warpAffine(nowMTB, tmpMTB, translationMatrix, nowMTB.size());
					cv::warpAffine(nowEB, tmpEB, translationMatrix, nowMTB.size());

					/*
						use XOR to calculate difference pixel value
							XOR(A, B) = abs(A-B)
						use AND to filter value that is near median value
							AND(A, B) = A.mul(B)
					*/
					cv::Mat XOR, AND;
					cv::bitwise_xor(mainVecMTB.at(_maxMTBLevel - level - 1), tmpMTB, XOR);
					cv::bitwise_and(XOR, mainVecEB.at(_maxMTBLevel - level - 1), AND);
					cv::bitwise_and(AND, tmpEB, AND);
					//Mat XOR = (mainVecMTB.at(_maxMTBLevel - level - 1) | tmpMTB) & (mainVecMTB.at(_maxMTBLevel - level - 1) != tmpMTB);
					//Mat AND = XOR & mainVecEB.at(_maxMTBLevel - level - 1);
					//AND = AND & tmpEB;

					//Mat XOR = cv::abs(mainVecMTB.at(_maxMTBLevel - level - 1) - tmpMTB);
					//Mat AND = XOR.mul(mainVecEB.at(_maxMTBLevel - level - 1));
					//AND = AND.mul(tmpEB);
					float error = float(cv::sum(AND)[0]);
					if (error < maxError) {
						maxError = error;
						dir = idx;
					}
				}

				offsetX += dx[dir];
				offsetY += dy[dir];
			}

			/*
				After we find the best movement,
				we need to translate image with it
			*/
			fprintf(stderr, "\tImage %d max offset : x = %d, y = %d\n", n+1, offsetX, offsetY);
			cv::Mat bestTranslation = GetTranslationMatrix(offsetX, offsetY);
			cv::warpAffine(_images.at(n), _images.at(n), bestTranslation, _images.at(n).size());
		}
	}

	fprintf(stderr, "# Finish to align images\n");
}

void HDRSolver::_calculateBitmap(cv::Mat image, std::vector<cv::Mat> &vecMTB, std::vector<cv::Mat> &vecEB) {
	cv::Mat grayImage;
	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

	for (int level = 0; level < _maxMTBLevel; level++) {
		int median = _findMedian(grayImage);
		cv::Mat mtb = cv::Mat::zeros(grayImage.size(), CV_8UC1);
		cv::Mat eb = cv::Mat::zeros(grayImage.size(), CV_8UC1);
		int width = grayImage.cols;
		int height = grayImage.rows;

		/*
			use median to be the threshold,
			and check if pixel value is near median
		*/
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				mtb.at<uchar>(j, i) = (grayImage.at<uchar>(j, i) <= median) ? 0 : 1;
				eb.at<uchar>(j, i) = (grayImage.at<uchar>(j, i) < median - 4 ||
									  grayImage.at<uchar>(j, i) > median + 4 )? 1 : 0;
			}
		}
		vecMTB.push_back(mtb);
		vecEB.push_back(eb);

		cv::resize(grayImage, grayImage, cv::Size(width / 2, height / 2));
	}
}

int HDRSolver::_findMedian(cv::Mat image) {
	int width = image.cols;
	int height = image.rows;
	int middle = (width*height + 1) / 2;

	/*
		First we calculate the histogram of image
	*/
	int hist[256] = { 0 };
	for (int j = 0; j < height; j++) 
		for (int i = 0; i < width; i++) 
			hist[image.at<uchar>(j, i)] += 1;
		
	/*
		Second we find cdf that its value is higher than middle (middle means half pixel number)
	*/
	int sum = 0;
	for (int i = 0; i < 256; i++) {
		sum += hist[i];
		if (sum >= middle)
			return i;
	}

	fprintf(stderr, "There is a fatal error, it can't find median number\n");
	return 0;
}

} // namespace shdr