#include "core/hdrSolver.h"

#include "crfSolver/debevecCrfSolver.h"
#include "mathUtils.h"
#include "toneMapper/bilateral.h"
#include "toneMapper/photographicGlobal.h"
#include "toneMapper/photographicLocal.h"

#include <cstdio>
#include <iostream>
#include <limits>

#if (defined(_MSC_VER) || \
     (defined(__GNUC__) && (__GNUC_MAJOR__ >= 8))) 
    #include <filesystem>
    namespace std_fs = std::filesystem;
#else
    #include <experimental/filesystem>
    namespace std_fs = std::experimental::filesystem;
#endif

namespace shdr {

HdrSolver::HdrSolver(const std::string& imageDirectory, 
                     const std::string& shutterFilename, 
                     const std::string& crfSolver, 
                     const std::string& toneMapper) {

    _readData(imageDirectory, shutterFilename);

    // decide which crfSolver to use
    if (crfSolver == "debevec") {
        _crfSolver = std::make_unique<DebevecCrfSolver>(DwfType::D_GAUSSIAN, 50, 40.0f);
    }
    else {
        std::cout << "Unknown crfSolver type, use debevec instead"
                  << std::endl;
        
        _crfSolver = std::make_unique<DebevecCrfSolver>(DwfType::D_GAUSSIAN, 50, 40.0f);
    }


    // decide which toneMapper to use
    if (toneMapper == "photographic-global") {
        _toneMapper = std::make_unique<PhotographicGlobalToneMapper>();
    }
    else if (toneMapper == "photographic-local") {
        _toneMapper = std::make_unique<PhotographicLocalToneMapper>();
    }
    else if (toneMapper == "bilateral") {
        _toneMapper = std::make_unique<BilateralToneMapper>();
    }
    else {
        std::cout << "Unknown toneMapper type, use photographic-local instead"
                  << std::endl;

        _toneMapper = std::make_unique<PhotographicLocalToneMapper>();
    }
}

HdrSolver::~HdrSolver() = default;

void HdrSolver::solve(cv::Mat* const out_hdri) {
    cv::Mat hdr;
    cv::Mat hdr_toneMapping;

    _alignMTB();
    _crfSolver->solve(_images, _shutterSpeeds, &hdr);
    _toneMapper->solve(hdr, &hdr_toneMapping);

    *out_hdri = hdr_toneMapping;
}

void HdrSolver::_readData(const std::string& imageDirectory, const std::string& shutterFilename) {
    /*
        First, we read shutter times from a file,
        and calculate its size
    */
    FILE *f = fopen(shutterFilename.c_str(), "r");
    if (!f) {
        std::cout << "Shutter times file can't open !"
                  << std::endl;

        exit(0);
    }

    char line[1024];
    while (fgets(line, 1024, f)) {
        if (line[strlen(line)] == '\n') {
            line[strlen(line) - 1] = '\0';
        }

        const float time = static_cast<float>(std::stold(line));
        _shutterSpeeds.push_back(time);
    }

    /*
        Second, we read image data from files
    */
    for (const auto& entry : std_fs::directory_iterator(imageDirectory)) {
        const cv::Mat img = cv::imread(entry.path().string());
        _images.push_back(img);
    }

    std::cout << "# Total read " << _images.size() << " images"
              << std::endl;
}

void HdrSolver::_alignMTB() {
    std::cout << "# Begin to align images using MTB method"
              << std::endl;

    const int numImages = static_cast<int>(_images.size());
    const int middle    = numImages / 2;

    std::cout << "    Using image " << (middle + 1) << " as center image"
              << std::endl;

    /*
        mainMtb means main median threshold bitmap
        mainEb means main exclusive bitmap
    */
    std::vector<cv::Mat> mainVecMtb;
    std::vector<cv::Mat> mainVecEb;
    _calculateBitmap(_images[middle], &mainVecMtb, &mainVecEb);

    /*
        for each image find its best offset that is the closest offset to main image
    */
    const int dx[9] = { -1, 0, 1, -1, 0, 1, -1,  0,  1 };
    const int dy[9] = {  1, 1, 1,  0, 0, 0, -1, -1, -1 };
    for (int n = 0; n < numImages; ++n) {
        if (n != middle) {
            std::vector<cv::Mat> tmpVecMtb;
            std::vector<cv::Mat> tmpVecEb;
            _calculateBitmap(_images[n], &tmpVecMtb, &tmpVecEb);

            /*
                trace each level of MTB & EB
            */
            int offsetX = 0; 
            int offsetY = 0;
            for (int level = 0; level < _maxMtbLevel; ++level) {
                const cv::Mat& nowMtb = tmpVecMtb[_maxMtbLevel - level - 1];
                const cv::Mat& nowEb  = tmpVecEb[_maxMtbLevel - level - 1];
                
                const int width  = nowMtb.cols;
                const int height = nowMtb.rows;

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
                float maxError = std::numeric_limits<float>::max();
                int dir;
                for (int idx = 0; idx < 9; ++idx) {
                    cv::Mat translationMatrix;
                    mathUtils::getTranslationMatrix(offsetX + dx[idx], offsetY + dy[idx], &translationMatrix);

                    cv::Mat tmpMtb;
                    cv::Mat tmpEb;
                    cv::warpAffine(nowMtb, tmpMtb, translationMatrix, nowMtb.size());
                    cv::warpAffine(nowEb, tmpEb, translationMatrix, nowMtb.size());

                    /*
                        use XOR to calculate difference pixel value
                            XOR(A, B) = abs(A-B)
                        use AND to filter value that is near median value
                            AND(A, B) = A.mul(B)
                    */
                    cv::Mat XOR;
                    cv::Mat AND;
                    cv::bitwise_xor(mainVecMtb[_maxMtbLevel - level - 1], tmpMtb, XOR);
                    cv::bitwise_and(XOR, mainVecEb[_maxMtbLevel - level - 1], AND);
                    cv::bitwise_and(AND, tmpEb, AND);

                    const float error = static_cast<float>(cv::sum(AND)[0]);
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
            cv::Mat bestTranslation; 
            mathUtils::getTranslationMatrix(offsetX, offsetY, &bestTranslation);
            
            cv::warpAffine(_images[n], _images[n], bestTranslation, _images[n].size());

            std::cout << "    Image " << (n + 1)
                      << " max offset: x = " << offsetX << ", y = " << offsetY
                      << std::endl;
        }
    }

    std::cout << "# Finish aligning images"
              << std::endl;
}

void HdrSolver::_calculateBitmap(const cv::Mat&              image, 
                                 std::vector<cv::Mat>* const out_vecMtb, 
                                 std::vector<cv::Mat>* const out_vecEb) const {
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    for (int level = 0; level < _maxMtbLevel; ++level) {
        const int median = _findMedian(grayImage);
        const int width  = grayImage.cols;
        const int height = grayImage.rows;

        cv::Mat mtb = cv::Mat::zeros(grayImage.size(), CV_8UC1);
        cv::Mat eb  = cv::Mat::zeros(grayImage.size(), CV_8UC1);

        /*
            use median to be the threshold,
            and check if pixel value is near median
        */
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                mtb.at<uchar>(j, i) = (grayImage.at<uchar>(j, i) <= median) ? 0 : 1;
                eb.at<uchar>(j, i)  = (grayImage.at<uchar>(j, i) < median - 4 ||
                                       grayImage.at<uchar>(j, i) > median + 4 )? 1 : 0;
            }
        }

        out_vecMtb->push_back(mtb);
        out_vecEb->push_back(eb);

        cv::resize(grayImage, grayImage, cv::Size(width / 2, height / 2));
    }
}

int HdrSolver::_findMedian(const cv::Mat& image) const {
    const int width  = image.cols;
    const int height = image.rows;
    const int middle = (width * height + 1) / 2;

    /*
        First we calculate the histogram of image
    */
    int hist[256] = { 0 };
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            hist[image.at<uchar>(j, i)] += 1;
        }
    }
		
    /*
        Second we find cdf that its value is higher than middle (middle means half pixel number)
    */
    int sum = 0;
    for (int i = 0; i < 256; ++i) {
        sum += hist[i];
        if (sum >= middle) {
            return i;
        }
    }

    std::cerr << "There is a fatal error, it can't find median number"
              << std::endl;
    
    return 0;
}

} // namespace shdr