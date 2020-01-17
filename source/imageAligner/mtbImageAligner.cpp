#include "imageAligner/mtbImageAligner.h"

#include "mathUtils.h"

#include <limits>

namespace shdr {

MtbImageAligner::MtbImageAligner() = default;

void MtbImageAligner::align(const std::vector<cv::Mat>& images,
                            std::vector<cv::Mat>* const out_alignImages) const {

    std::cout << "# Begin to align images using MTB method"
              << std::endl;

    out_alignImages->reserve(images.size());

    const int numImages = static_cast<int>(images.size());
    const int middle    = numImages / 2;

    std::cout << "    Using image " << (middle + 1) << " as center image"
              << std::endl;

    /*
        mainMtb means main median threshold bitmap
        mainEb means main exclusive bitmap
    */
    std::vector<cv::Mat> mainVecMtb;
    std::vector<cv::Mat> mainVecEb;
    _calculateBitmap(images[middle], &mainVecMtb, &mainVecEb);

    /*
        for each image find its best offset that is the closest offset to main image
    */
    const int dx[9] = { -1, 0, 1, -1, 0, 1, -1,  0,  1 };
    const int dy[9] = {  1, 1, 1,  0, 0, 0, -1, -1, -1 };
    for (int n = 0; n < numImages; ++n) {
        if (n == middle) {
            out_alignImages->push_back(images[n]);
        }

        // only align non-center images
        else {
            std::vector<cv::Mat> tmpVecMtb;
            std::vector<cv::Mat> tmpVecEb;
            _calculateBitmap(images[n], &tmpVecMtb, &tmpVecEb);

            /*
                trace each level of MTB & EB
            */
            int offsetX = 0;
            int offsetY = 0;
            for (int level = 0; level < MAX_MTB_LEVEL; ++level) {
                const cv::Mat& nowMtb = tmpVecMtb[MAX_MTB_LEVEL - level - 1];
                const cv::Mat& nowEb  = tmpVecEb[MAX_MTB_LEVEL - level - 1];

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
                    cv::bitwise_xor(mainVecMtb[MAX_MTB_LEVEL - level - 1], tmpMtb, XOR);
                    cv::bitwise_and(XOR, mainVecEb[MAX_MTB_LEVEL - level - 1], AND);
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

            cv::Mat alignImage;
            cv::warpAffine(images[n], alignImage, bestTranslation, images[n].size());

            out_alignImages->push_back(alignImage);

            std::cout << "    Image " << (n + 1)
                      << " max offset: x = " << offsetX << ", y = " << offsetY
                      << std::endl;
        }
    }

    std::cout << "# Finish aligning images"
              << std::endl;
}

void MtbImageAligner::_calculateBitmap(const cv::Mat&              image,
                                       std::vector<cv::Mat>* const out_vecMtb,
                                       std::vector<cv::Mat>* const out_vecEb) const {

    out_vecMtb->reserve(MAX_MTB_LEVEL);
    out_vecEb->reserve(MAX_MTB_LEVEL);

    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    for (int level = 0; level < MAX_MTB_LEVEL; ++level) {
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
                                       grayImage.at<uchar>(j, i) > median + 4) ? 1 : 0;
            }
        }

        out_vecMtb->push_back(mtb);
        out_vecEb->push_back(eb);

        cv::resize(grayImage, grayImage, cv::Size(width / 2, height / 2));
    }
}

int MtbImageAligner::_findMedian(const cv::Mat& image) const {
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