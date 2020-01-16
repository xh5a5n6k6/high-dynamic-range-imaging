#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace shdr {

class CrfSolver;
class ToneMapper;

class HdrSolver {
public:
    HdrSolver(const std::string& imageDirectory, 
              const std::string& shutterFilename,
              const std::string& crfSolver  = "debevec", 
              const std::string& toneMapper = "bilateral");
    ~HdrSolver();

    void solve(cv::Mat* const out_hdri);

private:
    void _readData(const std::string& imageDirectory, const std::string& shutterFilename);
    void _alignMTB();
    void _calculateBitmap(const cv::Mat&              image, 
                          std::vector<cv::Mat>* const out_vecMtb, 
                          std::vector<cv::Mat>* const out_vecEb) const;
    int  _findMedian(const cv::Mat& image) const;

    std::unique_ptr<CrfSolver>  _crfSolver;
    std::unique_ptr<ToneMapper> _toneMapper;

    std::vector<cv::Mat> _images;
    std::vector<float>   _shutterSpeeds;

    static const int _maxMtbLevel = 5;
};

} // namespace shdr