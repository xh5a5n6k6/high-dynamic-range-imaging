#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace shdr {

class CrfSolver;
class ImageAligner;
class ToneMapper;

class HdrSolver {
public:
    HdrSolver(const std::string& imageDirectory, 
              const std::string& shutterFilename,
              const std::string& imageAligner = "mtb",
              const std::string& crfSolver    = "debevec", 
              const std::string& toneMapper   = "bilateral");
    ~HdrSolver();

    void solve(cv::Mat* const out_hdri) const;

private:
    void _readData(const std::string& imageDirectory, const std::string& shutterFilename);

    std::vector<cv::Mat> _images;
    std::vector<float>   _shutterSpeeds;

    std::unique_ptr<ImageAligner> _imageAligner;
    std::unique_ptr<CrfSolver>    _crfSolver;
    std::unique_ptr<ToneMapper>   _toneMapper;
};

} // namespace shdr