#include "core/hdrSolver.h"

#include "crfSolver/debevecCrfSolver.h"
#include "imageAligner/mtbImageAligner.h"
#include "toneMapper/bilateralToneMapper.h"
#include "toneMapper/photographicGlobalToneMapper.h"
#include "toneMapper/photographicLocalToneMapper.h"

#include <algorithm>
#include <cstdio>
#include <iostream>

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
                     const std::string& imageAligner,
                     const std::string& crfSolver, 
                     const std::string& toneMapper) :
    _images(),
    _shutterSpeeds(),
    _imageAligner(nullptr),
    _crfSolver(nullptr),
    _toneMapper(nullptr) {

    // decide which imageAligner to use
    if (imageAligner == "mtb") {
        _imageAligner = std::make_unique<MtbImageAligner>();
    }
    else {
        std::cout << "Unknown imageAligner type: <"
                  << imageAligner << ">, use <mtb> instead"
                  << std::endl;

        _imageAligner = std::make_unique<MtbImageAligner>();
    }

    // decide which crfSolver to use
    if (crfSolver == "debevec") {
        _crfSolver = std::make_unique<DebevecCrfSolver>(DwfType::D_GAUSSIAN, 50, 40.0f);
    }
    else {
        std::cout << "Unknown crfSolver type: <"
                  << crfSolver << ">, use <debevec> instead"
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
        std::cout << "Unknown toneMapper type: <"
                  << toneMapper << ">, use <bilateral> instead"
                  << std::endl;

        _toneMapper = std::make_unique<BilateralToneMapper>();
    }

    // read input data (images and shutterspeeds)
    _readData(imageDirectory, shutterFilename);
}

HdrSolver::~HdrSolver() = default;

void HdrSolver::solve(cv::Mat* const out_hdri) const {
    std::vector<cv::Mat> alignImages;
    _imageAligner->align(_images, &alignImages);

    cv::Mat hdri;
    _crfSolver->solve(alignImages, _shutterSpeeds, &hdri);
    
    cv::Mat hdri_toneMapping;
    _toneMapper->map(hdri, &hdri_toneMapping);

    *out_hdri = hdri_toneMapping;
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
    std::cout << "# Begin to read images"
              << std::endl;

    std::vector<std::string> imageFilenames;
    imageFilenames.reserve(_shutterSpeeds.size());
    for (const auto& entry : std_fs::directory_iterator(imageDirectory)) {
        imageFilenames.push_back(entry.path().string());
    }

    // Because filename loading order may be different from standard order,
    // we need to sort it first to make sure its order fits shutterspeed's order.
    std::sort(imageFilenames.begin(), imageFilenames.end());

    for (std::size_t i = 0; i < imageFilenames.size(); ++i) {
        std::cout << "    Image " << (i + 1) << ": " << imageFilenames[i]
                  << std::endl;

        const cv::Mat img = cv::imread(imageFilenames[i]);
        _images.push_back(img);
    }

    std::cout << "# Total read " << _images.size() << " images"
              << std::endl;
}

} // namespace shdr