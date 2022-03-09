#pragma once
#include "iostream"

#include "opencv2/opencv.hpp"

#include "plate_sample.hpp"

struct PlatePriv;
class Plate: public PlateSample{
    public:
        Plate();
        ~Plate();
        int Init(const std::string& config_path);
        int Run(const std::string& img_path, 
                DetectionResult &detect_out, 
                RecognizeResult &reg_out);
        // int Run(const cv::Mat& img, 
        //         DetectionResult &detect_out, 
        //         RecognizeResult &reg_out);
    private:
        PlatePriv *priv = nullptr;
};

