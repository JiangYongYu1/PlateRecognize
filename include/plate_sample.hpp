#pragma once
#include "iostream"
#include "plate_info.hpp"


class PlateSample{
    public:
        PlateSample() = default;
        virtual ~PlateSample() {};
        virtual int Init(const std::string& config_path) = 0;
        virtual int Run(const std::string& img_path, 
                        DetectionResult &detect_out, 
                        RecognizeResult &reg_out) = 0;
};

