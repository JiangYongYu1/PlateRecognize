#pragma once

#include "opencv2/opencv.hpp"
#include "ort_handler.hpp"

#include "blob_data.h"
#include "plate_info.hpp"
#include "utils.hpp"
#include "transpose.h"

struct PlateRecPivate;

template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

class Recognize: public BasicOrtHandler {
    public:
        Recognize();
        ~Recognize() override;
        int InitModel(const std::string& config_file);
        int run(const cv::Mat &image, 
                const DetectionResult &detection_result, 
                RecognizeResult &RecognizeResult);
    protected:
        std::vector<torch::jit::IValue> transform(const cv::Mat &mat_rs) override;
    private:
        int GetRotateCropImage(const cv::Mat &srcimage, cv::Mat &out_image, 
                               const DetBoxTheta &boxtheta, int aim_h, int aim_w, int im_idx);
        int postprocess(const torch::Tensor &output_tensors,
                        const std::vector<std::string> &labels, 
                        RecognizeResult &re_result);
    private:
        PlateRecPivate *priv;
};