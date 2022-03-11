#pragma once

#include "opencv2/opencv.hpp"

#include "plate_info.hpp"
#include "ort_handler.hpp"
#include "lanms.hpp"
#include "utils.hpp"

struct PlateDetectorPrivate;

struct ResizeInfo;

static constexpr const float _M_PI = 3.1415926f;

class Detect: public BasicOrtHandler {
    public:
        Detect();
        ~Detect() override;

    protected:
        std::vector<torch::jit::IValue> transform(const cv::Mat &mat_rs) override;

    private:
        /**
         * @brief 
         * 
         * @param img cv::Mat type is 8U3C BGR
         * @param img_out cv::Mat type ist 8U3C RGB
         * @param resize_info the resize info of out
         * @param new_shape new shape
         * @param color empty value padding
         * @param interp resize interp method
         * @return int 
         */
        int resize_with_pad(const cv::Mat &img, cv::Mat &img_out, ResizeInfo &resize_info, 
                            const cv::Size& new_shape, const std::vector<int>& color, 
                            int interp);
        int postprocess(const torch::Tensor& detections,
                        float prob_threshold,
                        float nms_thresh,
                        float ratio_h,
                        float ratio_w,
                        DetectionResult &det_result);
        int get_ids(const float* heatmap, int h, int w, float thresh, std::vector<int>& ids);
        int get_recognize_data(const std::vector<std::vector<float>>& nms_out, 
                               std::vector<std::vector<float>>& detect_out, 
                               float ratio_h, float ratio_w);
        PlateDetectorPrivate *priv = nullptr;
    public:
        int InitModel(const std::string& config_file);
        int run(const cv::Mat &mat, DetectionResult& detect_result);
};