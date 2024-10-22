#pragma once
#include "string"
#include "fstream"
#include "numeric"

#include "opencv2/opencv.hpp"
#include "onnxruntime_cxx_api.h"
#include "tensorrt_provider_factory.h"

#include "utils.hpp"

struct PointF {
    float x;
    float y;
};

static constexpr const float _PI = 3.1415926f;

std::wstring to_wstring(const std::string &str);
std::string to_string(const std::wstring &wstr);
/**
 * @brief Create a tensor object
 * 
 * @param mat CV::Mat with type 'CV_32FC3'
 * @param tensor_dims {1,C,H,W}
 * @param memory_info_handler 
 * @param tensor_value_handler 
 * @return Ort::Value 
 */
Ort::Value create_tensor(const cv::Mat &mat, const std::vector<int64_t> &tensor_dims, 
                         const Ort::MemoryInfo &memory_info_handler, 
                         std::vector<float>& tensor_value_handler) throw(std::runtime_error);

void Normalize(cv::Mat *im, const std::vector<float> &mean, 
              const std::vector<float> &scale, const bool is_scale);

void Normalize(cv::Mat *im, bool is_scale);

int ReadDict(const std::string &path, std::vector<std::string> &m_vec);

inline double CrossProductZ(const PointF &a, const PointF &b);
inline double Orientation(const PointF &a, const PointF &b, const PointF &c);
void Sort4PointsClockwise(PointF points[4], int point_idx[4]);
void PrintPoints(const char *caption, const PointF points[4]);
void Sort4PointsAd(PointF points[4], int point_idx[4]);
size_t SortedBySize(const PointF points[4]);
void ResizeImg(cv::Mat &img, cv::Mat &resize_img, int h_get, int w_get);