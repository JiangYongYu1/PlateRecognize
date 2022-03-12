#pragma once
#include "string"
#include "fstream"
#include "numeric"
#include <Windows.h>
#include "opencv2/opencv.hpp"
#include "torch/torch.h"

struct PointF {
    float x;
    float y;
};

static constexpr const float _PI = 3.1415926f;

/**
 * @brief Create a tensor object
 * 
 * @param mat CV::Mat with type 'CV_32FC3'
 * @param tensor_dims {1,C,H,W}
 * @param memory_info_handler 
 * @param tensor_value_handler 
 * @return Ort::Value 
 */
std::vector<torch::jit::IValue> create_tensor(const cv::Mat& mat, const torch::Device& device, bool half);
std::string UTF8ToGB(const char* str);
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