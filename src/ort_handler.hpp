#pragma once
#include "opencv2/opencv.hpp"
#include "string"
#include <torch/script.h>
#include "torch/torch.h"
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>

// single input & multi outputs. not support for dynamic shape currently.
class BasicOrtHandler {
    protected:
        std::string torch_model_path;
        std::unique_ptr<torch::jit::script::Module> module_;
        torch::Device device_ = torch::kCPU;
        bool half_ = false;

    protected:
        BasicOrtHandler() = default;

        virtual ~BasicOrtHandler();

    protected:
        BasicOrtHandler(const BasicOrtHandler &) = delete;

        BasicOrtHandler(BasicOrtHandler &&) = delete;

        BasicOrtHandler &operator=(const BasicOrtHandler &) = delete;

        BasicOrtHandler &operator=(BasicOrtHandler &&) = delete;

    protected:
        virtual std::vector<torch::jit::IValue> transform(const cv::Mat &mat) = 0;
        int Init(const std::string &torch_model_path, const torch::DeviceType& device_type);
    private:
        void initialize_handler();
};