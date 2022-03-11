#include "ort_handler.hpp"

int BasicOrtHandler::Init(
    const std::string& model_path, const torch::DeviceType& device_type)
{
    this->torch_model_path = model_path;
    this->device_ = device_type;
    initialize_handler();
    return 0;
}

void BasicOrtHandler::initialize_handler()
{
    try {
        module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(torch_model_path));
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model!\n";
        std::exit(EXIT_FAILURE);
    }
    half_ = (device_ != torch::kCPU);
    module_->to(device_);

    if (half_) {
        module_->to(torch::kHalf);
    }
    module_->eval();

}

BasicOrtHandler::~BasicOrtHandler()
{
}