#include "detect.hpp"

typedef struct ResizeInfo{
    float ratio_h;
    float ratio_w;
}ResizeInfo;

typedef struct PlateDetectorPrivate{
    PlateDetectorPrivate()
        : img_size(640), threshold(0), nms_conf(0) {
    }
    int img_size;
    float threshold;
    float nms_conf;
}PlateDetectorPrivate;

Detect::Detect()
{
    priv = new PlateDetectorPrivate();
};

Detect::~Detect() {
    if (priv)
        delete priv;
    priv = nullptr;
}

int Detect::InitModel(const std::string& config_file)
{
    cv::FileStorage config(config_file, cv::FileStorage::READ);
    priv->img_size = (int)config["IMG_SIZE"];
    priv->nms_conf = (float)config["NMS"];
    priv->threshold = (float)config["THRESHOLD"];
    std::string detect_model_path = (std::string)config["DETECT_MODEL_PATH"];
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
    }
    else {
        device_type = torch::kCPU;
    }
    Init(detect_model_path, device_type);
    return 0;
}

int Detect::resize_with_pad(const cv::Mat &img, cv::Mat &img_out, ResizeInfo &resize_info, 
                            const cv::Size& new_shape, const std::vector<int>& color, 
                            int interp)
{
    const unsigned int org_w = img.cols;
    const unsigned int org_h = img.rows;
    const unsigned int new_shape_w = new_shape.width;
    const unsigned int new_shape_h = new_shape.height;

    float r = ((float)((std::max)(new_shape_w, new_shape_h))) / (std::max)(org_w, org_h);
    int new_unpad_w =  static_cast<int>(std::round(org_w * r));
    int new_unpad_h =  static_cast<int>(std::round(org_h * r));

    if(new_unpad_w % 32 != 0)
    {
        new_unpad_w = (new_unpad_w / 32 + 1) * 32;
    }

    if(new_unpad_h % 32 != 0)
    {
        new_unpad_h = (new_unpad_h / 32 + 1) * 32;
    }

    resize_info.ratio_h = new_unpad_h / ((float) org_h);
    resize_info.ratio_w = new_unpad_w / ((float) org_w);

    cv::Mat resized_image;
    cv::resize(img, resized_image, cv::Size(new_unpad_w, new_unpad_h), 0.0, 0.0, interp);

    int dw = new_shape_w - new_unpad_w;
    int dh = new_shape_h - new_unpad_h;

    cv::Scalar padding_clr;
    if (img.channels() == 1) {
        padding_clr = cv::Scalar(color[0]);
    } else if (img.channels() == 3) {
        padding_clr = cv::Scalar(color[0], color[1], color[2]);
    } else if (img.channels() == 4) {
        padding_clr = cv::Scalar(color[0], color[1], color[2], color[3]);
    }

    cv::copyMakeBorder(resized_image, img_out, 0, dh, 0, dw, cv::BORDER_CONSTANT, padding_clr);
    cv::cvtColor(img_out, img_out, cv::COLOR_BGR2RGB);
    return 0;
}

std::vector<torch::jit::IValue> Detect::transform(const cv::Mat &mat_rs){
    cv::Mat mat_resize = mat_rs;
    bool is_scale_ = true;
    Normalize(&mat_resize, is_scale_);
    return create_tensor(mat_resize, device_, half_);
}

int Detect::get_ids(const float* heatmap, int h, int w, float thresh, std::vector<int>& ids)
{
    for(int i = 0; i < h; i++)
    {
        for(int j = 0; j < w; j++)
        {
            if(heatmap[i*w + j] > thresh)
            {
                ids.push_back(i);
                ids.push_back(j);
            }
        }
    }
    return 0;
}

int Detect::get_recognize_data(const std::vector<std::vector<float>>& nms_out, 
                               std::vector<std::vector<float>>& detect_out, 
                               float ratio_h, float ratio_w)
{
    float pred_sin = 0;
    float pred_cos = 0;
    for(int i = 0; i< nms_out.size(); i++)
    {
        std::vector<float> per_detect_out;
        for (int j=0; j < 8; j++)
        {
            per_detect_out.push_back(nms_out[i][j]);
        }
        pred_sin = nms_out[i][9];
        pred_cos = nms_out[i][10];
        float theta = std::atan(pred_sin / (pred_cos + 0.000001));
        theta = theta*180/_M_PI;
        if((pred_sin >=0) & (pred_cos < 0))
        {
            theta = 180 + theta;
        }
        if((pred_sin < 0) & (pred_cos >= 0))
        {
            theta = 360 + theta;
        }
        if((pred_sin < 0) & (pred_cos < 0)){
            theta = 180 + theta;
        }
        per_detect_out.push_back(theta);
        detect_out.emplace_back(per_detect_out);
    }
    return 0;
}

int Detect::postprocess(const torch::Tensor& detections,
                        float prob_threshold,
                        float nms_thresh,
                        float ratio_h,
                        float ratio_w,
                        DetectionResult &det_result)
{
    const float *heatmap_ = detections.data_ptr<float>();
    int fea_h = detections.size(2);
    int fea_w = detections.size(3);
    int spacial_size = fea_w * fea_h;
    const float *x0 = heatmap_ + spacial_size;
    const float *y0 = x0 + spacial_size;
    const float *x1 = y0 + spacial_size;
    const float *y1 = x1 + spacial_size;
    const float *x2 = y1 + spacial_size;
    const float *y2 = x2 + spacial_size;
    const float *x3 = y2 + spacial_size;
    const float *y3 = x3 + spacial_size;
    const float *sin_theta = y3 + spacial_size;
    const float *cos_theta = sin_theta + spacial_size;

    std::vector<int> ids;
    this->get_ids(heatmap_, fea_h, fea_w, prob_threshold, ids);

    std::vector<std::vector<float>> box_score_thetas;
    for (int i = 0; i < ids.size() / 2; i++) {
        int id_h = ids[2 * i];
        int id_w = ids[2 * i + 1];
        int index = id_h*fea_w + id_w;
        float pp_x = (id_w + 0.5) * 4.0;
        float pp_y = (id_h + 0.5) * 4.0;
        float delta_x0 = x0[index];
        float delta_y0 = y0[index];
        float delta_x1 = x1[index];
        float delta_y1 = y1[index];
        float delta_x2 = x2[index];
        float delta_y2 = y2[index];
        float delta_x3 = x3[index];
        float delta_y3 = y3[index];
        float d_score = heatmap_[index];
        float d_sin_theta = sin_theta[index];
        float d_cos_theta = cos_theta[index];

        std::vector<float> box_score_theta;
        box_score_theta.push_back((pp_x - delta_x0 + 0.5) / ratio_w);
        box_score_theta.push_back((pp_y - delta_y0 + 0.5) / ratio_h);
        box_score_theta.push_back((pp_x - delta_x1 + 0.5) / ratio_w);
        box_score_theta.push_back((pp_y - delta_y1 + 0.5) / ratio_h);
        box_score_theta.push_back((pp_x - delta_x2 + 0.5) / ratio_w);
        box_score_theta.push_back((pp_y - delta_y2 + 0.5) / ratio_h);
        box_score_theta.push_back((pp_x - delta_x3 + 0.5) / ratio_w);
        box_score_theta.push_back((pp_y - delta_y3 + 0.5) / ratio_h);
        box_score_theta.push_back(d_score);
        box_score_theta.push_back(d_sin_theta);
        box_score_theta.push_back(d_cos_theta);
        box_score_thetas.push_back(box_score_theta);        
    }

    int N_Box = box_score_thetas.size();

    if (N_Box)
    {
        float* addata = new float[N_Box * 11];
        int ad_index;
        for (int vvi = 0; vvi < box_score_thetas.size(); vvi++)
        {
            for (int vvj = 0; vvj < 8; vvj++)
            {
                ad_index = vvi * box_score_thetas[0].size() + vvj;
                addata[ad_index] = box_score_thetas[vvi][vvj];
            }
            for (int vvj = 8; vvj < 11; vvj++)
            {
                ad_index = vvi * box_score_thetas[0].size() + vvj;
                addata[ad_index] = box_score_thetas[vvi][vvj];
            }
        }

        std::vector<std::vector<float>> m_data = lanms_adaptor::merge_quadrangle_n9(addata, N_Box, nms_thresh);
        delete[] addata;
        addata = nullptr;
        std::vector<std::vector<float>> detect_out;
        this->get_recognize_data(m_data, detect_out, ratio_h, ratio_w);

        for (int di = 0; di < detect_out.size(); di++)
        {
            DetBoxTheta per_box_theta;
            per_box_theta.x1 = detect_out[di][0];
            per_box_theta.y1 = detect_out[di][1];

            per_box_theta.x2 = detect_out[di][2];
            per_box_theta.y2 = detect_out[di][3];

            per_box_theta.x3 = detect_out[di][4];
            per_box_theta.y3 = detect_out[di][5];

            per_box_theta.x4 = detect_out[di][6];
            per_box_theta.y4 = detect_out[di][7];

            per_box_theta.theta = detect_out[di][8];
            det_result.boxthetas.push_back(per_box_theta);
        }
    }
    return 0;
}

int Detect::run(const cv::Mat &mat, DetectionResult& detect_result){
    torch::NoGradGuard no_grad;
    cv::Mat mat_resize;
    ResizeInfo resize_info;
    this->resize_with_pad(mat, mat_resize, resize_info, 
                          cv::Size(priv->img_size, priv->img_size), 
                          std::vector<int> {0, 0, 0}, cv::INTER_NEAREST);
    std::vector<torch::jit::IValue> input_tensor = this->transform(mat_resize);
    auto output_tensors = module_->forward(
        { input_tensor[0] }
    );
    torch::Tensor detections;
    detections = output_tensors.toTensor();
    detections = detections.to(torch::kFloat32).cpu();
    this->postprocess(detections, priv->threshold,
                      priv->nms_conf, resize_info.ratio_h,
                      resize_info.ratio_w, detect_result);
    return 0;
}