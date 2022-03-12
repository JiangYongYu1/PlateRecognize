#include "recognize.hpp"

typedef struct PlateRecPivate{
    int img_size_h;
    int img_size_w;
    std::vector<std::string> labels;
}PlateRecPivate;

Recognize::Recognize(){
    priv = new PlateRecPivate();
}

Recognize::~Recognize() {
    if(priv){
        delete priv;
    }
    priv = nullptr;
}

int Recognize::InitModel(const std::string& config_file) {
    cv::FileStorage config(config_file, cv::FileStorage::READ);
    priv->img_size_h = (int)config["IMG_SIZE_H"];
    priv->img_size_w = (int)config["IMG_SIZE_W"];
    std::string label_path = (std::string) config["LABEL_PATH"];
    ReadDict(label_path, priv->labels);
    std::string recognize_model_path = (std::string)config["RECOGNIZE_MODEL_PATH"];
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
    }
    else {
        device_type = torch::kCPU;
    }
    Init(recognize_model_path, device_type);
    return 0;
}

int Recognize::GetRotateCropImage(const cv::Mat &srcimage, cv::Mat &out_image, 
                        const DetBoxTheta &boxtheta, int aim_h, int aim_w, int im_idx)
{
    cv::Point2f o_points[4];
    o_points[0] = cv::Point2f(boxtheta.x1, boxtheta.y1);
    o_points[1] = cv::Point2f(boxtheta.x2, boxtheta.y2);
    o_points[2] = cv::Point2f(boxtheta.x3, boxtheta.y3);
    o_points[3] = cv::Point2f(boxtheta.x4, boxtheta.y4);

    float angle = - boxtheta.theta;
    float scale = 1.0;
    cv::Point2d center;
    center.x = (o_points[0].x + o_points[1].x + o_points[2].x + o_points[3].x) / 4.0;
    center.y = (o_points[0].y + o_points[1].y + o_points[2].y + o_points[3].y) / 4.0;
    float x_pad = 1.05;
    float y_pad = 1.0;
    o_points[0].x = center.x + (o_points[0].x - center.x) * x_pad;
    o_points[0].y = center.y + (o_points[0].y - center.y) * y_pad;
    o_points[1].x = center.x + (o_points[1].x - center.x) * x_pad;
    o_points[1].y = center.y + (o_points[1].y - center.y) * y_pad;
    o_points[2].x = center.x + (o_points[2].x - center.x) * x_pad;
    o_points[2].y = center.y + (o_points[2].y - center.y) * y_pad;
    o_points[3].x = center.x + (o_points[3].x - center.x) * x_pad;
    o_points[3].y = center.y + (o_points[3].y - center.y) * y_pad;

    cv::Mat rot_mat(2, 3, CV_32FC1);
    rot_mat = cv::getRotationMatrix2D(center, angle, scale);
    cv::Mat rot_mat_t(3, 2, CV_32FC1);
    rot_mat_t = rot_mat.t();

    float point_array[4][3] = {o_points[0].x, o_points[0].y, 1, 
                               o_points[1].x, o_points[1].y, 1, 
                               o_points[2].x, o_points[2].y, 1, 
                               o_points[3].x, o_points[3].y, 1};

    float rot_points[4][2];
    for (int i=0; i<4; i++)
    {
        for (int j=0; j<2; j++)
         {
            rot_points[i][j] = point_array[i][0]* rot_mat_t.at<double>(0,j)
             + point_array[i][1]* rot_mat_t.at<double>(1,j) 
             + point_array[i][2]* rot_mat_t.at<double>(2,j);
        }
    }
    
    PointF points[4];
    for (int i=0; i<4; i++)
    {
        points[i] = {rot_points[i][0], rot_points[i][1]};
    }

    int sample_idx[4]={0, 1, 2, 3};

    Sort4PointsAd(points, sample_idx);
    int img_crop_width = int((sqrt(pow(points[0].x - points[3].x, 2) + pow(points[0].y - points[3].y, 2)) + sqrt(pow(points[1].x - points[2].x, 2) + pow(points[1].y - points[2].y, 2)))/2.0);
    int img_crop_height = int((sqrt(pow(points[0].x - points[1].x, 2) + pow(points[0].y - points[1].y, 2)) + sqrt(pow(points[2].x - points[3].x, 2) + pow(points[2].y - points[3].y, 2)))/2.0);

    cv::Point2f pts_std[3];
    pts_std[0] = cv::Point2f(0., 0.);
    pts_std[1] = cv::Point2f(img_crop_width, 0.);
    pts_std[2] = cv::Point2f(0., img_crop_height);

    cv::Point2f pointsf[3];
    pointsf[0] = cv::Point2f(o_points[sample_idx[0]].x, o_points[sample_idx[0]].y);
    pointsf[1] = cv::Point2f(o_points[sample_idx[3]].x, o_points[sample_idx[3]].y);
    pointsf[2] = cv::Point2f(o_points[sample_idx[1]].x, o_points[sample_idx[1]].y);

    cv::Mat M = cv::getAffineTransform(pointsf, pts_std);

    cv::Mat dst_img;
    cv::warpAffine(srcimage, dst_img, M,
                   cv::Size(img_crop_width, img_crop_height), cv::INTER_CUBIC, 
                   cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    ResizeImg(dst_img, out_image, aim_h, aim_w);
    bool is_scale_ = true;
    Normalize(&out_image, is_scale_);
    return 0;
}

std::vector<torch::jit::IValue> Recognize::transform(const cv::Mat &mat_rs)
{
    return create_tensor(mat_rs, device_, half_);
}

int Recognize::postprocess(const torch::Tensor &output_tensors,
                           const std::vector<std::string> &labels, 
                           RecognizeResult &re_result)
{
    const float *heatmap_ = (float *)output_tensors.data_ptr();
    int channel = output_tensors.size(1);
    int height = output_tensors.size(2);
    int width = output_tensors.size(3);

    std::shared_ptr<BlobData> bottom_blob(new BlobData(1, channel, height, width));
    std::vector<int> orders = {0, 3, 2, 1};
    std::shared_ptr<BlobData> output_blob(new BlobData(1, width, height, channel));

    float *top_data = output_blob->data();
    float *bottom_data = bottom_blob->data();
    memcpy(bottom_data, heatmap_, channel * height * width * sizeof(float));

    std::vector<int> bottom_shape = {1, channel, height, width}; //reshape
    std::vector<int> top_shape = {1, width, height, channel};

    transpose(bottom_data, bottom_shape, orders, top_data, top_shape);

    std::vector<int> out_index;
    int w = width;
    int c = channel;
    for(int rec_i=0; rec_i < w; rec_i++)
    {
        out_index.push_back(argmax(&top_data[rec_i*c], &top_data[(rec_i + 1)*c]));
    }

    int pp_index;
    int intime_data;
    std::vector<int> output_near;
    for (int rec_j=0; rec_j<out_index.size(); rec_j++)
    {
        if(rec_j == 0)
        {
            pp_index = out_index[rec_j];
            output_near.push_back(pp_index);
        }
        else
        {
            intime_data = out_index[rec_j];
            if(pp_index == intime_data){
                continue;
            }
            else
            {
                output_near.push_back(intime_data);
                pp_index = intime_data;
            }
        }
    }

    std::vector<std::string> output_label;
    for(int i=0; i<output_near.size(); i++)
    {
        // std::cout << output_near[i] << std::endl;
        if(output_near[i] - 1 >= 0)
        {
            output_label.push_back(labels[output_near[i] - 1]);
        }
    }
    RegScore out_box_score;
    out_box_score.reg_out = output_label;
    out_box_score.score = 1.0;
    re_result.RegOuts.push_back(out_box_score);
    return 0;
}

int Recognize::run(const cv::Mat &image, 
                   const DetectionResult &detection_result, 
                   RecognizeResult &RecognizeResult)
{
    auto empty = std::vector<RegScore>();
    RecognizeResult.RegOuts.swap(empty);
    for (int i = detection_result.boxthetas.size() - 1; i >= 0; i--)
    {
        cv::Mat resize_img;
        this->GetRotateCropImage(image, resize_img, detection_result.boxthetas[i], 
                                 priv->img_size_h, priv->img_size_w, i);
        std::vector<torch::jit::IValue> input_tensor = this->transform(resize_img);
        auto output_tensors = this->module_->forward(input_tensor);
        auto outputs = output_tensors.toTensor().to(torch::kFloat).cpu();
        postprocess(outputs, priv->labels, RecognizeResult);
    }
    return 0;
}