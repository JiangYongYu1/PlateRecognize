
#include "utils.hpp"

std::vector<torch::jit::IValue> create_tensor(const cv::Mat& mat, const torch::Device& device, bool half)
{
  torch::NoGradGuard no_grad;
  const unsigned int rows = mat.rows;
  const unsigned int cols = mat.cols;
  const unsigned int channels = mat.channels();

  auto tensor_img = torch::from_blob(mat.data, { 1, rows, cols, channels }, torch::kFloat32).to(device);
  tensor_img = tensor_img.permute({ 0, 3, 1, 2 }).contiguous();
  if (half) {
      tensor_img = tensor_img.to(torch::kHalf);
  }
  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(tensor_img);
  return inputs;
}

void Normalize(cv::Mat *im, const std::vector<float> &mean, 
              const std::vector<float> &scale, const bool is_scale)
{
    double e = 1.0;
    if (is_scale) {
        e /= 255.0;
    }
    (*im).convertTo(*im, CV_32FC3, e);
    for (int h = 0; h < im->rows; h++) {
        for (int w = 0; w < im->cols; w++) {
        im->at<cv::Vec3f>(h, w)[0] =
            (im->at<cv::Vec3f>(h, w)[0] - mean[0]) * scale[0];
        im->at<cv::Vec3f>(h, w)[1] =
            (im->at<cv::Vec3f>(h, w)[1] - mean[1]) * scale[1];
        im->at<cv::Vec3f>(h, w)[2] =
            (im->at<cv::Vec3f>(h, w)[2] - mean[2]) * scale[2];
        }
    }
}

void Normalize(cv::Mat *im, bool is_scale) {
  double e = 1.0;
  if (is_scale) {
    e /= 255.0;
  }
  (*im).convertTo(*im, CV_32FC3, e);
}

std::string UTF8ToGB(const char* str)
{
    std::string result;
    WCHAR* strSrc;
    LPSTR szRes;

    //获得临时变量的大小
    int i = MultiByteToWideChar(CP_UTF8, 0, str, -1, NULL, 0);
    strSrc = new WCHAR[i + 1];
    MultiByteToWideChar(CP_UTF8, 0, str, -1, strSrc, i);

    //获得临时变量的大小
    i = WideCharToMultiByte(CP_ACP, 0, strSrc, -1, NULL, 0, NULL, NULL);
    szRes = new CHAR[i + 1];
    WideCharToMultiByte(CP_ACP, 0, strSrc, -1, szRes, i, NULL, NULL);

    result = szRes;
    delete[]strSrc;
    delete[]szRes;

    return result;
}

int ReadDict(const std::string &path, std::vector<std::string> &m_vec){
  std::ifstream in(path);
  std::string line;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(UTF8ToGB(line.c_str()));
    }
  } else {
    std::cout << "no such label file: " << path << ", exit the program..."
              << std::endl;
    exit(1);
  }
  return 0;
}

inline double CrossProductZ(const PointF &a, const PointF &b) {
    return a.x * b.y - a.y * b.x;
}

inline double Orientation(const PointF &a, const PointF &b, const PointF &c) {
    return CrossProductZ(a, b) + CrossProductZ(b, c) + CrossProductZ(c, a);
}

void Sort4PointsClockwise(PointF points[4], int point_idx[4]){
    PointF& a = points[0];
    PointF& b = points[1];
    PointF& c = points[2];
    PointF& d = points[3];

    // int point_idx[4] = {0, 1, 2, 3};
    int& a_idx = point_idx[0];
    int& b_idx = point_idx[1];
    int& c_idx = point_idx[2];
    int& d_idx = point_idx[3];

    if (Orientation(a, b, c) < 0.0) {
        // Triangle abc is already clockwise.  Where does d fit?
        if (Orientation(a, c, d) < 0.0) {
            return;           // Cool!
        } else if (Orientation(a, b, d) < 0.0) {
            std::swap(d, c);
            std::swap(d_idx, c_idx);
        } else {
            std::swap(a, d);
            std::swap(a_idx, d_idx);
        }
    } else if (Orientation(a, c, d) < 0.0) {
        // Triangle abc is counterclockwise, i.e. acb is clockwise.
        // Also, acd is clockwise.
        if (Orientation(a, b, d) < 0.0) {
            std::swap(b, c);
            std::swap(b_idx, c_idx);
        } else {
            std::swap(a, b);
            std::swap(a_idx, b_idx);
        }
    } else {
        // Triangle abc is counterclockwise, and acd is counterclockwise.
        // Therefore, abcd is counterclockwise.
        std::swap(a, c);
        std::swap(a_idx, c_idx);
    }
}

void PrintPoints(const char *caption, const PointF points[4]){
    printf("%s: (%f,%f),(%f,%f),(%f,%f),(%f,%f)\n", caption,
        points[0].x, points[0].y, points[1].x, points[1].y,
        points[2].x, points[2].y, points[3].x, points[3].y);
}


size_t SortedBySize(const PointF points[4])
{
    std::vector<int> sum_size_list;
    for(int p_i=0; p_i < 4; p_i++)
    {
        sum_size_list.push_back(points[p_i].x + points[p_i].y);
    }

    std::vector<size_t> idx(sum_size_list.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
            [&sum_size_list](size_t index_1, size_t index_2) 
            { return sum_size_list[index_1] < sum_size_list[index_2];});
    return idx[0];
}

void Sort4PointsAd(PointF points[4], int point_idx[4])
{
    Sort4PointsClockwise(points, point_idx);
    // PrintPoints("points: ", points);
    
    size_t final_start = SortedBySize(points);
    if(final_start)
    {
        int get_order[4];
        get_order[0] = final_start;
        get_order[1] = (final_start + 1) % 4;
        get_order[2] = (final_start + 2) % 4;
        get_order[3] = (final_start + 3) % 4;

        PointF a = points[get_order[0]];
        PointF b = points[get_order[1]];
        PointF c = points[get_order[2]];
        PointF d = points[get_order[3]];

        int a_idx = point_idx[get_order[0]];
        int b_idx = point_idx[get_order[1]];
        int c_idx = point_idx[get_order[2]];
        int d_idx = point_idx[get_order[3]];

        PointF& a1 = points[0];
        PointF& b1 = points[1];
        PointF& c1 = points[2];
        PointF& d1 = points[3];

        // int point_idx[4] = {0, 1, 2, 3};
        int& a_idx1 = point_idx[0];
        int& b_idx1 = point_idx[1];
        int& c_idx1 = point_idx[2];
        int& d_idx1 = point_idx[3];

        a1 = a;
        b1 = b;
        c1 = c;
        d1 = d;

        a_idx1 = a_idx;
        b_idx1 = b_idx;
        c_idx1 = c_idx;
        d_idx1 = d_idx;
    }
}

void ResizeImg(cv::Mat &img, cv::Mat &resize_img, int h_get, int w_get) {
  int w = img.cols;
  int h = img.rows;

  float ratio = float(h) / float(h_get);
  int w_resize = int(float(w)*ratio);

  if (w_resize > w_get)
  {
      cv::resize(img, resize_img, cv::Size(w_get, h_get));
  }
  else
  {
      cv::Mat small_resize_img;
      cv::resize(img, small_resize_img, cv::Size(w_resize, h_get));
      cv::copyMakeBorder(small_resize_img, resize_img, 0, 0, 0, w_get - w_resize, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
  }
}