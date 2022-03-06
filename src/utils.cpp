
#include "utils.hpp"

std::string to_string(const std::wstring &wstr)
{
  unsigned len = wstr.size() * 4;
  setlocale(LC_CTYPE, "");
  char *p = new char[len];
  wcstombs(p, wstr.c_str(), len);
  std::string str(p);
  delete[] p;
  return str;
}

std::wstring to_wstring(const std::string &str)
{
  unsigned len = str.size() * 2;
  setlocale(LC_CTYPE, "");
  wchar_t *p = new wchar_t[len];
  mbstowcs(p, str.c_str(), len);
  std::wstring wstr(p);
  delete[] p;
  return wstr;
}

Ort::Value create_tensor(const cv::Mat &mat, const std::vector<int64_t> &tensor_dims, 
                         const Ort::MemoryInfo &memory_info_handler, 
                         std::vector<float>& tensor_value_handler) 
throw(std::runtime_error)
{
  const unsigned int rows = mat.rows;
  const unsigned int cols = mat.cols;
  const unsigned int channels = mat.channels();

  if (tensor_dims.size() != 4) throw std::runtime_error("dims mismatch.");
  if (tensor_dims.at(0) != 1) throw std::runtime_error("batch != 1");
  
  const unsigned int target_channel = tensor_dims.at(1);
  const unsigned int target_height = tensor_dims.at(2);
  const unsigned int target_width = tensor_dims.at(3);
  const unsigned int target_tensor_size = target_channel * target_height * target_width;
  if (target_channel != channels) throw std::runtime_error("channel mismatch.");
  if (target_height != rows) throw std::runtime_error("height mismatch.");
  if (target_width != cols) throw std::runtime_error("width mismatch.");

  tensor_value_handler.resize(target_tensor_size);

  std::vector<cv::Mat> mat_channels;
  cv::split(mat, mat_channels);
  for(unsigned int i = 0; i < channels; ++i){
    std::memcpy(tensor_value_handler.data() + i * (target_height * target_width),
                mat_channels.at(i).data, target_height * target_width * sizeof(float));
  }
  return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(), 
                                         target_tensor_size, tensor_dims.data(), 
                                         tensor_dims.size());
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

int ReadDict(const std::string &path, std::vector<std::string> &m_vec){
  std::ifstream in(path);
  std::string line;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(line);
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