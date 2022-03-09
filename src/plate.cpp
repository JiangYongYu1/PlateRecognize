#include "plate.hpp"
#include "detect.hpp"
#include "recognize.hpp"

typedef struct PlatePriv{
    ~PlatePriv()
    {
        if (detector){
            delete detector;
        }
        detector = nullptr;

        if(recognizor){
            delete recognizor;
        }
        recognizor = nullptr;
    }
    Recognize *recognizor = nullptr;
    Detect *detector = nullptr;
}PlatePriv;


Plate::Plate()
{
    priv = new PlatePriv();
}

Plate::~Plate()
{
    if(priv)
    {
        delete priv;
    }
    priv = nullptr;
}

int Plate::Init(const std::string &config_path)
{
    priv->detector = new Detect();
    priv->recognizor = new Recognize();
    priv->detector->InitModel(config_path);
    priv->recognizor->InitModel(config_path);
    return 0;
}

int Plate::Run(const std::string& img_path, 
                DetectionResult &detect_out, 
                RecognizeResult &reg_out)
{
    cv::Mat cv_image = cv::imread(img_path, cv::IMREAD_COLOR);
    priv->detector->run(cv_image, detect_out);
    priv->recognizor->run(cv_image, detect_out, reg_out);
    return 0;
}

// int Plate::Run(const cv::Mat& img, 
//                DetectionResult &detect_out, 
//                RecognizeResult &reg_out)
// {
//     priv->detector->run(img, detect_out);
//     priv->recognizor->run(img, detect_out, reg_out);
//     return 0;
// }