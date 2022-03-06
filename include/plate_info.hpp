#pragma once
#include "vector"

typedef struct _DetBoxTheta {
    float x1;
    float y1;
    float x2;
    float y2;
    float x3;
    float y3;
    float x4;
    float y4;
    float theta;
} DetBoxTheta ;

typedef struct _DetectionResult {
    std::vector<DetBoxTheta> boxthetas;
} DetectionResult;

struct PlateRecognizePrivate;

typedef struct _RegScore {
    std::vector<std::string> reg_out;
    float score;
} RegScore;

typedef struct _RecognizeResult {
    std::vector<RegScore> RegOuts;
}RecognizeResult;