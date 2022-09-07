# 车牌识别
    在window上基于onnxruntime-gpu部署c++版本车牌识别模型

+ 基于onnxruntime-gpu的c++部署，支持linux和windows，提供两个检测和两个识别模型，可以任意组合，支持蓝绿车牌
    - 轻量级检测模型，只能检测-45 -> +45 plate/center_text_mbv3.onnx
    - 稍大检测模型，可以检测360°车牌 plate/center_text_640.onnx
    - 轻量级识别模型， 2.8M，未使用rnn和全连接 plate/plate_reg_res.onnx
    - 稍大识别模型， plate/plate_reg_best.onnx 精度比轻量级高1.3个点
+ libtorch的c++部署
    - 只提供轻量级模型的torchscript

+ 轻量级模型，只能检测-45 -> +45
    - 检测plate/center_text_mbv3.onnx
    - 识别plate/plate_reg_res.onnx

+ 大模型，检测360°车牌
    - 检测plate/center_text_640.onnx
    - 识别plate/plate_reg_best.onnx

## 用法
### 编译
```
mkdir build
cd build
cmake ..
```
### 运行
```
./plate_demo ../plate/config_lite.yml ../plate/canvas.png
```
### 精度
+ lite模型精度：检测和识别合起来的精度 ccpd 95.6%
+ big模型精度：检测和识别合起来的精度 ccpd 96.9%