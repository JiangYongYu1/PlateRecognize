# 车牌识别
    在window上基于onnxruntime-gpu部署c++版本车牌识别模型

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