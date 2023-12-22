# DeDoDe-onnxrun-cpp-py
使用ONNXRuntime部署DeDoDe："局部特征匹配：检测，不要描述——描述，不要检测"。依然是C++和Python两个版本的程序

训练源码地址是：https://github.com/Parskatt/DeDoDe

onnx文件在百度云盘，链接：https://pan.baidu.com/s/1q2aDXmpCLE__15xyW1IuJw 
提取码：okqw

由于本套程序是端到端的关键点检测和匹配的，因此在加载onnx文件时，需要选择weights文件夹里的以dedode_end2end开头的onnx文件。
检测，描述，分开的onnx文件也在weights文件夹，有兴趣的开发者，可以编写多个阶段的程序。


用深度学习做图像特征匹配，经典的算法模型有superpoint+superglue，lightglue，不过算法模型的发布时间，
DeDoDe比前两个都要新。
