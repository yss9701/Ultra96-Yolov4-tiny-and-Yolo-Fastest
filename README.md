# Ultra96 Yolov4-tiny and Yolo-Fastest
1. We convert dataset to VOC format. I use UA-DETRAC dataset, and we can use ./VOCdevkit/ files to convert dataset.

2. In the official yolov4-tiny, there is a slice operation to realize  the CSPnet, but the quantitative tools don't support the operation, so I use a 1*1 convolution to replace it.

3. Then we can use train.py to train the model, and save the model structure and weights as model.json and model.h5. I use TensorFlow-gpu 2.2.0.

4. Then we can generate pb file that is suitable for deployment tools. We can see ./frozon_result/readme.md for details.

5. Then we use Vitis-AI to quantify our model. We can use ./scripts/1_vitisAI_tf_printNode.sh to find the input and output, and use ./scripts/2_vitisAI_tf_quantize.sh to quantify our model.

6. We can compile our model. We can use ./scripts/3_vitisAI_tf_compile.sh to compile our model.

7. We should use vivado and Vitis to build the hardware platform. (./edge/readme.md)

8. The last, we can run our model on Ultra96-v2 board. There is an example that using yolo model to detate vehicles (./edge/dpu_yolo_v4_tiny.ipynb). There are the results, the fps is 25 with 320*320 images.

   ![Alt text] (https://github.com/yss9701/Ultra96-Yolov4-tiny/tree/main/img/1.png)

   ![Alt text] (https://github.com/yss9701/Ultra96-Yolov4-tiny/tree/main/img/2.png)

9. In order to achieve faster detection speed, I try to use Yolo-Fastest ([Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest)) and implement it with tensorflow, then deploy it to Ultra96-v2 board. There are the results, it can achieve 30fps+.

   ![Alt text] (https://github.com/yss9701/Ultra96-Yolov4-tiny/tree/main/img/3.png)

   ![Alt text] (https://github.com/yss9701/Ultra96-Yolov4-tiny/tree/main/img/4.png)

   

   

   References:

   [Yolov4-tiny-tf2](https://github.com/bubbliiiing/yolov4-tiny-tf2)

   [Yolo-v3-Xilinx](https://github.com/Xilinx/Vitis-AI-Tutorials/tree/ML-at-Edge-yolov3)

   [Yolo-v4-tutorial-Xilinx](https://github.com/Xilinx/Vitis-Tutorials/tree/33d6cf9686398ef1179778dc0da163291c68b465/Machine_Learning/Design_Tutorials/07-yolov4-tutorial)

   [Yolo-v3-dnndk](https://github.com/Xilinx/Vitis-AI/blob/v1.1/mpsoc/vitis_ai_dnndk_samples/tf_yolov3_voc_py/tf_yolov3_voc.py)

   [UA-DETRAC to VOC](https://blog.csdn.net/weixin_38106878/article/details/88684280?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-3.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-3.control)

   [Vitis-AI 1.1](https://www.xilinx.com/html_docs/vitis_ai/1_1/zkj1576857115470.html)

   [Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest)

