# Introduction
Translating the model in [yolox](https://github.com/Megvii-BaseDetection/YOLOX) to tensorflow2.0.  
you should download yolox model: [yolox_s](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth), 
[yolox_m](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth),
[yolox_l](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth),
[yolox_x](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth),
[yolox_nano](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth),
[yolox_tiny](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth) 
in exp folder( see the content ).   
Supported tensorflow: tensorflow saved_model, tflite
## Content
* yolox_torch2tf
  * datasets
    * datasets.py
  * exp
    * export.py
    * para.py
  * models
    * tf_darknet.py
    * tf_network_blocks.py
    * tf_yolo_head.py
    * tf_yolo_pafpn.py

****
**yolox_torch2tf**: root.  
**datasets**: to export tflite int8, download [coco128](https://www.kaggle.com/datasets/ultralytics/coco128) 
there as .\datasets\coco128.  
**models**: model translation code with tensorflow.  
**exp**: export code and save tf model, you also need download 
[yolox model](https://github.com/Megvii-BaseDetection/YOLOX) to this folder.

## Usage
parameter **int8** only for tflite int8, otherwise please ignore it.
```
    python path/to/export.py -n yolox_nano --tsize 640 --include saved_model --device cpu --int8
                                yolox_tiny                       tflite               gpu
                                yolox_s
                                yolox_x
                                yolox_m
                                yolox_l
```