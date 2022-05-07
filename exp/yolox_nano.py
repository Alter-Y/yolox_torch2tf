from models.tf_yolo_pafpn import TFYOLOPAFPN
from models.tf_yolo_head import TFYOLOXHead
import tensorflow as tf
import torch
import tf2onnx
import numpy as np
from tensorflow import keras

# class YOLOX(keras.layers.Layer):
#     def __init__(self, backbone=None, head=None):
#         super().__init__()
#         if backbone is None:
#             backbone = TFYOLOPAFPN()
#         if head is None:
#             head = TFYOLOXHead(80)
#
#         self.backbone = backbone
#         self.head = head
#
#     def call(self, inputs, targets=None):
#         # fpn output content features of [dark3, dark4, dark5]
#         fpn_outs = self.backbone(inputs)
#         outputs = self.head(fpn_outs)
#
#         return outputs
def parse_model(weight):
    depth = 0.33
    width = 0.375  # 0.25 for nano
    test_size = (416, 416)
    in_channels = [256, 512, 1024]
    act = "silu"
    backbone = TFYOLOPAFPN(
        depth, width, in_channels=in_channels,
        act=act, depthwise=False, w=weight['model']
    )
    head = TFYOLOXHead(
        80, width, in_channels=in_channels,
        act=act, depthwise=False, w=weight['model']
    )

    return keras.Sequential([backbone, head])

def predict(inputs, model):
    x = inputs
    for m in model.layers:
        x = m(x)

    return x

if __name__ == '__main__':
    pt = '.\\yolox_tiny.pth'
    m = torch.load(pt, map_location=torch.device('cpu'))
    inputs = tf.keras.Input(shape=(640, 640, 3))
    model = parse_model(m)
    out = predict(inputs, model)
    keras_model = tf.keras.Model(inputs=inputs, outputs=out)
    keras_model.trainable = False
    keras_model.summary()

    keras_model.save('.\\yolox_tiny_save_model', save_format='tf')
    #
    # pb = '.\\yolox_tiny_save_model_1'
    # tf_model = tf.keras.models.load_model(pb)
    # spec = (tf.TensorSpec((None, 640, 640, 3), tf.float32, name="input"),)
    # output_path = ".\\yolox_tiny_save_model_1.onnx"
    # tf2onnx.convert.from_keras(tf_model, input_signature=spec, opset=13, output_path=output_path)
    # # for layer in model.layers:
    #     print(layer)

    # load_model = tf.keras.models.load_model(pb)
    # load_model.summary()
    # yolopafpn = load_model.get_layer('tfyolopafpn').get_weights()
    # head = load_model.get_layer('tfyolox_head').get_weights()
    # seq1 = np.array(yolopafpn)
    # seq2 = np.array(head)
    # print(seq1.shape, seq2.shape)
    # for i in range(20):
    #     print(seq1[i].shape)
    # #
    # dconv = m['model']['backbone.backbone.dark2.0.dconv.conv.weight'].permute(2, 3, 0, 1).numpy()
    # dconv1 = seq1[5]
    # print((dconv==dconv1).all())

