# -*- coding: utf-8 -*-
# @Time    : 2020/2/20 3:34
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : DetectNet.py
# @Software: PyCharm
import tensorflow as tf

class DetectNet(tf.keras.Model):
    def __init__(self,num_classes):
        super(DetectNet,self).__init__()
        self.fc6_a = tf.keras.layers.Dense(4096)
        self.fc6_b = tf.keras.layers.ReLU()
        self.fc7_a = tf.keras.layers.Dense(4096)
        self.fc7_b = tf.keras.layers.ReLU()
        self.fc8_1a = tf.keras.layers.Dense(num_classes*4)
        self.fc8_2a = tf.keras.layers.Dense(num_classes)
        self.fc8_2b = tf.keras.layers.Softmax()
        pass

    def call(self,inputs,rois): #rois.shape=[num_boxes,5],每个box有5个属性：图片序号和4个坐标, inputs.shape = [1,38,50,512]
        # Crop image ROIs
        # 每个anchor经过回归后对应到原图，然后再对应到feature map,经过roi pooling后输出7 * 7的大小的map；
        # 最后对这个7 * 7的map进行分类和回归。
        x = crop_and_resize(inputs,rois) #x.shape = [num_boxes,7,7,512]

        # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
        # 输入输出shape: 具体而言，是将一个维度大于或等于3的高维矩阵，“压扁”为一个二维矩阵。
        # 即保留第一个维度（如：batch的个数），然后将剩下维度的值相乘为“压扁”矩阵的第二个维度。如输入是(None,32，32，3)，则输出是(None, 3072)

        x = tf.reshape(x,[x.shape[0],-1])
        x = self.fc6_a(x)
        x = self.fc6_b(x)
        x = self.fc7_a(x)
        x = self.fc7_b(x)
        x1 = self.fc8_1a(x)
        x2 = self.fc8_2a(x)
        x0 = x2
        x2 = self.fc8_2b(x2)
        return x0, x1, x2 #x1.shape=[num_boxes,num_classes*4],x2.shape=[num_boxes,num_classes]
        pass
    pass

#batch_size为1
def crop_and_resize(inputs,rois):#rois.shape=[num_boxes,5],每个box有5个属性：图片序号和4个坐标
    feat_map_shape = tf.shape(inputs) #[1,38,50,512]
    h = (feat_map_shape[1]-1) * 16
    w = (feat_map_shape[2]-1) * 16
    pred_ratio_x1 = tf.cast(rois[:, 1] / w,tf.float32)
    pred_ratio_y1 = tf.cast(rois[:, 2] / h,tf.float32)
    pred_ratio_x2 = tf.cast(rois[:, 3] / w,tf.float32)
    pred_ratio_y2 = tf.cast(rois[:, 4] / h,tf.float32)

    pred_ratio_x1 = tf.reshape(pred_ratio_x1,(len(pred_ratio_x1),1))
    pred_ratio_y1 = tf.reshape(pred_ratio_y1,(len(pred_ratio_y1),1))
    pred_ratio_x2 = tf.reshape(pred_ratio_x2,(len(pred_ratio_x2),1))
    pred_ratio_y2 = tf.reshape(pred_ratio_y2,(len(pred_ratio_y2),1))

    bboxes = tf.concat([pred_ratio_x1,pred_ratio_y1,pred_ratio_x2,pred_ratio_y2],-1)
    #bbox_idxs = tf.cast(tf.tile([0],[len(bboxes)]),dtype=tf.int32)
    bbox_idxs = rois[:,0]
    crops = tf.image.crop_and_resize(inputs,bboxes,bbox_idxs,[14,14])
    res = tf.nn.max_pool2d(crops,2,1,padding="SAME")
    return res
    pass

if __name__ == '__main__':
    import numpy as np
    inputs = np.random.random((1,38,50,512))
    inputs = tf.cast(inputs, tf.float64)
    rois = np.random.random((300,5)) * 600
    rois[:, 0] = 0
    rois = tf.cast(rois, tf.int32)

    model = DetectNet(21)
    res1, res2 = model(inputs,rois)
    print(res1.shape)
    print(res2.shape)
    pass