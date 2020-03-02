# -*- coding: utf-8 -*-
# @Time    : 2020/2/28 8:10
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : predict.py
# @Software: PyCharm
from FasterRCNN import FasterRCNN
import numpy as np

if __name__ == "__main__":
    model = FasterRCNN(False,21)
    inputs = np.random.random((1,600,1000,3)) * 255
    imgs_info = [[1,600,1000,1]]
    res = model(inputs,imgs_info)
    print(res.shape)
    pass