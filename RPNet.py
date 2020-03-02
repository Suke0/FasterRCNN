#-- coding: utf-8 --
import tensorflow as tf

class RPNet(tf.keras.Model):
    def __init__(self):
        super(RPNet,self).__init__()
        self.layer_stage_1 = Conv_BN_Relu(512,3,1)
        self.layer_stage_2_1 = Conv_BN(18,1,1)
        self.layer_stage_2_2 = Conv_BN(36,1,1)
        pass


    def call(self,inputs): #inputs.shape = [1,38,50,512]
        x = self.layer_stage_1(inputs)
        x1,x2 = x,x
        x1 = self.layer_stage_2_1(x1)
        x0 = x1
        x1 = tf.reshape(x1,(1,inputs.shape[1],inputs.shape[2],9,2))
        x1 = tf.nn.softmax(x1)
        x1 = tf.reshape(x1,(1,inputs.shape[1],inputs.shape[2],18))
        x2 = self.layer_stage_2_2(x2)
        #anchors_tensor = create_anchors_tensor(x.shape,scales=[8, 16, 32], ratios=[0.5, 1.0 , 2.0])
        return x0, x1, x2
        pass
    pass


def Conv_BN_Relu(filters,kernel_size,strides,padding="same"):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters,kernel_size,strides,padding=padding,use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU()
    ])
    pass

def Conv_BN(filters,kernel_size,strides,padding="same"):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters,kernel_size,strides,padding=padding,use_bias=False),
        tf.keras.layers.BatchNormalization()
    ])
    pass

if __name__ == '__main__':
    import numpy as np
    inputs = np.random.random((1,38,50,512))
    inputs = tf.cast(inputs, tf.float32)
    model = RPNet()
    res0, res1, res2 = model(inputs)
    print(res0.shape)
    print(res1.shape)
    print(res2.shape)
    pass