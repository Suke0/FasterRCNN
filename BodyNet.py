#-- coding: utf-8 --
import tensorflow as tf
#VGG16
class BodyNet(tf.keras.Model):
    def __init__(self):
        super(BodyNet,self).__init__()
        self.layers_stage_1a = Conv_BN_Relu(64,3,1)
        self.layers_stage_1b = Conv_BN_Relu(64,3,1)
        self.layers_stage_1c = tf.keras.layers.MaxPool2D(2,2,padding='same')

        self.layers_stage_2a = Conv_BN_Relu(128,3,1)
        self.layers_stage_2b = Conv_BN_Relu(128,3,1)
        self.layers_stage_2c = tf.keras.layers.MaxPool2D(2,2,padding='same')

        self.layers_stage_3a = Conv_BN_Relu(256,3,1)
        self.layers_stage_3b = Conv_BN_Relu(256,3,1)
        self.layers_stage_3c = Conv_BN_Relu(256,3,1)
        self.layers_stage_3d = tf.keras.layers.MaxPool2D(2,2,padding='same')

        self.layers_stage_4a = Conv_BN_Relu(512,3,1)
        self.layers_stage_4b = Conv_BN_Relu(512,3,1)
        self.layers_stage_4c = Conv_BN_Relu(512,3,1)
        self.layers_stage_4d = tf.keras.layers.MaxPool2D(2,2,padding='same')

        self.layers_stage_5a = Conv_BN_Relu(512,3,1)
        self.layers_stage_5b = Conv_BN_Relu(512,3,1)
        self.layers_stage_5c = Conv_BN_Relu(512,3,1)
        pass

    def call(self,inputs):#(batch_size,600,1000,3)
        x = self.layers_stage_1a(inputs)
        x = self.layers_stage_1b(x)
        x = self.layers_stage_1c(x)

        x = self.layers_stage_2a(x)
        x = self.layers_stage_2b(x)
        x = self.layers_stage_2c(x)

        x = self.layers_stage_3a(x)
        x = self.layers_stage_3b(x)
        x = self.layers_stage_3c(x)
        x = self.layers_stage_3d(x)

        x = self.layers_stage_4a(x)
        x = self.layers_stage_4b(x)
        x = self.layers_stage_4c(x)
        x = self.layers_stage_4d(x)

        x = self.layers_stage_5a(x)
        x = self.layers_stage_5b(x)
        x = self.layers_stage_5c(x)
        return x #(batch_size,38,50,512)
        pass
    pass

def Conv_BN_Relu(filters,kernel_size,strides,padding="same"):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters,kernel_size,strides,padding=padding,use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU()
    ])
    pass

if __name__ == '__main__':
    import numpy as np
    inputs = np.random.random((1,600,800,3))
    inputs = tf.cast(inputs, tf.float32)
    model = BodyNet()
    res = model(inputs)
    print(res.shape)
    pass