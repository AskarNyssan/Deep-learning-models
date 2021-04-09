# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 19:48:17 2021

@author: NysanAskar
"""
# This file contains modules common to various models
import keras
import tensorflow as tf
from keras import backend as K

class Mish(tf.keras.layers.Layer):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Mish()(X_input)
    '''
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True
    def call(self, input_tensor):
        return input_tensor * K.tanh(K.softplus(input_tensor))
    def get_config(self):
        config = super(Mish, self).get_config()
        return config
    def compute_output_shape(self, input_shape):
        return input_shape

def DWConv(c2, c1, k=1, s=1, act=True):
    # Depthwise convolution
    d = DWSConv(c1=c1, c2=c2, k=k, s=s, act=act)
    return d


class DWSConv(tf.keras.layers.Layer):
    # Depthwise 2-D convolution
    def __init__(self,c2, c1=1, k=1, s=1, padding='same', act=True): # c2-output filter, c1-input channel size, kernel, stride, padding, groups
        super(DWSConv, self).__init__()
        self.depth_mult = c2//c1
        self.act = act
        self.conv = tf.keras.layers.SeparableConv2D(filters=c2, kernel_size=k, strides=s, padding =padding, depth_multiplier=self.depth_mult, use_bias=False, activation=None)
        self.activation = Mish()
    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        if self.act: x = self.activation(x)
        return x


class Conv(tf.keras.layers.Layer):
    # convolution layer with Mish activation
    def __init__(self,c2, c1=1, k=1, s=1, padding='same', act=True): # c2-output filter, c1-input channel size, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.act = act
        self.conv = tf.keras.layers.Conv2D(filters=c2, kernel_size=k, strides=s, padding =padding, use_bias=False, activation=None)
        self.activation = Mish()
    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        if self.act: x = self.activation(x)
        return x

############
############
#conv_1 = Conv(c2=10)
#x_conv_1 = conv_1(x_train_keras)
#x_conv_1.shape # TensorShape([3, 224, 224, 10])
############
############


class Bottleneck(tf.keras.layers.Layer): # Dark Layer
    # Standard bottleneck
    def __init__(self, c2, c1=3, shortcut=True, e=0.5):  # c2-output filter, c1-input channel size, kernel, stride, padding, groups
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c_, k=1, s=1)
        self.cv2 = Conv(c2, k=3, s=1)
        self.add = shortcut and c1 == c2
    def call(self, input_tensor):
        return input_tensor + self.cv2(self.cv1(input_tensor)) if self.add else self.cv2(self.cv1(input_tensor))

############
############
#m_bottleneck = Bottleneck(c2=10)
#output_bottleneck = conv_1(x_train_keras)
#output_bottleneck.shape # TensorShape([3, 224, 224, 10])
############
############


class BottleneckCSP(tf.keras.layers.Layer):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c2, c1=3,n=1, shortcut=True, e=0.5):  # c2-output filter, c1-input channel size, kernel, stride, padding
        super(BottleneckCSP, self).__init__()
        self.e = e
        self.c_ = int(c2 * self.e)  # hidden channels
        self.cv1 = Conv(self.c_, k=1, s=1)
        self.cv2 = tf.keras.layers.Conv2D(self.c_, 1, 1, use_bias=False)
        self.cv3 = tf.keras.layers.Conv2D(self.c_, 1, 1, use_bias=False)
        self.cv4 = Conv(c2, k=1, s=1)
        self.bn = tf.keras.layers.BatchNormalization()# applied to concat(cv2, cv3)
        self.act = Mish()
        self.m = keras.Sequential(*[Bottleneck(self.c_, self.c_, shortcut=shortcut, e=1.0) for _ in range(n)])
    def call(self, input_tensor):
        y1 = self.cv3(self.m(self.cv1(input_tensor)))
        y2 = self.cv2(input_tensor)
        return self.cv4(self.act(self.bn(tf.concat([y1, y2], -1))))

############
############
#m_bottleneck = BottleneckCSP(c2=100)
#output_bottleneck = m_bottleneck(x_train_keras)
#output_bottleneck.shape # TensorShape([3, 224, 224, 100])
############
############


class BottleneckCSP2(tf.keras.layers.Layer):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c2, c1=3, n=1, shortcut=False, e=0.5):  # c2-output filter, c1-input channel size, kernel, stride, padding, groups
        super(BottleneckCSP2, self).__init__()
        self.c_ = int(c2) # hidden channels
        self.cv1 = Conv(self.c_, k=1, s=1)
        self.cv2 = tf.keras.layers.Conv2D(self.c_, 1, 1, use_bias=False)
        self.cv3 = Conv(c2, k=1, s=1)
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = Mish()
        self.m = keras.Sequential(*[Bottleneck(c2=self.c_, c1=self.c_, shortcut=shortcut, e=1.0) for _ in range(n)])
    def call(self, input_tensor):
        x1 = self.cv1(input_tensor)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.bn(tf.concat([y1, y2], -1))))

############
############
#m_bottleneck_2 = BottleneckCSP2(c2=100)
#output_bottleneck_2 = m_bottleneck_2(x_train_keras)
#output_bottleneck_2.shape # TensorShape([3, 224, 224, 100])
############
############


class VoVCSP(tf.keras.layers.Layer):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c2, c1=3, n=1, shortcut=True, e=0.5):  # # c2-output filter, c1-input channel size, kernel, stride, padding, groups
        super(VoVCSP, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c_//2, k=3, s=1)
        self.cv2 = Conv(c_//2, k=3, s=1)
        self.cv3 = Conv(c2, k=1, s=1)

    def call(self, input_tensor):
        _, x1 = tf.split(input_tensor, 2, axis=-1)
        x1 = self.cv1(x1)
        x2 = self.cv2(x1)
        return self.cv3(tf.concat([x1, x2], -1))

############
############
#Bottleneck_CSP = VoVCSP(c2=50)
#Bottleneck_CSP = Bottleneck_CSP(output_bottleneck_2)
#Bottleneck_CSP.shape # TensorShape([3, 224, 224, 50])
############
############


class SPP(tf.keras.layers.Layer):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c2, c1, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.k = k
        self.cv1 = Conv(c_, k=1, s=1)
        self.cv2 = Conv(c1=c_ * (len(k) + 1), c2=c2, k=1, s=1)
        self.m = [tf.keras.layers.MaxPooling2D(pool_size=x, strides=(1, 1), padding='same') for x in self.k]
    def call(self, input_tensor):
        x = self.cv1(input_tensor)
        return self.cv2(tf.concat([x] + [m(x) for m in self.m], -1))

############
############
#Bottleneck_SPP = SPP(c1=50, c2=30)
#x_Bottleneck_SPP = Bottleneck_SPP(Bottleneck_CSP)
#x_Bottleneck_SPP.shape # TensorShape([3, 224, 224, 50])
############
############


class SPPCSP(tf.keras.layers.Layer):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c2, c1=1, n=1, shortcut=False, e=0.5, k=(5, 9, 13)):
        super(SPPCSP, self).__init__()
        self.e = e
        self.k = k
        c_ = int(2 * c2 * self.e)  # hidden channels
        self.cv1 = Conv(c_, k=1, s=1)
        self.cv2 = tf.keras.layers.Conv2D(c_, 1, 1, use_bias=False)
        self.cv3 = Conv(c_, k=3, s=1)
        self.cv4 = Conv(c_, k=1, s=1)
        self.m = [tf.keras.layers.MaxPooling2D(pool_size=x, strides=(1, 1), padding='same') for x in self.k]
        self.cv5 = Conv(c_, k=1, s=1)
        self.cv6 = Conv(c_, k=3, s=1)
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = Mish()
        self.cv7 = Conv(c2, k=1, s=1)

    def call(self, input_tensor):
        x1 = self.cv4(self.cv3(self.cv1(input_tensor)))
        y1 = self.cv6(self.cv5(tf.concat([x1] + [m(x1) for m in self.m], -1)))
        y2 = self.cv2(input_tensor)
        return self.cv7(self.act(self.bn(tf.concat([y1, y2], -1))))

############
############
#Bottleneck_SPPCSP = SPPCSP(c2=30)
#x_Bottleneck_SPPCSP = Bottleneck_SPPCSP(x_Bottleneck_SPP)
#x_Bottleneck_SPPCSP.shape # TensorShape([3, 224, 224, 30])
############
############


class MP(tf.keras.layers.Layer):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = tf.keras.layers.MaxPooling2D(pool_size=k, strides=k, padding='valid')
    def call(self, input_tensor):
        x= self.m(input_tensor)
        return x

############
############
#MP_1 = MP(k=2)
#x_MP_1 = MP_1(x_Bottleneck_SPPCSP)
#x_MP_1.shape # TensorShape([3, 112, 112, 30])
############
############


