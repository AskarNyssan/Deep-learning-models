# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:16:07 2021

@author: NysanAskar
"""
"""Inception-V3 model for Tensorflow and Keras.
Reference:
  - [Rethinking the Inception Architecture for Computer Vision](
      http://arxiv.org/abs/1512.00567) (CVPR 2016)
"""

import tensorflow as tf
from tensorflow import keras

## First convolutional block that consists of 7 layers
class Conv2D_BN_Block_1(tf.keras.layers.Layer):
    def __init__(self):
        super(Conv2D_BN_Block_1, self).__init__()
        self.kernel_num = 192
        self.conv_1 = Conv2D_BN_Layer_1(kernel_num=32, kernel_size=(3,3), striding=(2, 2), padding='valid')
        self.conv_2 = Conv2D_BN_Layer(kernel_size=(3, 3), padding='valid')
        self.conv_3 = Conv2D_BN_Layer(kernel_num=64, kernel_size=(3,3))
        self.conv_4 = Conv2D_BN_Layer(kernel_num=80, padding='valid')
        self.conv_5 = Conv2D_BN_Layer(kernel_num=192, kernel_size=(3,3), padding='valid')
    def call(self, input_tensor, training=False):
        x = self.conv_1(input_tensor)
        x = keras.layers.BatchNormalization()(x)
        x = self.conv_2(x)
        x = keras.layers.BatchNormalization()(x)
        x = self.conv_3(x)
        x = keras.layers.BatchNormalization()(x)
        x = tf.nn.max_pool(x, ksize=(3,3), strides=(2,2), padding='VALID')
        x = self.conv_4(x)
        x = keras.layers.BatchNormalization()(x)
        x = self.conv_5(x)
        x = keras.layers.BatchNormalization()(x)
        x = tf.nn.max_pool(x, ksize=(3,3), strides=(2,2), padding='VALID')
        if x.shape[-1] != self.kernel_num:
            raise Exception("Number of channels is not equal to " + str(self.kernel_num))
        return x

## First convolutional layer
class Conv2D_BN_Layer_1(tf.keras.layers.Layer):
    def __init__(self, kernel_num=32, kernel_size=(1,1), striding=(1, 1), padding='same'):
        super(Conv2D_BN_Layer_1, self).__init__()
        self.kernel_num = kernel_num
        self.conv_1 = tf.keras.layers.Conv2D(kernel_num, input_shape = (299, 299,3),
                        kernel_size=kernel_size, strides=striding, padding=padding, activation='relu')

    def call(self, input_tensor, training=False):
        x = self.conv_1(input_tensor)
        x = keras.layers.BatchNormalization()(x)
        if x.shape[-1] != self.kernel_num:
            raise Exception("Number of channels is not equal to " + str(self.kernel_num))
        return x

## Convolutional layer
class Conv2D_BN_Layer(tf.keras.layers.Layer):
    def __init__(self, kernel_num=32, kernel_size=(1,1), striding=(1, 1), padding='same'):
        super(Conv2D_BN_Layer, self).__init__()
        self.kernel_num = kernel_num
        self.conv_1 = tf.keras.layers.Conv2D(kernel_num,
                        kernel_size=kernel_size, strides=striding, padding=padding, activation='relu')
    def call(self, input_tensor, training=False):
        x = self.conv_1(input_tensor)
        x = keras.layers.BatchNormalization()(x)
        if x.shape[-1] != self.kernel_num:
            raise Exception("Number of channels is not equal to " + str(self.kernel_num))
        return x

## Inception Block A that consists of 3 inception modules
class Inception_Block_A(tf.keras.layers.Layer):
    def __init__(self):
        super(Inception_Block_A, self).__init__()
        self.kernel_num = 288
        self.module_a1 = Inception_Module_A1()
        self.module_a2 = Inception_Module_A2()
        self.module_a3 = Inception_Module_A2()
    def call(self, input_tensor, training=False):
        x_Inception_Module_A_1 = self.module_a1(input_tensor)
        x_Inception_Module_A_2_1 = self.module_a2(x_Inception_Module_A_1)
        x_Inception_Module_A_2_2 = self.module_a3(x_Inception_Module_A_2_1)
        if x_Inception_Module_A_2_2.shape[-1] != self.kernel_num:
            raise Exception("Number of channels is not equal to " + str(self.kernel_num))
        return x_Inception_Module_A_2_2

## First inseption module A
class Inception_Module_A1(tf.keras.layers.Layer):
    def __init__(self):
        super(Inception_Module_A1, self).__init__()
        self.kernel_num = 256
        self.conv_a1 = Conv2D_BN_Layer(kernel_num=64)
        self.conv_b1 = Conv2D_BN_Layer(kernel_num=48)
        self.conv_b2 = Conv2D_BN_Layer(kernel_num=64, kernel_size=(5,5))
        self.conv_c1 = Conv2D_BN_Layer(kernel_num=64)
        self.conv_c2 = Conv2D_BN_Layer(kernel_num=96, kernel_size=(3,3))
        self.conv_c3 = Conv2D_BN_Layer(kernel_num=96, kernel_size=(3,3))
        self.conv_d2 = Conv2D_BN_Layer(kernel_num=32)
    def call(self, input_tensor, training=False):
        x_a1 = self.conv_a1(input_tensor)
        x_a1 = keras.layers.BatchNormalization()(x_a1)
        x_b1 = self.conv_b1(input_tensor)
        x_b1 = keras.layers.BatchNormalization()(x_b1)
        x_b2 = self.conv_b2(x_b1)
        x_b2 = keras.layers.BatchNormalization()(x_b2)
        x_c1 = self.conv_c1(input_tensor)
        x_c1 = keras.layers.BatchNormalization()(x_c1)
        x_c2 = self.conv_c2(x_c1)
        x_c2 = keras.layers.BatchNormalization()(x_c2)
        x_c3 = self.conv_c3(x_c2)
        x_c3 = keras.layers.BatchNormalization()(x_c3)
        x_d1 = tf.nn.avg_pool(input_tensor, ksize=(3,3), strides=(1,1), padding='SAME')
        x_d2 = self.conv_d2(x_d1)
        x_d2 = keras.layers.BatchNormalization()(x_d2)
        x_module_A = tf.keras.layers.Concatenate(axis=-1)(
            [x_a1, x_b2, x_c3, x_d2])
        if x_module_A.shape[-1] != self.kernel_num:
            raise Exception("Number of channels is not equal to " + str(self.kernel_num))
        return x_module_A

## Second inception module A
class Inception_Module_A2(tf.keras.layers.Layer):
    def __init__(self):
        super(Inception_Module_A2, self).__init__()
        self.kernel_num = 288
        self.conv_a1 = Conv2D_BN_Layer(kernel_num=64)
        self.conv_b1 = Conv2D_BN_Layer(kernel_num=48)
        self.conv_b2 = Conv2D_BN_Layer(kernel_num=64, kernel_size=(5,5))
        self.conv_c1 = Conv2D_BN_Layer(kernel_num=64)
        self.conv_c2 = Conv2D_BN_Layer(kernel_num=96, kernel_size=(3,3))
        self.conv_c3 = Conv2D_BN_Layer(kernel_num=96, kernel_size=(3,3))
        self.conv_d2 = Conv2D_BN_Layer(kernel_num=64)
    def call(self, input_tensor, training=False):
        x_a1 = self.conv_a1(input_tensor)
        x_a1 = keras.layers.BatchNormalization()(x_a1)
        x_b1 = self.conv_b1(input_tensor)
        x_b1 = keras.layers.BatchNormalization()(x_b1)
        x_b2 = self.conv_b2(x_b1)
        x_b2 = keras.layers.BatchNormalization()(x_b2)
        x_c1 = self.conv_c1(input_tensor)
        x_c1 = keras.layers.BatchNormalization()(x_c1)
        x_c2 = self.conv_c2(x_c1)
        x_c2 = keras.layers.BatchNormalization()(x_c2)
        x_c3 = self.conv_c3(x_c2)
        x_c3 = keras.layers.BatchNormalization()(x_c3)
        x_d1 = tf.nn.avg_pool(input_tensor, ksize=(3,3), strides=(1,1), padding='SAME')
        x_d2 = self.conv_d2(x_d1)
        x_d2 = keras.layers.BatchNormalization()(x_d2)
        x_module_A = tf.keras.layers.Concatenate(axis=-1)(
            [x_a1, x_b2, x_c3, x_d2])
        if x_module_A.shape[-1] != self.kernel_num:
            raise Exception("Number of channels is not equal to " + str(self.kernel_num))
        return x_module_A

## Inception Block B that consists of 5 inception modules B
class Inception_Block_B(tf.keras.layers.Layer):
    def __init__(self):
        super(Inception_Block_B, self).__init__()
        self.kernel_num = 768
        self.module_b1 = Inception_Module_B1()
        self.module_b2 = Inception_Module_B2()
        self.module_b3_1 = Inception_Module_B3()
        self.module_b3_2 = Inception_Module_B3()
        self.module_b4 = Inception_Module_B4()
    def call(self, input_tensor, training=False):
        x_Inception_Module_B_1 = self.module_b1(input_tensor)
        x_Inception_Module_B_2 = self.module_b2(x_Inception_Module_B_1)
        x_Inception_Module_B_3_1 = self.module_b3_1(x_Inception_Module_B_2)
        x_Inception_Module_B_3_2 = self.module_b3_2(x_Inception_Module_B_3_1)
        x_Inception_Module_B_4 = self.module_b4(x_Inception_Module_B_3_2)
        if x_Inception_Module_B_4.shape[-1] != self.kernel_num:
            raise Exception("Number of channels is not equal to " + str(self.kernel_num))
        return x_Inception_Module_B_4

## First Inception Module B
class Inception_Module_B1(tf.keras.layers.Layer):
    def __init__(self):
        super(Inception_Module_B1, self).__init__()
        self.kernel_num = 768
        self.conv_a1 = Conv2D_BN_Layer(kernel_num=384, kernel_size=(3,3), striding=(2, 2), padding='valid')
        self.conv_b1 = Conv2D_BN_Layer(kernel_num=64)
        self.conv_b2 = Conv2D_BN_Layer(kernel_num=96, kernel_size=(3,3))
        self.conv_b3 = Conv2D_BN_Layer(kernel_num=96, kernel_size=(3,3), striding=(2, 2), padding='valid')
    def call(self, input_tensor, training=False):
        x_a1 = self.conv_a1(input_tensor)
        x_a1 = keras.layers.BatchNormalization()(x_a1)
        x_b1 = self.conv_b1(input_tensor)
        x_b1 = keras.layers.BatchNormalization()(x_b1)
        x_b2 = self.conv_b2(x_b1)
        x_b2 = keras.layers.BatchNormalization()(x_b2)
        x_b3 = self.conv_b3(x_b2)
        x_b3 = keras.layers.BatchNormalization()(x_b3)
        x_c1 = tf.nn.max_pool2d(input_tensor, ksize=(3,3), strides=(2,2), padding = 'VALID')
        x_module_B1 = tf.keras.layers.Concatenate(axis=-1)(
            [x_a1, x_b3, x_c1])
        if x_module_B1.shape[-1] != self.kernel_num:
            raise Exception("Number of channels is not equal to " + str(self.kernel_num))
        return x_module_B1

## Second Inception Module B
class Inception_Module_B2(tf.keras.layers.Layer):
    def __init__(self):
        super(Inception_Module_B2, self).__init__()
        self.kernel_num = 768
        self.conv_a1 = Conv2D_BN_Layer(kernel_num=192)
        self.conv_b1 = Conv2D_BN_Layer(kernel_num=128)
        self.conv_b2 = Conv2D_BN_Layer(kernel_num=128, kernel_size=(1,7))
        self.conv_b3 = Conv2D_BN_Layer(kernel_num=192, kernel_size=(7,1))
        self.conv_c1 = Conv2D_BN_Layer(kernel_num=128)
        self.conv_c2 = Conv2D_BN_Layer(kernel_num=128, kernel_size=(7,1))
        self.conv_c3 = Conv2D_BN_Layer(kernel_num=128, kernel_size=(1,7))
        self.conv_c4 = Conv2D_BN_Layer(kernel_num=128, kernel_size=(7,1))
        self.conv_c5 = Conv2D_BN_Layer(kernel_num=192, kernel_size=(1,7))
        self.conv_d2 = Conv2D_BN_Layer(kernel_num=192)
    def call(self, input_tensor, training=False):
        x_a1 = self.conv_a1(input_tensor)
        x_a1 = keras.layers.BatchNormalization()(x_a1)
        x_b1 = self.conv_b1(input_tensor)
        x_b1 = keras.layers.BatchNormalization()(x_b1)
        x_b2 = self.conv_b2(x_b1)
        x_b2 = keras.layers.BatchNormalization()(x_b2)
        x_b3 = self.conv_b3(x_b2)
        x_b3 = keras.layers.BatchNormalization()(x_b3)
        x_c1 = self.conv_c1(input_tensor)
        x_c1 = keras.layers.BatchNormalization()(x_c1)
        x_c2 = self.conv_c2(x_c1)
        x_c2 = keras.layers.BatchNormalization()(x_c2)
        x_c3 = self.conv_c3(x_c2)
        x_c3 = keras.layers.BatchNormalization()(x_c3)
        x_c4 = self.conv_c4(x_c3)
        x_c4 = keras.layers.BatchNormalization()(x_c4)
        x_c5 = self.conv_c5(x_c4)
        x_c5 = keras.layers.BatchNormalization()(x_c5)
        x_d1 = tf.nn.avg_pool(input_tensor, ksize=(3,3), strides=(1,1), padding='SAME')
        x_d2 = self.conv_d2(x_d1)
        x_d2 = keras.layers.BatchNormalization()(x_d2)
        x_module_B2 = tf.keras.layers.Concatenate(axis=-1)(
            [x_a1, x_b3, x_c5, x_d2])
        if x_module_B2.shape[-1] != self.kernel_num:
            raise Exception("Number of channels is not equal to " + str(self.kernel_num))
        return x_module_B2

## Third Inception Module B
class Inception_Module_B3(tf.keras.layers.Layer):
    def __init__(self):
        super(Inception_Module_B3, self).__init__()
        self.kernel_num = 768
        self.conv_a1 = Conv2D_BN_Layer(kernel_num=192)
        self.conv_b1 = Conv2D_BN_Layer(kernel_num=160)
        self.conv_b2 = Conv2D_BN_Layer(kernel_num=160, kernel_size=(1,7))
        self.conv_b3 = Conv2D_BN_Layer(kernel_num=192, kernel_size=(7,1))
        self.conv_c1 = Conv2D_BN_Layer(kernel_num=160)
        self.conv_c2 = Conv2D_BN_Layer(kernel_num=160, kernel_size=(7,1))
        self.conv_c3 = Conv2D_BN_Layer(kernel_num=160, kernel_size=(1,7))
        self.conv_c4 = Conv2D_BN_Layer(kernel_num=160, kernel_size=(7,1))
        self.conv_c5 = Conv2D_BN_Layer(kernel_num=192, kernel_size=(1,7))
        self.conv_d2 = Conv2D_BN_Layer(kernel_num=192)
    def call(self, input_tensor, training=False):
        x_a1 = self.conv_a1(input_tensor)
        x_a1 = keras.layers.BatchNormalization()(x_a1)
        x_b1 = self.conv_b1(input_tensor)
        x_b1 = keras.layers.BatchNormalization()(x_b1)
        x_b2 = self.conv_b2(x_b1)
        x_b2 = keras.layers.BatchNormalization()(x_b2)
        x_b3 = self.conv_b3(x_b2)
        x_b3 = keras.layers.BatchNormalization()(x_b3)
        x_c1 = self.conv_c1(input_tensor)
        x_c1 = keras.layers.BatchNormalization()(x_c1)
        x_c2 = self.conv_c2(x_c1)
        x_c2 = keras.layers.BatchNormalization()(x_c2)
        x_c3 = self.conv_c3(x_c2)
        x_c3 = keras.layers.BatchNormalization()(x_c3)
        x_c4 = self.conv_c4(x_c3)
        x_c4 = keras.layers.BatchNormalization()(x_c4)
        x_c5 = self.conv_c5(x_c4)
        x_c5 = keras.layers.BatchNormalization()(x_c5)
        x_d1 = tf.nn.avg_pool(input_tensor, ksize=(3,3), strides=(1,1), padding='SAME')
        x_d2 = self.conv_d2(x_d1)
        x_d2 = keras.layers.BatchNormalization()(x_d2)
        x_module_B3 = tf.keras.layers.Concatenate(axis=-1)(
            [x_a1, x_b3, x_c5, x_d2])
        if x_module_B3.shape[-1] != self.kernel_num:
            raise Exception("Number of channels is not equal to " + str(self.kernel_num))
        return x_module_B3

## Fourth Inception Module B
class Inception_Module_B4(tf.keras.layers.Layer):
    def __init__(self):
        super(Inception_Module_B4, self).__init__()
        self.kernel_num = 768
        self.conv_a1 = Conv2D_BN_Layer(kernel_num=192)
        self.conv_b1 = Conv2D_BN_Layer(kernel_num=192)
        self.conv_b2 = Conv2D_BN_Layer(kernel_num=192, kernel_size=(1,7))
        self.conv_b3 = Conv2D_BN_Layer(kernel_num=192, kernel_size=(7,1))
        self.conv_c1 = Conv2D_BN_Layer(kernel_num=192)
        self.conv_c2 = Conv2D_BN_Layer(kernel_num=192, kernel_size=(7,1))
        self.conv_c3 = Conv2D_BN_Layer(kernel_num=192, kernel_size=(1,7))
        self.conv_c4 = Conv2D_BN_Layer(kernel_num=192, kernel_size=(7,1))
        self.conv_c5 = Conv2D_BN_Layer(kernel_num=192, kernel_size=(1,7))
        self.conv_d2 = Conv2D_BN_Layer(kernel_num=192)
    def call(self, input_tensor, training=False):
        x_a1 = self.conv_a1(input_tensor)
        x_a1 = keras.layers.BatchNormalization()(x_a1)
        x_b1 = self.conv_b1(input_tensor)
        x_b1 = keras.layers.BatchNormalization()(x_b1)
        x_b2 = self.conv_b2(x_b1)
        x_b2 = keras.layers.BatchNormalization()(x_b2)
        x_b3 = self.conv_b3(x_b2)
        x_b3 = keras.layers.BatchNormalization()(x_b3)
        x_c1 = self.conv_c1(input_tensor)
        x_c1 = keras.layers.BatchNormalization()(x_c1)
        x_c2 = self.conv_c2(x_c1)
        x_c2 = keras.layers.BatchNormalization()(x_c2)
        x_c3 = self.conv_c3(x_c2)
        x_c3 = keras.layers.BatchNormalization()(x_c3)
        x_c4 = self.conv_c4(x_c3)
        x_c4 = keras.layers.BatchNormalization()(x_c4)
        x_c5 = self.conv_c5(x_c4)
        x_c5 = keras.layers.BatchNormalization()(x_c5)
        x_d1 = tf.nn.avg_pool(input_tensor, ksize=(3,3), strides=(1,1), padding='SAME')
        x_d2 = self.conv_d2(x_d1)
        x_d2 = keras.layers.BatchNormalization()(x_d2)
        x_module_B4 = tf.keras.layers.Concatenate(axis=-1)(
            [x_a1, x_b3, x_c5, x_d2])
        if x_module_B4.shape[-1] != self.kernel_num:
            raise Exception("Number of channels is not equal to " + str(self.kernel_num))
        return x_module_B4

## Inception Module C
class Inception_Block_C(tf.keras.layers.Layer):
    def __init__(self):
        super(Inception_Block_C, self).__init__()
        self.kernel_num = 2048
        self.module_c1 = Inception_Module_C1()
        self.module_c2 = Inception_Module_C2()
        self.module_c3 = Inception_Module_C2()
    def call(self, input_tensor, training=False):
        x_Inception_Module_C_1 = self.module_c1(input_tensor)
        x_Inception_Module_C_2 = self.module_c2(x_Inception_Module_C_1)
        x_Inception_Module_C_3 = self.module_c3(x_Inception_Module_C_2)
        if x_Inception_Module_C_3.shape[-1] != self.kernel_num:
            raise Exception("Number of channels is not equal to " + str(self.kernel_num))
        return x_Inception_Module_C_3

## First inception module C
class Inception_Module_C1(tf.keras.layers.Layer):
    def __init__(self):
        super(Inception_Module_C1, self).__init__()
        self.kernel_num = 1280
        self.conv_a1 = Conv2D_BN_Layer(kernel_num=192)
        self.conv_a2 = Conv2D_BN_Layer(kernel_num=320, kernel_size=(3,3), striding=(2, 2), padding='valid')
        self.conv_b1 = Conv2D_BN_Layer(kernel_num=192)
        self.conv_b2 = Conv2D_BN_Layer(kernel_num=192, kernel_size=(1,7))
        self.conv_b3 = Conv2D_BN_Layer(kernel_num=192, kernel_size=(7,1))
        self.conv_b4 = Conv2D_BN_Layer(kernel_num=192, kernel_size=(3,3), striding=(2, 2), padding='valid')
    def call(self, input_tensor, training=False):
        x_a1 = self.conv_a1(input_tensor)
        x_a1 = keras.layers.BatchNormalization()(x_a1)
        x_a2 = self.conv_a2(x_a1)
        x_a2 = keras.layers.BatchNormalization()(x_a2)
        x_b1 = self.conv_b1(input_tensor)
        x_b1 = keras.layers.BatchNormalization()(x_b1)
        x_b2 = self.conv_b2(x_b1)
        x_b2 = keras.layers.BatchNormalization()(x_b2)
        x_b3 = self.conv_b3(x_b2)
        x_b3 = keras.layers.BatchNormalization()(x_b3)
        x_b4 = self.conv_b4(x_b3)
        x_b4 = keras.layers.BatchNormalization()(x_b4)
        x_c1 = tf.nn.max_pool2d(input_tensor, ksize=(3,3), strides=(2,2), padding = 'VALID')
        x_module_C1 = tf.keras.layers.Concatenate(axis=-1)(
            [x_a2, x_b4, x_c1])
        if x_module_C1.shape[-1] != self.kernel_num:
            raise Exception("Number of channels is not equal to " + str(self.kernel_num))
        return x_module_C1

## Second inception module C
class Inception_Module_C2(tf.keras.layers.Layer):
    def __init__(self):
        super(Inception_Module_C2, self).__init__()
        self.kernel_num = 2048
        self.conv_a1 = Conv2D_BN_Layer(kernel_num=320)
        self.conv_b1 = Conv2D_BN_Layer(kernel_num=384)
        self.conv_b2_1 = Conv2D_BN_Layer(kernel_num=384, kernel_size=(1,3))
        self.conv_b2_2 = Conv2D_BN_Layer(kernel_num=384, kernel_size=(3,1))
        self.conv_c1 = Conv2D_BN_Layer(kernel_num=448)
        self.conv_c2 = Conv2D_BN_Layer(kernel_num=384, kernel_size=(31,3))
        self.conv_c3_1 = Conv2D_BN_Layer(kernel_num=384, kernel_size=(1,3))
        self.conv_c3_2 = Conv2D_BN_Layer(kernel_num=384, kernel_size=(3,1))
        self.conv_d2 = Conv2D_BN_Layer(kernel_num=192)
    def call(self, input_tensor, training=False):
        x_a1 = self.conv_a1(input_tensor)
        x_a2 = keras.layers.BatchNormalization()(x_a1)
        x_b1 = self.conv_b1(input_tensor)
        x_b1 = keras.layers.BatchNormalization()(x_b1)
        x_b2_1 = self.conv_b2_1(x_b1)
        x_b2_1 = keras.layers.BatchNormalization()(x_b2_1)
        x_b2_2 = self.conv_b2_2(x_b1)
        x_b2_2 = keras.layers.BatchNormalization()(x_b2_2)
        x_b3 = tf.keras.layers.Concatenate(axis=-1)([x_b2_1, x_b2_2])
        x_c1 = self.conv_c1(input_tensor)
        x_c1 = keras.layers.BatchNormalization()(x_c1)
        x_c2 = self.conv_c2(x_c1)
        x_c2 = keras.layers.BatchNormalization()(x_c2)
        x_c3_1 = self.conv_c3_1(x_c2)
        x_c3_1 = keras.layers.BatchNormalization()(x_c3_1)
        x_c3_2 = self.conv_c3_2(x_c2)
        x_c3_2 = keras.layers.BatchNormalization()(x_c3_2)
        x_c4 = tf.keras.layers.Concatenate(axis=-1)([x_c3_1, x_c3_2])
        x_d1 = tf.nn.avg_pool(input_tensor, ksize=(3,3), strides=(1,1), padding = 'SAME')
        x_d2 = self.conv_d2(x_d1)
        x_module_C2 = tf.keras.layers.Concatenate(axis=-1)(
            [x_a2, x_b3, x_c4, x_d2])
        if x_module_C2.shape[-1] != self.kernel_num:
            raise Exception("Number of channels is not equal to " + str(self.kernel_num))
        return x_module_C2

## Linear Layer with softmax function
class Linear_Layer_Softmax(keras.layers.Layer):
    def __init__(self, output_units=5):
        super(Linear_Layer_Softmax, self).__init__()
        self.output_units = output_units
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.output_units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.output_units,), initializer="random_normal", trainable=True
        )
    def call(self, inputs):
        x = tf.matmul(inputs, self.w) + self.b
        x = tf.nn.softmax(x)
        return x

## Last dense block
class Block_Dense(tf.keras.layers.Layer):
    def __init__(self, kernel_num=5):
        super(Block_Dense, self).__init__()
        self.kernel_num = kernel_num
        self.linear_1 = Linear_Layer_Softmax(output_units=5)
    def call(self, input_tensor, training=False):
        x_1 = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
        x_2 = self.linear_1(x_1)
        if x_2.shape[-1] != self.kernel_num:
            raise Exception("Number of channels is not equal to " + str(self.kernel_num))
        return x_2

## Final model
class Inception_V3(tf.keras.Model):
    def __init__(self):
        super(Inception_V3, self).__init__()
        self.block_1 = Conv2D_BN_Block_1()
        self.block_2 = Inception_Block_A()
        self.block_3 = Inception_Block_B()
        self.block_4 = Inception_Block_C()
        self.block_5 = Block_Dense()
    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        return x


