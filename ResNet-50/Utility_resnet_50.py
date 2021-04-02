# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 17:47:46 2021

@author: NysanAskar
"""

"""ResNet models for Keras.
Reference:
  - [Deep Residual Learning for Image Recognition](
      https://arxiv.org/abs/1512.03385) (CVPR 2015)
"""

import tensorflow as tf
from tensorflow import keras


## First non-residual block
class Stage_1(tf.keras.layers.Layer):
    def __init__(self):
        super(Stage_1, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(64,
                        kernel_size=(7,7), strides=(2,2), activation=None)
    def call(self, input_tensor):
        x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(input_tensor)
        x = self.conv_1(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = tf.nn.max_pool(x, ksize=(3,3), strides=(2,2), padding='VALID')
        return x


## Residual convolutional block
class Block(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, stride=1, conv_shortcut=True):
        self.filters = filters
        self.kernel_size=kernel_size
        self.stride=stride
        self.conv_shortcut=conv_shortcut
        super(Block, self).__init__()
        self.shortcut_conv = tf.keras.layers.Conv2D(4*filters,
                    kernel_size=(1,1), strides=stride, activation=None)
        self.conv_1 = tf.keras.layers.Conv2D(filters,
                    kernel_size=(1,1), strides=stride, activation=None)
        self.conv_2 = tf.keras.layers.Conv2D(filters,
                    kernel_size=kernel_size, strides=1, activation=None, padding='same')
        self.conv_3 = tf.keras.layers.Conv2D(4*filters,
                    kernel_size=1, strides=1, activation=None)
    def call(self, input_tensor):
        if self.conv_shortcut:
            shortcut = self.shortcut_conv(input_tensor)
            shortcut = keras.layers.BatchNormalization()(shortcut)
        else:
            shortcut = input_tensor
        x = self.conv_1(input_tensor)
        x = keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = self.conv_3(x)
        x = keras.layers.BatchNormalization()(x)
        x = tf.math.add(x, shortcut)
        x = tf.nn.relu(x)
        return x


## Stage 1 with residual convolutional blocks
class Stage_res_1(tf.keras.layers.Layer):
    def __init__(self):
        super(Stage_res_1, self).__init__()
        self.block_1 = Block(filters=64)
        self.block_2 = Block(filters=64, conv_shortcut=False)
        self.block_3 = Block(filters=64, conv_shortcut=False)
    def call(self, input_tensor):
        x = self.block_1(input_tensor)
        x = self.block_2(x)
        x = self.block_3(x)
        return x

## Stage 2 with residual convolutional blocks
class Stage_res_2(tf.keras.layers.Layer):
    def __init__(self):
        super(Stage_res_2, self).__init__()
        self.block_1 = Block(filters=128, stride=2)
        self.block_2 = Block(filters=128, conv_shortcut=False)
        self.block_3 = Block(filters=128, conv_shortcut=False)
        self.block_4 = Block(filters=128, conv_shortcut=False)        
    def call(self, input_tensor):
        x = self.block_1(input_tensor)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        return x


## Stage 3 with residual convolutional blocks
class Stage_res_3(tf.keras.layers.Layer):
    def __init__(self):
        super(Stage_res_3, self).__init__()
        self.block_1 = Block(filters=256, stride=2)
        self.block_2 = Block(filters=256, conv_shortcut=False)
        self.block_3 = Block(filters=256, conv_shortcut=False)
        self.block_4 = Block(filters=256, conv_shortcut=False)
        self.block_5 = Block(filters=256, conv_shortcut=False)
        self.block_6 = Block(filters=256, conv_shortcut=False)
    def call(self, input_tensor):
        x = self.block_1(input_tensor)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        return x


## Stage 4 with residual convolutional blocks
class Stage_res_4(tf.keras.layers.Layer):
    def __init__(self):
        super(Stage_res_4, self).__init__()
        self.block_1 = Block(filters=512, stride=2)
        self.block_2 = Block(filters=512, conv_shortcut=False)
        self.block_3 = Block(filters=512, conv_shortcut=False)
    def call(self, input_tensor):
        x = self.block_1(input_tensor)
        x = self.block_2(x)
        x = self.block_3(x)
        return x


## Stacked stages with residual blocks
class Stack_stages(tf.keras.layers.Layer):
    def __init__(self):
        super(Stack_stages, self).__init__()
        self.stage_1 = Stage_res_1()
        self.stage_2 = Stage_res_2()
        self.stage_3 = Stage_res_3()
        self.stage_4 = Stage_res_4()
    def call(self, input_tensor):
        x = self.stage_1(input_tensor)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        return x


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
    def call(self, input_tensor):
        x_1 = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
        x_2 = self.linear_1(x_1)
        if x_2.shape[-1] != self.kernel_num:
            raise Exception("Number of channels is not equal to " + str(self.kernel_num))
        return x_2


## Final model
class Resnet_50(tf.keras.Model):
    def __init__(self):
        super(Resnet_50, self).__init__()
        self.block_1 = Stage_1()
        self.block_2 = Stack_stages()
        self.block_3 = Block_Dense()
    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.block_3(x)
        return x



