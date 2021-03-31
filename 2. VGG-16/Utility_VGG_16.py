# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:01:22 2021

@author: NysanAskar
"""

import tensorflow as tf
from tensorflow import keras

### Block 1
class Conv2D_Block_1(tf.keras.layers.Layer):
    def __init__(self, kernel_num=64, kernel_size=(3,3), padding='same'):
        super(Conv2D_Block_1, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(kernel_num, input_shape = (224, 224,3),
                        kernel_size=kernel_size, padding=padding)
        self.conv_2 = tf.keras.layers.Conv2D(kernel_num, 
                        kernel_size=kernel_size, padding=padding)
        self.Max_Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))

    def call(self, input_tensor, training=False):
        x = self.conv_1(input_tensor)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = tf.nn.relu(x)
        x = self.Max_Pooling(x)
        return x


### Block 2
class Conv2D_Block_2(tf.keras.layers.Layer):
    def __init__(self, kernel_num=128, kernel_size=(3,3), padding='same'):
        super(Conv2D_Block_2, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(kernel_num,
                        kernel_size=kernel_size, padding=padding)
        self.conv_2 = tf.keras.layers.Conv2D(kernel_num, 
                        kernel_size=kernel_size, padding=padding)
        self.Max_Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))

    def call(self, input_tensor, training=False):
        x = self.conv_1(input_tensor)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = tf.nn.relu(x)
        x = self.Max_Pooling(x)
        return x


### Block 3
class Conv2D_Block_3(tf.keras.layers.Layer):
    def __init__(self, kernel_num=256, kernel_size=(3,3), padding='same'):
        super(Conv2D_Block_3, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(kernel_num,
                        kernel_size=kernel_size, padding=padding)
        self.conv_2 = tf.keras.layers.Conv2D(kernel_num, 
                        kernel_size=kernel_size, padding=padding)
        self.Max_Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))

    def call(self, input_tensor, training=True):
        x = self.conv_1(input_tensor)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = tf.nn.relu(x)
        x = self.Max_Pooling(x)
        return x


### Block 4
class Conv2D_Block_4(tf.keras.layers.Layer):
    def __init__(self, kernel_num=512, kernel_size=(3,3), padding='same'):
        super(Conv2D_Block_4, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(kernel_num,
                        kernel_size=kernel_size, padding=padding)
        self.conv_2 = tf.keras.layers.Conv2D(kernel_num, 
                        kernel_size=kernel_size, padding=padding)
        self.Max_Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))

    def call(self, input_tensor, training=True):
        x = self.conv_1(input_tensor)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = tf.nn.relu(x)
        x = self.Max_Pooling(x)
        return x


### Block 5
class Conv2D_Block_5(tf.keras.layers.Layer):
    def __init__(self, kernel_num=512, kernel_size=(3,3), padding='same'):
        super(Conv2D_Block_5, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(kernel_num,
                        kernel_size=kernel_size, padding=padding)
        self.conv_2 = tf.keras.layers.Conv2D(kernel_num, 
                        kernel_size=kernel_size, padding=padding)
        self.conv_3 = tf.keras.layers.Conv2D(kernel_num, 
                        kernel_size=(1,1), padding=padding)
        self.Max_Pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))

    def call(self, input_tensor, training=True):
        x = self.conv_1(input_tensor)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = tf.nn.relu(x)
        x = self.conv_3(x)
        x = tf.nn.relu(x)
        x = self.Max_Pooling(x)
        x = tf.keras.layers.Flatten()(x)
        return x



class Linear_Layer_14(keras.layers.Layer):
    def __init__(self, output_units=4096):
        super(Linear_Layer_14, self).__init__()
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
        x = tf.nn.relu(x)
        return x


class Linear_Layer_15(keras.layers.Layer):
    def __init__(self, output_units=4096):
        super(Linear_Layer_15, self).__init__()
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
        x = tf.nn.relu(x)
        return x


class Linear_Layer_16(keras.layers.Layer):
    def __init__(self, output_units=5):
        super(Linear_Layer_16, self).__init__()
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


class VGG_16(tf.keras.Model):
    def __init__(self):
        super(VGG_16, self).__init__()
        self.block_1 = Conv2D_Block_1()
        self.block_2 = Conv2D_Block_2()
        self.block_3 = Conv2D_Block_3()
        self.block_4 = Conv2D_Block_4()
        self.block_5 = Conv2D_Block_5()
        self.layer_14 = Linear_Layer_14()
        self.layer_15 = Linear_Layer_15()
        self.layer_16 = Linear_Layer_16()

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.layer_14(x)
        x = self.layer_15(x)
        x = self.layer_16(x)
        return x


