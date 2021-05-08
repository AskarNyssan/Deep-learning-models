# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 12:19:01 2021

@author: NysanAskar
"""

import tensorflow as tf
from tensorflow import keras

### Layer 1
class Conv2D_Layer_1(tf.keras.layers.Layer):
    def __init__(self, kernel_num=96, kernel_size=(11,11), strides=4, padding='valid'):
        super(Conv2D_Layer_1, self).__init__()
        self.conv = tf.keras.layers.Conv2D(kernel_num, input_shape = (64, 227, 227,3),
                        kernel_size=kernel_size, 
                        strides=strides, padding=padding)
        self.Max_Pooling = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = tf.nn.relu(x)
        x = self.Max_Pooling(x)
        return x


### Layer 2
class Conv2D_Layer_2(tf.keras.layers.Layer):
    def __init__(self, kernel_num=256, kernel_size=(5,5), padding='valid'):
        super(Conv2D_Layer_2, self).__init__()
        self.conv = tf.keras.layers.Conv2D(kernel_num, 
                        kernel_size=kernel_size, 
                   padding=padding)
        self.Max_Pooling = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)
        self.padd = tf.keras.layers.ZeroPadding2D(padding=(2,2))

    def call(self, input_tensor, training=False):
        x = self.padd(input_tensor)
        x = self.conv(x)
        x = tf.nn.relu(x)
        x = self.Max_Pooling(x)
        return x


### Layer 3
class Conv2D_Layer_3(tf.keras.layers.Layer):
    def __init__(self, kernel_num=384, kernel_size=(3,3), padding='valid'):
        super(Conv2D_Layer_3, self).__init__()
        self.conv = tf.keras.layers.Conv2D(kernel_num, 
                        kernel_size=kernel_size, 
                   padding=padding)
        self.padd = tf.keras.layers.ZeroPadding2D(padding=(1,1))

    def call(self, input_tensor, training=False):
        x = self.padd(input_tensor)
        x = self.conv(x)
        x = tf.nn.relu(x)
        return x


### Layer 4
class Conv2D_Layer_4(tf.keras.layers.Layer):
    def __init__(self, kernel_num=384, kernel_size=(3,3), padding='valid'):
        super(Conv2D_Layer_4, self).__init__()
        self.conv = tf.keras.layers.Conv2D(kernel_num, 
                        kernel_size=kernel_size, 
                   padding=padding)
        self.padd = tf.keras.layers.ZeroPadding2D(padding=(1,1))

    def call(self, input_tensor, training=False):
        x = self.padd(input_tensor)
        x = self.conv(x)
        x = tf.nn.relu(x)
        return x


### Layer 5
class Conv2D_Layer_5(tf.keras.layers.Layer):
    def __init__(self, kernel_num=256, kernel_size=(3,3), padding='valid'):
        super(Conv2D_Layer_5, self).__init__()
        self.conv = tf.keras.layers.Conv2D(kernel_num, 
                        kernel_size=kernel_size, 
                   padding=padding)
        self.Max_Pooling = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)
        self.padd = tf.keras.layers.ZeroPadding2D(padding=(1,1))

    def call(self, input_tensor, training=False):
        x = self.padd(input_tensor)
        x = self.conv(x)
        x = tf.nn.relu(x)
        x = self.Max_Pooling(x)
        x = tf.keras.layers.Flatten()(x)
        return x


### Layer 6
class Linear_Layer_6(keras.layers.Layer):
    def __init__(self, output_units=4096):
        super(Linear_Layer_6, self).__init__()
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


### Layer 7
class Linear_Layer_7(keras.layers.Layer):
    def __init__(self, output_units=4096):
        super(Linear_Layer_7, self).__init__()
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


### Layer 8
class Linear_Layer_8(keras.layers.Layer):
    def __init__(self, output_units=10):
        super(Linear_Layer_8, self).__init__()
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



class AlexNet(tf.keras.Model):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.layer_1 = Conv2D_Layer_1()
        self.layer_2 = Conv2D_Layer_2()
        self.layer_3 = Conv2D_Layer_3()
        self.layer_4 = Conv2D_Layer_4()
        self.layer_5 = Conv2D_Layer_5()
        self.layer_6 = Linear_Layer_6()
        self.layer_7 = Linear_Layer_7()
        self.layer_8 = Linear_Layer_8()

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        return x

