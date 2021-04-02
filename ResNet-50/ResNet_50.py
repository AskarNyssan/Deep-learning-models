# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 18:07:13 2021

@author: NysanAskar
"""

"""ResNet models for Keras.
Reference:
  - [Deep Residual Learning for Image Recognition](
      https://arxiv.org/abs/1512.03385) (CVPR 2015)
"""
import os
# change current working directory
os.chdir('C:\\Users\\NysanAskar\\Desktop\\Deep Learning\\Deep learning projects\\4. ResNet-50')


import tensorflow as tf
import pathlib
from Utility_resnet_50 import Resnet_50
tf.config.run_functions_eagerly(True)

# Load flower dataset from a folder
data_dir = 'C:\\Users\\NysanAskar\\Desktop\\Deep Learning\\Deep learning projects\\Dataset\\flowers'
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# Determine batch size and preprocess dataset
batch_size = 3
img_height = 299
img_width = 299

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)


# Print class names
class_names = train_ds.class_names
print(class_names)

# Print shapes of feature nd label datasets
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# Configure the dataset for performance. Data.prefetch() overlaps data preprocessing and 
# model execution while training.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Train the model
Resnet_50_model = Resnet_50()
Resnet_50_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
Resnet_50_model.fit(train_ds, epochs=2)
Resnet_50_model.summary()











