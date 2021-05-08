# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:17:54 2021

@author: NysanAskar
"""
"""Inception-V3 model for Tensorflow and Keras.
Reference:
  - [Rethinking the Inception Architecture for Computer Vision](
      http://arxiv.org/abs/1512.00567) (CVPR 2016)
"""

import tensorflow as tf
import pathlib
from Utility_Inception_V3 import Inception_V3
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
Inception_V3_model = Inception_V3()
Inception_V3_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
Inception_V3_model.fit(train_ds, epochs=1)
Inception_V3_model.summary()