# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 21:44:56 2021

@author: NysanAskar
"""

import tensorflow as tf
from Utility_AlexNet import AlexNet
import pathlib


# Load flower dataset from a folder
data_dir = 'C:\\Users\\NysanAskar\\Desktop\\Deep Learning\\Deep learning projects\\1. AlexNet\\\Dataset\\flowers'
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# Determine batch size and preprocess dataset
batch_size = 32
img_height = 227
img_width = 227

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
AlexNet_model = AlexNet()
AlexNet_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
AlexNet_model.fit(train_ds, epochs=10)

