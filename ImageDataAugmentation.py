#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 19:16:20 2020

@author: rickysu
"""

import numpy as np 
import os
import pandas as pd
import matplotlib.pyplot as plt
import random

'''
How to set up your directory...

- train
	- class1
	- class2
	- class3
- val
	- class1
	- class2
	- class3
'''

# Some snippets to create those classes above
###
# Scenario 1: All images are in one folder called 'train/'. Create 'val/' folder and all classes within both 'train/' and 'val/'.
#			  Randomly select images to move from 'train/' to 'val/'.
TRAINING_PATH = "./train/"
VAL_PATH = "./val/"

###########################################################################
from tensorflow.keras.preprocessing.image import ImageDataGenerator

trainAug = ImageDataGenerator()
valAug = ImageDataGenerator()

BATCH_SIZE = 64

# initialize the training generator
trainGen = trainAug.flow_from_directory(
	TRAINING_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=True,
	batch_size=BATCH_SIZE)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	VAL_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=BATCH_SIZE)

# Set labels
num_classes = X


# Start the model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False
    
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])


H = model.fit(
	x=trainGen,
	validation_data=valGen,
	epochs=10,
    verbose=1)



