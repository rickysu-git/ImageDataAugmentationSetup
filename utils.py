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
SPLIT = 0.80
TRAINING_PATH = "./train/"

VAL_PATH = "./val/"
os.mkdir(VAL_PATH)

# Get number of images for train/val based on SPLIT variable above
total_num_images = len(os.listdir(TRAINING_PATH))
total_train = int(SPLIT*total_num_images)
total_val = total_num_images-total_train

# Move images to "./val/" from "./train/"
for _ in range(total_val):
    image_name = random.choice(os.listdir(TRAINING_PATH))
    os.rename(TRAINING_PATH + image_name, 
              VAL_PATH + image_name)

# Create classes
labels_dict = {'imageID1':1,'imageID2':2,'imageID3':3}
for imageID in os.listdir(TRAINING_PATH):
	label = labels_dict[imageID]
	if os.isdir(TRAINING_PATH + label):
		os.mkdir(TRAINING_PATH + label)
    os.rename(TRAINING_PATH + image_name, 
              TRAINING_PATH + label + "/" + image_name)

for imageID in os.listdir(VAL_PATH):
	label = labels_dict[imageID]
	if os.isdir(VAL_PATH + label):
		os.mkdir(VAL_PATH + label)
    os.rename(VAL_PATH + image_name, 
              VAL_PATH + label + "/" + image_name)


###########################################################################
trainPath = "/Users/rickysu/Desktop/kaggle/cassava/train_images"
valPath = "/Users/rickysu/Desktop/kaggle/cassava/val_images"
totalTrain = len(os.listdir(trainPath))
totalVal = len(os.listdir(valPath))


# Start ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

trainAug = ImageDataGenerator()
valAug = ImageDataGenerator()

# initialize the training generator
trainGen = trainAug.flow_from_directory(
	trainPath,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=True,
	batch_size=64)
# initialize the validation generator
valGen = valAug.flow_from_directory(
	valPath,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=64)


# Start the model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

labels = train_df['label'].tolist()

base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

x = base_model.output
x = Flatten()(x)
# Add a fully-connected layer
x = Dense(128, activation='relu')(x)
predictions = Dense(len(set(labels)), activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False
    
model.compile(loss="categorical_crossentropy", optimizer='adam',
	metrics=["accuracy"])


print("[INFO] training head...")
H = model.fit(
	x=trainGen,
	steps_per_epoch=total_train // 64,
	validation_data=valGen,
	validation_steps=total_val // 64,
	epochs=10,
    verbose=1)

# Helpers
def plot_training(H, N, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)
    
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

labels = train_csv['label'].tolist()
from sklearn.preprocessing import OneHotEncoder

# One hot encode the labels
enc = OneHotEncoder()
y_encoded = enc.fit_transform(np.array(labels).reshape(-1,1)).toarray()


train_test_split = int(len(all_images)*0.80)

x_train = all_images_array[:train_test_split]
y_train = y_encoded[:train_test_split]
x_test = all_images_array[train_test_split:]
y_test = y_encoded[train_test_split:]





# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model on the new data for a few epochs
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1)









