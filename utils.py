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


