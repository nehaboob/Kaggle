#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 20:58:20 2018

@author: neha
"""

import os
import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")
from sklearn.model_selection import train_test_split
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize    
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img #,save_img
import tensorflow as tf
from sklearn.model_selection import KFold

# Set some parameters
im_width = 101
im_height = 101
im_chan = 1
basicpath = '/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/'
path_train = basicpath + 'train/'
path_test = basicpath + 'test/'

path_train_images = path_train + 'images/'
path_train_masks = path_train + 'masks/'
path_test_images = path_test + 'images/'

img_size_ori = 101
img_size_target = 128

# Loading of training/testing ids and depths

train_df = pd.read_csv("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

len(train_df)

train_df["images"] = [np.array(load_img("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]
train_df["masks"] = [np.array(load_img("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]
train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)

def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i
        
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

train_images = train_df.index.values
kf = KFold(n_splits=10,random_state=88)
for i, (train_index, valid_index) in enumerate(kf.split(train_images)):
    train_fold = train_images[train_index]
    valid_fold = train_images[valid_index]
    pd.Series(train_fold).to_csv('train_fold_new_'+str(i)+'.csv')
    pd.Series(valid_fold).to_csv('valid_fold_new_'+str(i)+'.csv')    
    
