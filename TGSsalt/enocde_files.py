#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 18:29:18 2018

@author: neha

enocode prediction files
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
from keras.models import Model, load_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from keras import backend as K
import time 
import tensorflow as tf
import h5py
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img #,save_img


train_df = pd.read_csv("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

preds_test0 = np.load('/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/old_predictions/preds_test_rev_sgd_lh_aug_0.npy')    
pred_dict = {idx: rle_encode(preds_test0[i]) for i, idx in enumerate(test_df.index.values)}


