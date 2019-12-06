#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 00:17:47 2018

@author: neha
"""

import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import os

def IOU_scoring(y_true, y_pred):
    y_true = set(y_true)
    y_pred = set(y_pred)
    len_intersection = len(y_true.intersection(y_pred))
    len_union = len(y_true.union(y_pred))
    
    if(len(y_true) == 0):
        if(len(y_pred) == 0):
            score = 1
        else:
            score = 0
    else: 
        IoU = len_intersection/len_union
    
        threshholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
        IoU_thresh = [IoU > x for x in threshholds] 
        score = sum(IoU_thresh)/10
    
    return score

# read all the files and ouput distinct image size
# check the depth file
# number of empty masks
# relationship between surrounding pixels for score improvement
lst = pd.Series()
train = os.listdir('/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train/images')
for img in train:
    im = Image.open(img)
    width, height = im.size
    lst = lst.append(pd.Series(str(width)+'_'+str(height)))

lst = pd.Series()
train = os.listdir('/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train/masks')
for img in train:
    im = Image.open(img)
    width, height = im.size
    lst = lst.append(pd.Series(str(width)+'_'+str(height)))
        
lst = pd.Series()
test = os.listdir('/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/test')
for img in test:
    im = Image.open(img)
    width, height = im.size
    lst = lst.append(pd.Series(str(width)+'_'+str(height)))
    
# all images are 101 X 101
depth = pd.read_csv('depths.csv')
depth.describe()
sns.distplot(depth.z)

# for each mask image number of white px
im = Image.open('/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train/masks/00a3af90ab.png')
im_array = np.array(im)
im_array = np.transpose(im_array)
im_array = im_array.ravel()
    

# create simple keras CNN to predict the masks  