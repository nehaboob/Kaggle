#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:36:57 2017

@author: neha
"""

import os
import numpy as np
from skimage.exposure import equalize_hist
from skimage import color
from skimage import io

#for large image size
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

np.random.seed(108)

root_train = './train_split'
root_val = './val_split'
grey_train ='./train_splitG'
grey_val = './val_splitG'
Ctypes = ['Type_1', 'Type_2', 'Type_3']
nbr_train_samples = 0
nbr_val_samples = 0

os.mkdir(grey_train)
os.mkdir(grey_val)

for ctype in Ctypes:
    if ctype not in os.listdir(grey_train):
        os.mkdir(os.path.join(grey_train, ctype))
        
    if ctype in os.listdir(root_train):
        train_images = os.listdir(os.path.join(root_train, ctype))
        for img in train_images:
            if img.endswith(".jpg"):
                source = os.path.join(root_train, ctype, img)
                target = os.path.join(grey_train, ctype, img)
                img = color.rgb2gray(io.imread(source))
                equalized_image = equalize_hist(img)
                io.imsave(target, equalized_image)
                nbr_train_samples += 1
        
    if ctype not in os.listdir(grey_val):
        os.mkdir(os.path.join(grey_val, ctype))

    if ctype in os.listdir(root_val):
        val_images = os.listdir(os.path.join(root_val, ctype))
        for img in val_images:
            if img.endswith(".jpg"):
                source = os.path.join(root_val, ctype, img)
                target = os.path.join(grey_val, ctype, img)
                img = color.rgb2gray(io.imread(source))
                equalized_image = equalize_hist(img)
                io.imsave(target, equalized_image)
                nbr_val_samples += 1

print('Finish splitting train and val images!')
print('# training samples: {}, # val samples: {}'.format(nbr_train_samples, nbr_val_samples))


# test images

test_dir = './test_stg1'
target_test_dir = './test_stg1G'
nbr_test_samples = 0
os.mkdir(target_test_dir)

Ctypes = ['test']


for ctype in Ctypes:
    if ctype not in os.listdir(target_test_dir):
        os.mkdir(os.path.join(target_test_dir, ctype))

    if ctype in os.listdir(test_dir):
        test_images = os.listdir(os.path.join(test_dir, ctype))
        for img in test_images:
            if img.endswith(".jpg"):
                source = os.path.join(test_dir, ctype, img)
                target = os.path.join(target_test_dir, ctype, img)
                img = color.rgb2gray(io.imread(source))
                equalized_image = equalize_hist(img)
                io.imsave(target, equalized_image)
                nbr_test_samples += 1      

print('Finish splitting test images!')
print('# training samples: {}'.format(nbr_test_samples))

