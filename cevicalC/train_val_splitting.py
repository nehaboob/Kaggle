#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:42:05 2017

@author: neha
"""
import os
import numpy as np
import shutil

np.random.seed(108)

root_train = '../../data/additional/train_split_crop_auto40_all_lcolor'
root_val = '../../data/additional/val_split_crop_auto40_all_lcolor'

root_total = '../../data/additional/train_crop_auto40_all_lcolor'

Ctypes = ['Type_1', 'Type_2', 'Type_3']

nbr_train_samples = 0
nbr_val_samples = 0

# Training proportion
split_proportion = 0.9

os.mkdir(root_train)
os.mkdir(root_val)

for ctype in Ctypes:
    if ctype not in os.listdir(root_train):
        os.mkdir(os.path.join(root_train, ctype))

    total_images = os.listdir(os.path.join(root_total, ctype))

    nbr_train = int(len(total_images) * split_proportion)

    np.random.shuffle(total_images)

    train_images = total_images[:nbr_train]

    val_images = total_images[nbr_train:]

    for img in train_images:
        source = os.path.join(root_total, ctype, img)
        target = os.path.join(root_train, ctype, img)
        shutil.copy(source, target)
        nbr_train_samples += 1

    if ctype not in os.listdir(root_val):
        os.mkdir(os.path.join(root_val, ctype))

    for img in val_images:
        source = os.path.join(root_total, ctype, img)
        target = os.path.join(root_val, ctype, img)
        shutil.copy(source, target)
        nbr_val_samples += 1

print('Finish splitting train and val images!')
print('# training samples: {}, # val samples: {}'.format(nbr_train_samples, nbr_val_samples))





