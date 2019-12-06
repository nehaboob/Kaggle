#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 09:57:34 2017

@author: neha
"""

import os
import numpy as np
import shutil

np.random.seed(108)

root_fold_1 = '../../data/train_split_clahe_256px_crop_auto_lcolor/Type_2/Fold_1'
root_fold_2 = '../../data/val_split_clahe_256px_crop_auto_lcolor/Type_2/Fold_2'
root_fold_3 = '../../data/val_split_clahe_256px_crop_auto_lcolor/Type_2/Fold_3'

root_total = '../../data/train_split_clahe_256px_crop_auto_lcolor/Type_2'

Ctypes = ['Type_1', 'Type_2', 'Type_3']

nbr_fold_1 = 0
nbr_fold_2 = 0
nbr_fold_3 = 0

# Training proportion
split_proportion = 0.3



for ctype in Ctypes:

    total_images = os.listdir(os.path.join(root_total, ctype))

    nbr_train = int(len(total_images) * split_proportion)

    np.random.shuffle(total_images)

    fold_1 = total_images[0, nbr_train]
    fold_2 = total_images[nbr_train, nbr_train*2]
    fold_3 = total_images[nbr_train*2:]


    for img in fold_1:
        source = os.path.join(root_total, ctype, img)
        target = os.path.join(root_fold_1, ctype, img)
        shutil.copy(source, target)
        nbr_fold_1 += 1

    for img in fold_2:
        source = os.path.join(root_total, ctype, img)
        target = os.path.join(root_fold_2, ctype, img)
        shutil.copy(source, target)
        nbr_fold_2 += 1
        
    for img in fold_3:
        source = os.path.join(root_total, ctype, img)
        target = os.path.join(root_fold_3, ctype, img)
        shutil.copy(source, target)
        nbr_fold_3 += 1        


print('Finish splitting train and val images!')
print('# fold_1 samples: {}, # fold_2 samples: {}, # fold_2 samples: {}'.format(nbr_fold_1, nbr_fold_2, nbr_fold_3))
