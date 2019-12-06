#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:17:57 2017

@author: nehaboob
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 22:26:38 2017

@author: nehaboob
"""

import os
import numpy as np
import shutil

root = './'
#root_train = './train_split'
#root_val = './val_split'
root_total = './train'
n_batch = 10

FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

nbr_train_samples = 0
nbr_val_samples = 0

# Training proportion
split_proportion = 0.8

for idx in range(n_batch):
    print('{}th Batch for split ...'.format(idx))
    nbr_train_samples = 0
    nbr_val_samples = 0
    train_dir = 'train_split_'+str(idx+1)
    val_dir = 'val_split_'+str(idx+1)
    root_train = os.path.join(root, train_dir)
    root_val = os.path.join(root, val_dir)
    os.mkdir(root_train)
    os.mkdir(root_val)
    np.random.seed(idx)
    for fish in FishNames:
        if fish not in os.listdir(root_train):
            os.mkdir(os.path.join(root_train, fish))
    
        total_images = os.listdir(os.path.join(root_total, fish))
    
        nbr_train = int(len(total_images) * split_proportion)
    
        np.random.shuffle(total_images)
    
        train_images = total_images[:nbr_train]
    
        val_images = total_images[nbr_train:]
    
        for img in train_images:
            source = os.path.join(root_total, fish, img)
            target = os.path.join(root_train, fish, img)
            shutil.copy(source, target)
            nbr_train_samples += 1
    
        if fish not in os.listdir(root_val):
            os.mkdir(os.path.join(root_val, fish))
    
        for img in val_images:
            source = os.path.join(root_total, fish, img)
            target = os.path.join(root_val, fish, img)
            shutil.copy(source, target)
            nbr_val_samples += 1
    
    print('Finish splitting train and val images!')
    print('# training samples: {}, # val samples: {}'.format(nbr_train_samples, nbr_val_samples))