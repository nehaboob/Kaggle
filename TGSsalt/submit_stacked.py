#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 11:58:53 2018

@author: neha

code for sumbission file creation
"""


import pandas as pd
import numpy as np

def rle_decode(mask_rle):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    img = np.zeros(101*101, dtype=np.uint8)
    if(not pd.isnull(mask_rle)):
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
    return img.reshape(101,101)

def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)



pred_dict = {}
df_1 = pd.read_csv('/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/submission_7.csv')
df_2 = pd.read_csv('/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/submission_5.csv')
submission_file = 'submission_5_7_2.csv'

for index,ids in enumerate(df_1.id):
        img_1=rle_decode(df_1.loc[df_1.id == ids,'rle_mask'].values[0])
        img_2=rle_decode(df_2.loc[df_2.id == ids,'rle_mask'].values[0])

        pred_dict[ids] = rle_encode(((img_1+img_2)/2) > 0.50)

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']

sub.to_csv(submission_file)

