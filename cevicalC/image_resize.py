#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 23:02:12 2017

@author: neha
"""
import os
from PIL import Image
import cv2
import numpy as np
from PIL import ImageFile, Image

#for large image size
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# resize images
img_size = 150
folders = ['Type_1', 'Type_2', 'Type_3']
train_dir = '../../data/train_split_150px/'
val_dir = '../../data/val_split_150px/'
train_lc_dir = '../../data/train_split_lc_150px/'
val_lc_dir = '../../data/val_split_lc_150px/'
train_split = '../../data/train_split/'
val_split = '../../data/val_split/'

Ctypes = ['Type_1', 'Type_2', 'Type_3']

#train resize
os.mkdir(train_dir)
source = train_split
traget = train_dir
for ctype in Ctypes:
    if ctype not in os.listdir(traget):
        os.mkdir(os.path.join(traget, ctype))
    
    items = os.listdir(os.path.join(source, ctype))
    for names in items:
        if names.endswith(".jpg"):
            img = Image.open(source+ctype+'/'+names)
            img = img.resize((img_size, img_size))
            img.save(traget+ctype+'/'+names) 

#val resize
os.mkdir(val_dir)
source = val_split
traget = val_dir
for ctype in Ctypes:
    if ctype not in os.listdir(traget):
        os.mkdir(os.path.join(traget, ctype))
    
    items = os.listdir(os.path.join(source, ctype))
    for names in items:
        if names.endswith(".jpg"):
            img = Image.open(source+ctype+'/'+names)
            img = img.resize((img_size, img_size))
            img.save(traget+ctype+'/'+names) 
            

#diff the files
'''
diff  <(ls -1a train_split/Type_1) <(ls -1a train_split_clahe_256px/Type_1)
diff  <(ls -1a train_split/Type_2) <(ls -1a train_split_clahe_256px/Type_2)
diff  <(ls -1a train_split/Type_3) <(ls -1a train_split_clahe_256px/Type_3)
diff  <(ls -1a val_split/Type_1) <(ls -1a val_split_clahe_256px/Type_1)
diff  <(ls -1a val_split/Type_2) <(ls -1a val_split_clahe_256px/Type_2)
diff  <(ls -1a val_split/Type_3) <(ls -1a val_split_clahe_256px/Type_3)
'''

#subtract the local color
# apply local color normalization
os.mkdir(train_lc_dir)
source = train_dir
traget = train_lc_dir

for ctype in folders:
    if ctype not in os.listdir(traget):
        os.mkdir(os.path.join(traget, ctype))
    
    items = os.listdir(os.path.join(source, ctype))
    for names in items:
        if names.endswith(".jpg"):
            img = cv2.imread(os.path.join(source, ctype, names))
            img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 20), -4, 128)
            cv2.imwrite(os.path.join(traget, ctype, names), img)
            
os.mkdir(val_lc_dir)
source = val_dir
traget = val_lc_dir

for ctype in folders:
    if ctype not in os.listdir(traget):
        os.mkdir(os.path.join(traget, ctype))
    
    items = os.listdir(os.path.join(source, ctype))
    for names in items:
        if names.endswith(".jpg"):
            img = cv2.imread(os.path.join(source, ctype, names))
            img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 20), -4, 128)
            cv2.imwrite(os.path.join(traget, ctype, names), img)
            

#save lcolor image as 256X256
img_size = 256
train_lc_dir = '../../data/train_split_clahe_256px/'
val_lc_dir = '../../data/val_split_clahe_256px/'
train_split = '../../data/train_split/'
val_split = '../../data/val_split/'
test_stg1 =  '../../data/test_stg1/'
test_stg1_lc_dir = '../../data/test_stg1_split_clahe_256px/'

folders = ['Type_1', 'Type_2', 'Type_3']


#normalize brightness
def norm_image(img):
    """
    Normalize PIL image
    
    Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
    """
    img_y, img_b, img_r = img.convert('YCbCr').split()
    
    img_y_np = np.asarray(img_y).astype(float)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                    np.abs(np.percentile(img_y_np, 99.0))])
    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0
    
    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)

    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))
    
    img_nrm = img_ybr.convert('RGB')
    
    return img_nrm


os.mkdir(train_lc_dir)
source = train_split
traget = train_lc_dir

for ctype in folders:
    if ctype not in os.listdir(traget):
        os.mkdir(os.path.join(traget, ctype))
    
    items = os.listdir(os.path.join(source, ctype))
    for names in items:
        if names.endswith(".jpg"):
            #img = Image.open(os.path.join(source, ctype, names))
            #img = norm_image(img)
            #img.save(os.path.join(traget, ctype, names))
            img = cv2.imread(os.path.join(source, ctype, names))
            img = cv2.resize(img, (img_size, img_size))
            lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            #-----Splitting the LAB image to different channels-------------------------
            l, a, b = cv2.split(lab)

            #-----Applying CLAHE to L-channel-------------------------------------------
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
            limg = cv2.merge((cl,a,b))
            
            #-----Converting image from LAB Color model to RGB model--------------------
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            #img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), img_size/30), -4, 128)
            cv2.imwrite(os.path.join(traget, ctype, names), final)
            
os.mkdir(val_lc_dir)
source = val_split
traget = val_lc_dir

for ctype in folders:
    if ctype not in os.listdir(traget):
        os.mkdir(os.path.join(traget, ctype))
    
    items = os.listdir(os.path.join(source, ctype))
    for names in items:
        if names.endswith(".jpg"):
            #img = Image.open(os.path.join(source, ctype, names))
            #img = norm_image(img)
            #img.save(os.path.join(traget, ctype, names))
            
            #img = cv2.imread(os.path.join(source, ctype, names))
            #img = cv2.resize(img, (img_size, img_size))
            #img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), img_size/30), -4, 128)
            #cv2.imwrite(os.path.join(traget, ctype, names), img)
            img = cv2.imread(os.path.join(source, ctype, names))
            img = cv2.resize(img, (img_size, img_size))
            lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            #-----Splitting the LAB image to different channels-------------------------
            l, a, b = cv2.split(lab)

            #-----Applying CLAHE to L-channel-------------------------------------------
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
            limg = cv2.merge((cl,a,b))
            
            #-----Converting image from LAB Color model to RGB model--------------------
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            #img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), img_size/30), -4, 128)
            cv2.imwrite(os.path.join(traget, ctype, names), final)
            
# test resize
source = test_stg1
traget = test_stg1_lc_dir

for ctype in ['test']:
    if ctype not in os.listdir(traget):
        os.mkdir(os.path.join(traget, ctype))
    
    items = os.listdir(os.path.join(source, ctype))
    for names in items:
        if names.endswith(".jpg"):
            #img = Image.open(os.path.join(source, ctype, names))
            #img = norm_image(img)
            #img.save(os.path.join(traget, ctype, names))
            
            #img = cv2.imread(os.path.join(source, ctype, names))
            #img = cv2.resize(img, (img_size, img_size))
            #img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), img_size/30), -4, 128)
            #cv2.imwrite(os.path.join(traget, ctype, names), img)
            img = cv2.imread(os.path.join(source, ctype, names))
            img = cv2.resize(img, (img_size, img_size))
            lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            #-----Splitting the LAB image to different channels-------------------------
            l, a, b = cv2.split(lab)

            #-----Applying CLAHE to L-channel-------------------------------------------
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
            limg = cv2.merge((cl,a,b))
            
            #-----Converting image from LAB Color model to RGB model--------------------
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            #img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), img_size/30), -4, 128)
            cv2.imwrite(os.path.join(traget, ctype, names), final)