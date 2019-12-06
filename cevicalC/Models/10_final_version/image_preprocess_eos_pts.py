#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 23:25:02 2017

@author: neha

resize images to 256X256 and apply CLAHE

"""
import os
from PIL import Image
import cv2
import numpy as np
from PIL import ImageFile, Image

#for large image size
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

Ctypes = ['Type_1', 'Type_2', 'Type_3']
Ctypes = ['test']
source = '../../data/test_stg2/'
traget = '../../data/test_stg2_clahe_256px/'
img_size = 256

for ctype in Ctypes:
    if ctype not in os.listdir(traget):
        os.mkdir(os.path.join(traget, ctype))
    
    items = os.listdir(os.path.join(source, ctype))
    for names in items:
        if names.endswith(".jpg"):
            print(names)
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
            
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            cv2.imwrite(os.path.join(traget, ctype, names), final)