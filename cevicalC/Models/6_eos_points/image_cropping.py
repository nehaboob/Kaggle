#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 17:50:15 2017

@author: neha
"""

import cv2
import numpy as np

from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from vis.utils import utils
from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam
from vis.visualization import visualize_activation, get_num_filters
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import math
import os
import statistics

#for large image size
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# set the vairables

img_width = 150
img_height = 150
target_width = 200
target_height = 200
crop_size = 1000
batch_size = 32
nbr_test_samples = 512
val_steps = 16
train_step = 47
nbr_augmentation = 10
n_folds = 1

Ctypes = ['Type_1', 'Type_2', 'Type_3', 'p1x', 'p1y', 'midx', 'midy', 'p2x', 'p2y', 'is_circle']

 
weights_path = 'eos_points_deepsense_2_ep100_150X150.h5'
root_path = '.'
train_dir = '../../train/'
test_dir = '../../test_stg1/'
train_out_file = 'train_pred_150X150.csv'
test_out_file = 'name.csv'
crop_dir = '../../train_crop_auto/'
local_color_dir = '../../train_local_color/'
folders = ['Type_1', 'Type_2', 'Type_3']

#sigmoid funtion
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# get the model for checking saliency map
def init_model(weights_path):
    root_path = '.'
    weights_path = os.path.join(root_path,weights_path)
    model = load_model(weights_path)
    print('Model loaded.')
    print(model.input)
    return model

def generate_train_data_prediction(root_path, train_dir, train_out_file, model):
    train_data_dir = os.path.join(root_path, train_dir)
    
    # test data generator for prediction
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.01,
        width_shift_range=0.01,
        height_shift_range=0.01,
        horizontal_flip=False,
        vertical_flip=False)
    
    random_seed = np.random.random_integers(0, 100000)
        
    train_generator = train_datagen.flow_from_directory(
                    train_data_dir,
                    target_size=(img_width, img_height),
                    batch_size=batch_size,
                    shuffle = False, # Important !!!
                    seed = random_seed,
                    classes = None,
                    class_mode = None)
        
    train_image_list = train_generator.filenames
    print('Begin to predict for training data ...')
    predictions = model.predict_generator(train_generator, train_step)
    #print(predictions.shape)
    
    y_pred = {}
    print('Begin to write submission file ..')
    f_submit = open(os.path.join(root_path, train_out_file), 'w')
    f_submit.write('image_name,Type_1,Type_2,Type_3,p1x,p1y,midx,midy,p2x,p2y,is_circle\n')
    for i, image_name in enumerate(train_image_list):
        pred = ['%.6f' % p for p in predictions[i, :]]
        if i % 100 == 0:
            print('{} / {}'.format(i, nbr_test_samples))
        f_submit.write('%s,%s\n' % (image_name, ','.join(pred)))
        y_pred[image_name] = pred
    f_submit.close()    
    print('Submission file successfully generated!')
    return y_pred

def view_eos(path, eos_values, out_img):
    collage = []
    
    items = os.listdir(root_path)
    image_paths = []
    for names in items:
        if names.endswith(".jpg"):
            image_paths.append(root_path+"/"+names)
    print(image_paths)
    
    for path in image_paths:
        img = Image.open(path+key)
        orig_w = img.size[0]
        orig_h = img.size[1]
        x_ratio = img_width/orig_w
        y_ratio = img_height/orig_h
        keras_img = image.load_img(path, target_size=(img_width, img_height))
        seed_img = np.array([img_to_array(keras_img)])
        cv2.circle(seed_img,(eos_values[path][0]*x_ratio,eos_values[path][1]*y_ratio), 3, (0,0,0), -1)
        cv2.putText(seed_img,key, (0,0), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        collage.append(seed_img)
    # apply local color normalization
for ctype in folders:
    items = os.listdir(os.path.join(crop_dir, ctype))
    for names in items:
        if names.endswith(".jpg"):
            img = cv2.imread(os.path.join(crop_dir, ctype, names))
            img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 20), -4, 128)
            cv2.imwrite(os.path.join(local_color_dir, ctype, names), img)

    cv2.imwrite(out_img,utils.stitch_images(collage))


# load the model
model = init_model(weights_path)
#model.summary()

y_pred = generate_train_data_prediction(root_path, train_dir, train_out_file, model)

train_zoom = {}
eos_list = []
# get the essenstial features for image zooming
for key, value in y_pred.items():
    img = Image.open(train_dir+key)
    orig_w = img.size[0]
    orig_h = img.size[1]
    x_ratio = img_width/orig_w
    y_ratio = img_height/orig_h
    
    p1x = float(value[3])/x_ratio
    p1y = float(value[4])/y_ratio
    midx = float(value[5])/x_ratio
    midy = float(value[6])/y_ratio
    p2x = float(value[7])/x_ratio
    p2y = float(value[8])/y_ratio
    
    eosx = (p1x + midx + p2x)/3
    eosy = (p1y + midy + p2y)/3

    diag = math.sqrt(orig_w ** 2 + orig_h ** 2)
    eos_size = (math.sqrt((p1x - midx) ** 2 + (p1y - midy) **2) + math.sqrt((p2x - midx) ** 2 + (p2y - midy) **2) + math.sqrt((p1x - p2x) ** 2 + (p1y - p2y) **2))/3
    n_eos = eos_size/diag
    train_zoom[key] = [eosx, eosy, n_eos, orig_w, orig_h, eos_size, diag]
    eos_list.append(n_eos)

f_submit = open('train_eos_150.csv', 'w')
f_submit.write('image_name,eosx,eosy,eos_size_normalized,orig_w,orig_h,eos_size,diag\n')
for key, value in train_zoom.items():
    val = ['%.6f' % p for p in value[:]]
    f_submit.write('%s,%s\n' % (key, ','.join(val)))
f_submit.close()    

#get the median of the nomalized eos sizes
normalized_eos_median = statistics.median(eos_list)

#apply zoom and crop image from center   
for key, value in train_zoom.items():
        img = Image.open(train_dir+key)
        eos_size = value[2]
        eos_factor = normalized_eos_median/eos_size
        if(eos_factor >= 1):
            zoom_factor=0.022*eos_factor+0.978
        else:
            zoom_factor=0.5*eos_factor+0.5
        
        print(key+"-"+str(zoom_factor)+"-"+str(eos_factor))
        new_w = img.size[0]*zoom_factor
        new_h = img.size[1]*zoom_factor
        #apply zoom on the image which should be centered around eos and get new location of eos after zoom
        img = img.resize((int(new_w), int(new_h)))
        eos_x = value[0]*zoom_factor
        eos_y = value[1]*zoom_factor
        
        # crop just 1000X1000 image around eos and resize it again to the trage size
        img = img.crop((eos_x-(crop_size/2), eos_y-(crop_size/2), eos_x+(crop_size/2), eos_y+(crop_size/2)))
        img = img.resize((target_width, target_height))
        img.save(crop_dir+key)  
        
    
    # handel the case where there are not sufficient pixels for cropping around eos

# apply local color normalization
for ctype in folders:
    items = os.listdir(os.path.join(crop_dir, ctype))
    for names in items:
        if names.endswith(".jpg"):
            img = cv2.imread(os.path.join(crop_dir, ctype, names))
            img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 20), -4, 128)
            cv2.imwrite(os.path.join(local_color_dir, ctype, names), img)
