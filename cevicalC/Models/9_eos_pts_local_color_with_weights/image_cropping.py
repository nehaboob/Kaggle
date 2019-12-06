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
from keras import backend as K
from PIL import Image
import math
import os
import statistics

#for large image size
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# set the vairables

img_width = 100
img_height = 100
target_width = 200
target_height = 200
crop_size = 1000
batch_size = 32
nbr_test_samples = 512
val_steps = 10
train_steps = 37
test_steps = 16
nbr_augmentation = 10
n_folds = 1
eos_side_desired = 30

Ctypes = ['Type_1', 'Type_2', 'Type_3', 'p1x', 'p1y', 'midx', 'midy', 'p2x', 'p2y', 'is_circle']

 
weights_path = 'eos_points_clahe_150X150.h5'
root_path = '.'
train_dir = '../../data/train/'
test_dir = '../../data/test_stg1/'
train_split_dir = '../../data/train_split_clahe_256px/'
val_split_dir = '../../data/val_split_clahe_256px/'
test_split_dir = '../../data/test_stg1_clahe_256px/'
train_out_file = 'train_split_pred_150X150.csv'
val_out_file = 'val_split_pred_150X150.csv'
test_out_file = 'test_split_pred_150X150.csv'
train_crop_dir = '../../data/train_split_clahe_256px_crop_auto_30_normalise/'
val_crop_dir = '../../data/val_split_clahe_256px_crop_auto_30_normalise/'
test_crop_dir = '../../data/test_stg1_clahe_256px_crop_auto_30_normalise/'
train_local_color_dir = '../../data/train_split_clahe_256px_crop_auto_30_lcolor/'
val_local_color_dir = '../../data/val_split_clahe_256px_crop_auto_30_lcolor/'
test_local_color_dir = '../../data/test_stg1_clahe_256px_crop_auto_30_lcolor/'

folders = ['Type_1', 'Type_2', 'Type_3']

#sigmoid funtion
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def actual_mse(y_true, y_pred):
    y_true = np.power(10, y_true)
    y_pred = np.power(10, y_pred)
    return K.mean(K.square(y_pred - y_true), axis=-1)

def actual_abs(y_true, y_pred):
    y_true = np.power(10, y_true)
    y_pred = np.power(10, y_pred)
    return K.mean(K.abs(y_pred - y_true), axis=-1)

# get the model for checking saliency map
def init_model(weights_path):
    root_path = '.'
    weights_path = os.path.join(root_path,weights_path)
    model = load_model(weights_path, custom_objects={'actual_mse': actual_mse, 'actual_abs': actual_abs})
    print('Model loaded.')
    print(model.input)
    return model

#preprocess image 
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def generate_train_data_prediction(root_path, train_dir, train_out_file, model, step):
    train_data_dir = os.path.join(root_path, train_dir)
    
    for fold in range(n_folds):
        print('{}th fold for prediction ...'.format(fold))
        
        for idx in range(nbr_augmentation):
            # test data generator for prediction
            train_datagen = ImageDataGenerator(
                shear_range=0.01,
                width_shift_range=0.01,
                height_shift_range=0.01,
                horizontal_flip=False,
                vertical_flip=False,
                preprocessing_function=preprocess_input)
            
            random_seed = np.random.randint(0, 100001)
                
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
            
            if idx == 0 and fold == 0:
                predictions = model.predict_generator(train_generator, step, verbose=0)
            else:
                predictions += model.predict_generator(train_generator, step, verbose=0)
    #print(predictions.shape)
    predictions /= (nbr_augmentation*n_folds)

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
    
    cv2.imwrite(out_img,utils.stitch_images(collage))


# load the model
model = init_model(weights_path)
#model.summary()
dir1 = train_split_dir
crop_dir = train_crop_dir
out_file = train_out_file
train_dir = train_dir
steps = train_steps
y_pred = generate_train_data_prediction(root_path, dir1, out_file, model, steps)

train_zoom = {}

# get the essenstial features for image zooming
for key, value in y_pred.items():
    img = Image.open(train_dir+key)
    orig_w = img.size[0]
    orig_h = img.size[1]
    x_ratio = img_width/orig_w
    y_ratio = img_height/orig_h
    
    p1x = ((float(value[0])*(img_width/2))+(img_width/2))/x_ratio
    p1y = ((float(value[1])*(img_width/2))+(img_width/2))/y_ratio
    midx = ((float(value[2])*(img_width/2))+(img_width/2))/x_ratio
    midy = ((float(value[3])*(img_width/2))+(img_width/2))/y_ratio
    p2x = ((float(value[4])*(img_width/2))+(img_width/2))/x_ratio
    p2y = ((float(value[5])*(img_width/2))+(img_width/2))/y_ratio
    
    eosx = (p1x + midx + p2x)/3
    eosy = (p1y + midy + p2y)/3

    cx1 = ((float(value[0])*(img_width/2))+(img_width/2))
    cy1 = ((float(value[1])*(img_width/2))+(img_width/2))
    cx2 = ((float(value[2])*(img_width/2))+(img_width/2))
    cy2 = ((float(value[3])*(img_width/2))+(img_width/2))
    cx3 = ((float(value[4])*(img_width/2))+(img_width/2))
    cy3 = ((float(value[5])*(img_width/2))+(img_width/2))

    diag = math.sqrt(orig_w ** 2 + orig_h ** 2)
    eos_size_original = max([math.sqrt((p1x - midx) ** 2 + (p1y - midy) **2),  math.sqrt((p2x - midx) ** 2 + (p2y - midy) **2) ,  math.sqrt((p1x - p2x) ** 2 + (p1y - p2y) **2)])
    eos_size_cropped = max([math.sqrt((cx1-  cx2) ** 2 + (cy1 - cy2) **2),  
                            math.sqrt((cx1 - cx3) ** 2 + (cy1 - cy3) **2) ,  
                            math.sqrt((cx2 - cx3) ** 2 + (cy2 - cy3) **2)])
    eos_size_ratio = eos_size_original/eos_size_cropped
    train_zoom[key] = [eosx, eosy, orig_w, orig_h, eos_size_original, eos_size_cropped, eos_size_ratio]

f_submit = open('train_eos_150.csv', 'w')
f_submit.write('image_name, eosx, eosy, orig_w, orig_h, eos_size_original, eos_size_cropped, eos_size_ratio\n')
for key, value in train_zoom.items():
    val = ['%.6f' % p for p in value[:]]
    f_submit.write('%s,%s\n' % (key, ','.join(val)))
f_submit.close()    


#apply zoom and crop image from center   
for key, value in train_zoom.items():
        img = Image.open(train_dir+key)
        eos_size = value[4]
        zoom_factor = eos_side_desired/eos_size
        
        print(key+"-"+str(zoom_factor)+"-"+str(eos_size))
        new_w = img.size[0]*zoom_factor
        new_h = img.size[1]*zoom_factor
        #apply zoom on the image which should be centered around eos and get new location of eos after zoom
        img = img.resize((int(new_w), int(new_h)))
        eos_x = value[0]*zoom_factor
        eos_y = value[1]*zoom_factor
        
        # crop just 1000X1000 image around eos and resize it again to the trage size
        img = img.crop((eos_x-(target_width/2), eos_y-(target_height/2), eos_x+(target_width/2), eos_y+(target_height/2)))
        #img = img.resize((target_width, target_height))
        img.save(crop_dir+key)  
        
        # handel the case where there are not sufficient pixels for cropping around eos

local_color_dir = val_local_color_dir
crop_dir = val_crop_dir
# apply local color normalization
for ctype in folders:
    items = os.listdir(os.path.join(crop_dir, ctype))
    for names in items:
        if names.endswith(".jpg"):
            img = cv2.imread(os.path.join(crop_dir, ctype, names))
            img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 30), -4, 128)
            cv2.imwrite(os.path.join(local_color_dir, ctype, names), img)


local_color_dir = test_local_color_dir
crop_dir = test_crop_dir
# apply local color normalization
for ctype in ['test']:
    items = os.listdir(os.path.join(crop_dir, ctype))
    for names in items:
        if names.endswith(".jpg"):
            img = cv2.imread(os.path.join(crop_dir, ctype, names))
            img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 30), -4, 128)
            cv2.imwrite(os.path.join(local_color_dir, ctype, names), img)
            
            
        
#image convert to 3 colors only 
from sklearn.cluster import MiniBatchKMeans
import cv2
from skimage.exposure import equalize_hist


local_color_dir = '../../data/val_split_clahe_256px_crop_auto_30_3color/'
crop_dir = val_crop_dir
# apply local color normalization
for ctype in folders:
    items = os.listdir(os.path.join(crop_dir, ctype))
    for names in items:
        if names.endswith(".jpg"):
            im = cv2.imread(os.path.join(crop_dir, ctype, names))
            im = cv2.medianBlur(im, 9)
            cv2.imwrite(os.path.join(local_color_dir, ctype, names), im)
            '''
            (h, w) = im.shape[:2]
             
            # convert the image from the RGB color space to the L*a*b*
            # color space -- since we will be clustering using k-means
            # which is based on the euclidean distance, we'll use the
            # L*a*b* color space where the euclidean distance implies
            # perceptual meaning
            im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
             
            # reshape the image into a feature vector so that k-means
            # can be applied
            im = im.reshape((im.shape[0] * im.shape[1], 3))
             
            # apply k-means using the specified number of clusters and
            # then create the quantized image based on the predictions
            clt = MiniBatchKMeans(n_clusters = 100)
            labels = clt.fit_predict(im)
            quant = clt.cluster_centers_.astype("uint8")[labels]
             
            # reshape the feature vectors to images
            quant = quant.reshape((h, w, 3))
            im = im.reshape((h, w, 3))
             
            # convert from L*a*b* to RGB
            quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
            im = cv2.cvtColor(im, cv2.COLOR_LAB2BGR)
             
            # two color image
            two_color = np.hstack([im, quant])
            
            img = cv2.imread(os.path.join(crop_dir, ctype, names))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            img = equalize_hist(img)
            thresh = np.percentile(img, 5)
            
            binary = img > thresh
            ret = np.empty((200, 200, 3), dtype=bool)
            ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  np.invert(binary)
               
            quant[ret] = 0
            cv2.imwrite(os.path.join(local_color_dir, ctype, names), quant)
            #cv2.imwrite(os.path.join(local_color_dir, ctype, names), cv2.addWeighted(im, 10, quant, -10, 0))
            '''
            
            
            





