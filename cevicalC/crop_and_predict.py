#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:00:07 2017

@author: neha
"""

from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
#for large image size
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.metrics import confusion_matrix
import math
import cv2

img_width = 200
img_height = 200
crop_size = 1000
batch_size = 32
nbr_test_samples = 3506 
val_steps = 110
train_step = 10
nbr_augmentation = 40
n_folds = 9
target_width = 200
target_height = 200

Ctypes = ['Type_1', 'Type_2', 'Type_3']

root_path = '.'

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def generate_stg1_submission_file(root_path, weights_file, test_dir, test_out_file):
    test_data_dir = os.path.join(root_path, test_dir)
    
    # test data generator for prediction
    test_datagen = ImageDataGenerator(
        shear_range=0.02,
        rotation_range=360.,
        zoom_range=0.05,
        width_shift_range=0.08,
        height_shift_range=0.08,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input)
    
    print('Loading model and weights from training process for all folds ....')
    InceptionV3_model = [None] * n_folds
    #print(InceptionV3_model.summary())
    for fold in range(n_folds):
        print('{}th fold for testing ...'.format(fold))
        #weights_file = 'ex1_model_clahe_lcolor_40wh_4.h5'
        weights_file = 'ex1_model_clahe_lcolor_40wh_'+str(fold+2)+'.h5'
        weights_path = os.path.join(root_path,weights_file)
        InceptionV3_model[fold] = load_model(weights_path)
        for idx in range(nbr_augmentation):
            print('{}th augmentation for testing ...'.format(idx))
            random_seed = np.random.random_integers(0, 100000)
        
            test_generator = test_datagen.flow_from_directory(
                    test_data_dir,
                    target_size=(img_width, img_height),
                    batch_size=batch_size,
                    shuffle = False, # Important !!!
                    seed = random_seed,
                    color_mode = 'rgb',
                    classes = None,
                    class_mode = None)
        
            test_image_list = test_generator.filenames
            #print('image_list: {}'.format(test_image_list[:10]))
            print('Begin to predict for testing data ...')
            if idx == 0 and fold == 0:
                predictions = InceptionV3_model[fold].predict_generator(test_generator, val_steps)
            else:
                predictions += InceptionV3_model[fold].predict_generator(test_generator, val_steps)
    
    predictions /= (nbr_augmentation*n_folds)
    print(type(predictions))
    print('Begin to write submission file ..')
    f_submit = open(os.path.join(root_path, test_out_file), 'w')
    f_submit.write('image_name,Type_1,Type_2,Type_3\n')
    for i, image_name in enumerate(test_image_list):
        print(i)
        pred = ['%.6f' % p for p in predictions[i, :]]
        if i % 100 == 0:
            print('{} / {}'.format(i, nbr_test_samples))
        f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))
    
    f_submit.close()
    
    print('Submission file successfully generated!')
    
def generate_train_data_prediction(root_path, weights_file, train_dir, train_out_file):
    train_data_dir = os.path.join(root_path, train_dir)
    
    # test data generator for prediction
    train_datagen = ImageDataGenerator(
        shear_range=0.02,
        rotation_range=360.,
        zoom_range=0.05,
        width_shift_range=0.08,
        height_shift_range=0.08,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input)
    
    print('Loading model and weights from training process for all folds ....')
    InceptionV3_model = [None] * n_folds
    for fold in range(n_folds):
        print('{}th fold for testing ...'.format(fold))
        weights_file = 'ex1_model_clahe_lcolor_40wh_4.h5'
        weights_path = os.path.join(root_path,weights_file)
        InceptionV3_model[fold] = load_model(weights_path)
        random_seed = np.random.random_integers(0, 100000)
        for idx in range(nbr_augmentation):    
            train_generator = train_datagen.flow_from_directory(
                            train_data_dir,
                            target_size=(img_width, img_height),
                            batch_size=batch_size,
                            shuffle = False, # Important !!!
                            seed = random_seed,
                            classes = None,
                            class_mode = None)
        
            test_image_list = train_generator.filenames
            print('Begin to predict for training data ...')
            if idx == 0 and fold == 0:
                predictions = InceptionV3_model[fold].predict_generator(train_generator, train_step)
            else:
                predictions += InceptionV3_model[fold].predict_generator(train_generator, train_step)
    
    predictions /= (nbr_augmentation*n_folds)
    y_true = np.array([])
    print('Begin to write submission file ..')
    f_submit = open(os.path.join(root_path, train_out_file), 'w')
    f_submit.write('image_name,Type_1,Type_2,Type_3\n')
    for i, image_name in enumerate(test_image_list):
        pred = ['%.6f' % p for p in predictions[i, :]]
        if i % 100 == 0:
            print('{} / {}'.format(i, i))
        f_submit.write('%s,%s\n' % (image_name, ','.join(pred)))
        if "Type_1" in image_name:
            y_true = np.append(y_true, [0])
        elif "Type_2" in image_name:
            y_true = np.append(y_true, [1])
        else:
            y_true = np.append(y_true, [2])
        #print(y_true)    
    f_submit.close()    
    print('Submission file successfully generated!')
    y_pred = np.argmax(predictions, axis=1)
    print(predictions.shape)
    print(len(test_image_list))
    print(len(y_pred))
    print(len(y_true))
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

def init_model(weights_path):
    root_path = '.'
    weights_path = os.path.join(root_path,weights_path)
    model = load_model(weights_path)
    print('Model loaded.')
    print(model.input)
    return model


def crop_and_preprocess(test_in_dir, test_out_dir, weights_path, test_out_file, normalized_eos_median):
    test_data_dir = os.path.join(root_path, test_in_dir)
    
    # test data generator for prediction
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.5,
        width_shift_range=0.01,
        height_shift_range=0.01,
        horizontal_flip=False,
        vertical_flip=False)
    
    random_seed = np.random.random_integers(0, 100000)
        
    test_generator = test_datagen.flow_from_directory(
                    test_data_dir,
                    target_size=(img_width, img_height),
                    batch_size=batch_size,
                    shuffle = False, # Important !!!
                    seed = random_seed,
                    classes = None,
                    class_mode = None)
        
    test_image_list = test_generator.filenames
    print('Begin to predict for test data ...')
    model = init_model(weights_path)
    predictions = model.predict_generator(test_generator, train_step)
    #print(predictions.shape)
    
    y_pred = {}
    print('Begin to write submission file ..')
    f_submit = open(os.path.join(root_path, test_out_file), 'w')
    f_submit.write('image_name,Type_1,Type_2,Type_3,p1x,p1y,midx,midy,p2x,p2y,is_circle\n')
    for i, image_name in enumerate(test_image_list):
        pred = ['%.6f' % p for p in predictions[i, :]]
        if i % 100 == 0:
            print('{} / {}'.format(i, nbr_test_samples))
        f_submit.write('%s,%s\n' % (image_name, ','.join(pred)))
        y_pred[image_name] = pred
    f_submit.close()    
    print('Submission file successfully generated!')
    
    train_zoom = {}
    eos_list = []
    
    # crop the images using predicted zoom
    for key, value in y_pred.items():
        img = Image.open(test_in_dir+key)
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
    
    for key, value in train_zoom.items():
        img = Image.open(test_in_dir+key)
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
        img.save(test_crop_dir+key)                   

    
    folders = ['test']
    # apply local color normalization
    for ctype in folders:
        items = os.listdir(os.path.join(test_crop_dir, ctype))
        for names in items:
            if names.endswith(".jpg"):
                img = cv2.imread(os.path.join(test_crop_dir, ctype, names))
                img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 20), -4, 128)
                cv2.imwrite(os.path.join(test_out_dir, ctype, names), img)
 
weights_file = 'eos_points_deepsense_2_ep100_150X150.h5'
pred_weights = 'ex1_model_clahe_lcolor_40wh_4.h5'
test_in_dir = '../../test_stg1/'
test_out_dir = '../../data/test_stg1_clahe_256px/'
test_crop_dir = '../../data/test_stg2_clahe_256px_crop_auto_40_lcolor/'
test_out_file = 'stg2_model_clahe_lcolor_40wh_ensem.csv'
train_dir = '../../data/additional/val_split_crop_auto40_all_lcolor'
train_out_file = 'ex1_model_clahe_lcolor_30wh_l77.csv'

normalized_eos_median = 0.0333637

generate_train_data_prediction(root_path, pred_weights, train_dir, train_out_file)

crop_and_preprocess(test_in_dir, test_out_dir, weights_file, test_out_file, normalized_eos_median)
generate_stg1_submission_file(root_path, pred_weights, test_crop_dir, test_out_file)

import h5py
f = h5py.File('ex1_model_clahe_lcolor_40wh_10.h5', 'r+')
del f['optimizer_weights']
f.close()
