#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:00:07 2017

@author: neha
"""

from keras.models import load_model
from keras.losses import categorical_crossentropy
import tensorflow as tf
from keras import backend as K
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
#for large image size
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.metrics import confusion_matrix

#250:781:450

img_width = 150
img_height = 150
batch_size = 32
nbr_test_samples = 512
val_steps = 16
train_step = 47
nbr_augmentation = 10
n_folds = 1

Ctypes = ['Type_1', 'Type_2', 'Type_3', 'p1x', 'p1y', 'midx', 'midy', 'p2x', 'p2y', 'is_circle']

root_path = '.'

def generate_stg1_submission_file(root_path, weights_file, test_dir, test_out_file):
    test_data_dir = os.path.join(root_path, test_dir)
    
    # test data generator for prediction
    test_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)
    
    print('Loading model and weights from training process for all folds ....')
    
    for fold in range(n_folds):
        print('{}th fold for testing ...'.format(fold))
        weights_path = os.path.join(root_path,weights_file)
        InceptionV3_model = load_model(weights_path)
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
                predictions = InceptionV3_model.predict_generator(test_generator, val_steps)
            else:
                predictions += InceptionV3_model.predict_generator(test_generator, val_steps)
    
    predictions /= (nbr_augmentation*n_folds)
    
    print('Begin to write submission file ..')
    f_submit = open(os.path.join(root_path, test_out_file), 'w')
    f_submit.write('image_name,Type_1,Type_2,Type_3\n')
    for i, image_name in enumerate(test_image_list):
        print(predictions[i, :])
        pred = ['%.6f' % p for p in predictions[i, :-1]]
        if i % 100 == 0:
            print('{} / {}'.format(i, nbr_test_samples))
        f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))
    
    f_submit.close()
    
    print('Submission file successfully generated!')
    
def generate_train_data_prediction(root_path, weights_file, train_dir, train_out_file):
    train_data_dir = os.path.join(root_path, train_dir)
    
    # test data generator for prediction
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.01,
        width_shift_range=0.01,
        height_shift_range=0.01,
        horizontal_flip=False,
        vertical_flip=False)
    
    print('Loading model and weights from training process for all folds ....')
    
    weights_path = os.path.join(root_path,weights_file)
    model = load_model(weights_path)
    random_seed = np.random.random_integers(0, 100000)
        
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
    predictions = model.predict_generator(train_generator, train_step)
    #print(predictions.shape)
    y_true = np.array([])
    y_true_arr = np.empty([1481, 3])
    print('Begin to write submission file ..')
    f_submit = open(os.path.join(root_path, train_out_file), 'w')
    f_submit.write('image_name,Type_1,Type_2,Type_3,p1x,p1y,midx,midy,p2x,p2y,is_circle\n')
    for i, image_name in enumerate(test_image_list):
        pred = ['%.6f' % p for p in predictions[i, :]]
        if i % 100 == 0:
            print('{} / {}'.format(i, nbr_test_samples))
        f_submit.write('%s,%s\n' % (image_name, ','.join(pred)))
        if "Type_1" in image_name:
            y_true = np.append(y_true, [0])
            y_true_arr[i] = [1, 0, 0]
        elif "Type_2" in image_name:
            y_true = np.append(y_true, [1])
            y_true_arr[i] = [0, 1, 0]
        else:
            y_true = np.append(y_true, [2])
            y_true_arr[i] = [0, 0, 1]
    f_submit.close()    
    print('Submission file successfully generated!')
    #y_pred = np.argmax(predictions[:,:-1], axis=1)
    #np.savetxt("y_pred.csv", predictions[:,:-1], delimiter=",")
    #np.savetxt("y_true.csv", y_true_arr, delimiter=",")
    #cm = confusion_matrix(y_true, y_pred)
    #T = K.variable(y_true_arr, name="T")
    #P = K.variable(predictions[:,:-1], name="P")

    #loss = categorical_crossentropy(T, P)
    #print(cm)
    #print(predictions[:,:-1])
    #print(y_true_arr)
    #print(K.eval(loss))

    #print(K.eval(K.mean(loss)))
 
weights_file = 'eos_points_deepsense_2_ep100_150X150.h5'
train_dir = '../../train/'
test_dir = '../../test_stg1_crop/'
train_out_file = 'train_pred_150X150.csv'
test_out_file = 'name.csv'
#generate_stg1_submission_file()
generate_train_data_prediction(root_path, weights_file, train_dir, train_out_file)
