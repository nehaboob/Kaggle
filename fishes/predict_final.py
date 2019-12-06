#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:28:23 2017

@author: nehaboob
"""


from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_width = 224
img_height = 224
batch_size = 32
nbr_test_samples = 12153
nbr_augmentation = 10
n_folds = 2

FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

#root_path = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM'
root_path = '.'

test_data_dir = os.path.join(root_path, 'test_stg2/')

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
    weights_path = os.path.join(root_path,'Res50_Pop_lr000'+str(fold+1)+'.h5')
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
                classes = None,
                class_mode = None)
    
        test_image_list = test_generator.filenames
        #print('image_list: {}'.format(test_image_list[:10]))
        print('Begin to predict for testing data ...')
        if idx == 0 and fold == 0:
            predictions = InceptionV3_model.predict_generator(test_generator, nbr_test_samples)
        else:
            predictions += InceptionV3_model.predict_generator(test_generator, nbr_test_samples)

predictions /= (nbr_augmentation*n_folds)

print('Begin to write submission file ..')
f_submit = open(os.path.join(root_path, 'submit_test_stg2_rest50.csv'), 'w')
f_submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
for i, image_name in enumerate(test_image_list):
    pred = ['%.6f' % p for p in predictions[i, :]]
    if i % 100 == 0:
        print('{} / {}'.format(i, nbr_test_samples))
    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

f_submit.close()

print('Submission file successfully generated!')