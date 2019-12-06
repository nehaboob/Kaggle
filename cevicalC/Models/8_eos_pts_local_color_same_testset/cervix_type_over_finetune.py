#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:43:41 2017

@author: neha
changed augumentataion for next learning
"""

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, AveragePooling2D, Conv2D
from keras.layers import Input
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler, Callback, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam, Adadelta
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.initializers import RandomNormal
from keras import backend as K
from keras import layers
from keras.models import load_model

#for large image size
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np


img_width = 200  #299 - Xception InceptionV3
img_height = 200 #224 - VGG19 VGG16 ResNet50
img_channel = 3
nbr_train_samples = 1184
nbr_validation_samples = 297
nbr_epochs =  100
batch_size = 32
train_step = 37
val_step = 10
n_classes = 3

# traing the model for 80% 20% split
train_data_dir = '../../data/train_split_clahe_256px_crop_auto_30_lcolor'
val_data_dir = '../../data/val_split_clahe_256px_crop_auto_30_lcolor'
best_model_file = "./ex1_model_clahe_lcolor_30wh_finetune.h5"

Ctypes = ['Type_1', 'Type_2', 'Type_3']

weights_path = 'ex1_model_clahe_lcolor_30wh_over.h5'
model = load_model(weights_path)

#get the model 
#optimizer=SGD(lr=learning_rate, momentum = 0.9, decay = learning_rate_decay, nesterov = True)
#optimizer = Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.000008)
optimizer = Adam(lr=0.00008, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])
model.summary()

# autosave best Model
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose = 1, save_best_only = True)
stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=0, mode='auto')
lrreduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

#preprocess image 
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

# train the model on the new data for a few epochs
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        shear_range=0.02,
        rotation_range=360.,
        zoom_range=0.5,
        width_shift_range=0.08,
        height_shift_range=0.08,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input)
        
# this is the augmentation configuration we will use for validation: only rescaling
val_datagen = ImageDataGenerator(
        shear_range=0.02,
        rotation_range=360.,
        zoom_range=0.5,
        width_shift_range=0.08,
        height_shift_range=0.08,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        color_mode = 'rgb',
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        classes = Ctypes,
        class_mode = 'categorical'
        )

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        color_mode = 'rgb',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        classes = Ctypes,
        class_mode = 'categorical')

history = model.fit_generator(
            train_generator,
            steps_per_epoch = train_step,            
            epochs = nbr_epochs,
            validation_data = validation_generator,
            validation_steps = val_step,
            callbacks = [best_model, lrreduce],
            verbose = 1)
 
# save the training history
np.savetxt("grey_200.csv", history, delimiter=",")

# list all data in history
print(history.history.keys())
## summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'][0:])
plt.plot(history.history['val_loss'][0:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


val_loss = history.history['val_loss']
train_loss = history.history['loss']

val_loss.extend(history.history['val_loss'])
train_loss.extend(history.history['loss'])

plt.plot(train_loss[0:])
plt.plot(val_loss[0:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
