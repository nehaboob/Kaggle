#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:51:42 2017

@author: neha

model is same as the base model: 
this experiment is just to check if grey scale images with hist equalization
are performing better or not
"""

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, AveragePooling2D, Conv2D
from keras.layers import Input
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from skimage.exposure import equalize_hist
from skimage import color
import numpy as np

#for large image size
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# param setup
learning_rate = 0.001
learning_rate_decay = 1e-6
img_width = 299  #299 - Xception InceptionV3
img_height = 299 #224 - VGG19 VGG16 ResNet50
img_channel = 3
nbr_train_samples = 1183
nbr_validation_samples = 298
nbr_epochs = 50
batch_size = 64
train_step = 19
val_step = 5
n_classes = 3


# traing the model for 80% 20% split
train_data_dir = '../../train_splitG'
val_data_dir = '../../val_splitG'
best_model_file = "./InceptionV3_grey_lr001_epoch20.h5"

Ctypes = ['Type_1', 'Type_2', 'Type_3']

# create the base pre-trained model
base_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=Input(shape=(img_width,img_height,img_channel)))
base_model.summary()

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(n_classes, activation='softmax')(x)

InceptionV3_model = Model(base_model.input, predictions)

# compile the model
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional Res Net 50 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
# adadelta good
InceptionV3_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

#optimizer = SGD(lr = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
#InceptionV3_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

# autosave best Model
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose = 1, save_best_only = True)
stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=0, mode='auto')

# train the model on the new data for a few epochs
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

# this is the augmentation configuration we will use for validation: only rescaling
val_datagen = ImageDataGenerator(
        rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        classes = Ctypes,
        class_mode = 'categorical')

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        classes = Ctypes,
        class_mode = 'categorical')

history = InceptionV3_model.fit_generator(
            train_generator,
            steps_per_epoch = train_step,
            epochs = nbr_epochs,
            validation_data = validation_generator,
            validation_steps = val_step,
            callbacks = [best_model],
            verbose = 1)

# fine tune section
nbr_epochs = 400

for layer in InceptionV3_model.layers[229:]:
   layer.trainable = True

InceptionV3_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',  metrics = ['accuracy'])

history = InceptionV3_model.fit_generator(
            train_generator,
            steps_per_epoch = train_step,
            epochs = nbr_epochs,
            validation_data = validation_generator,
            validation_steps = val_step,
            callbacks = [best_model],
            verbose = 1)

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
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
