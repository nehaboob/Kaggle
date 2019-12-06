#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 22:14:34 2017

@author: nehaboob

We will use call backs to store best model

and seperate file to predict the model

"""

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, AveragePooling2D, Conv2D
#from keras import backend as K
from keras import regularizers
from keras.layers import Input
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

#from keras.applications.vgg19 import VGG19
#from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
#from keras.applications.xception import Xception
#import os


# param setup
learning_rate = 0.0001
learning_rate_decay = 1e-6
img_width = 224  #299 - Xception InceptionV3
img_height = 224 #224 - VGG19 VGG16 ResNet50
nbr_train_samples = 3019
nbr_validation_samples = 758
nbr_epochs = 55
batch_size = 32
n_classes = 8

# traing the model for 6 folds of 80% 20% split

#train_data_dir = 'train_split'
#val_data_dir = 'val_split'
#best_model_file = "./Res50_Pop_lr0001.h5"

train_data_dir = 'train_split_1'
val_data_dir = 'val_split_1'
best_model_file = "./Res50_Pop_lr0002.h5"





FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

# create the base pre-trained model
base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(img_width,img_height,3)))

# remove the last layer
base_model.layers.pop()

# add a simple network 1
print('Adding Average Pooling Layer and Softmax Output Layer ...')
output = base_model.get_layer(index = -1).output  # Shape: (8, 8, 2048)
output = Conv2D(512, 1, 1, activation = 'relu', name = 'out_con1_512')(output)
output = AveragePooling2D((2, 2), strides=(2, 2), name='out_pool_1')(output) # for resnet
output = Flatten(name='flatten')(output)
# softmax classification
output = Dense(n_classes, activation='softmax', name='predictions')(output)

InceptionV3_model = Model(base_model.input, output)
#InceptionV3_model.summary()

# compile the model
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional Res Net 50 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])

optimizer = SGD(lr = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
InceptionV3_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

# autosave best Model
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)
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
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        classes = FishNames,
        class_mode = 'categorical')

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        classes = FishNames,
        class_mode = 'categorical')

history = InceptionV3_model.fit_generator(
            train_generator,
            samples_per_epoch = nbr_train_samples,
            nb_epoch = nbr_epochs,
            validation_data = validation_generator,
            nb_val_samples = nbr_validation_samples,
            callbacks = [best_model],
            verbose = 1)


# list all data in history
print(history.history.keys())
## summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()