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
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import csv
import collections
#for large image size
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_multilabels():
    reader = csv.reader(open('../../annotations/type_1_train_annotations.csv', 'r'))
    y_labels = collections.defaultdict(dict)
    for row in reader:
        if(row[0] != '#filename'):
            key = 'Type_1/'+row[0]
            y_labels[key].update({'t1':1})
            y_labels[key].update({'t2':0})
            y_labels[key].update({'t3':0})
            
            if(row[4] == '0'):  
                y_labels[key].update({'p1':row[5]})
            elif(row[4] == '1'):
                y_labels[key].update({'mid':row[5]})
            elif(row[4] == '2'):
                y_labels[key].update({'p2':row[5]})
            elif(row[4] == '3'):
                y_labels[key].update({'tz':row[5]})
                y_labels[key].update({'cshape':row[6]})
            
    reader = csv.reader(open('../../annotations/type_2_train_annotations.csv', 'r'))
    for row in reader:
        if(row[0] != '#filename'):
            key = 'Type_2/'+row[0]
            y_labels[key].update({'t1':0})
            y_labels[key].update({'t2':1})
            y_labels[key].update({'t3':0})
            
            if(row[4] == '0'):  
                y_labels[key].update({'p1':row[5]})
            elif(row[4] == '1'):
                y_labels[key].update({'mid':row[5]})
            elif(row[4] == '2'):
                y_labels[key].update({'p2':row[5]})
            elif(row[4] == '3'):
                y_labels[key].update({'tz':row[5]})
                y_labels[key].update({'cshape':row[6]})
            
    reader = csv.reader(open('../../annotations/type_3_train_annotations.csv', 'r'))
    for row in reader:
        if(row[0] != '#filename'):
            key = 'Type_3/'+row[0]
            y_labels[key].update({'t1':0})
            y_labels[key].update({'t2':0})
            y_labels[key].update({'t3':1})
            
            if(row[4] == '0'):  
                y_labels[key].update({'p1':row[5]})
            elif(row[4] == '1'):
                y_labels[key].update({'mid':row[5]})
            elif(row[4] == '2'):
                y_labels[key].update({'p2':row[5]})
            elif(row[4] == '3'):
                y_labels[key].update({'tz':row[5]})
                y_labels[key].update({'cshape':row[6]})
    return y_labels




# param setup
learning_rate = 0.0005
learning_rate_decay = 1e-6
img_width = 300  #299 - Xception InceptionV3
img_height = 300 #224 - VGG19 VGG16 ResNet50
img_channel = 3
nbr_train_samples = 1183
nbr_validation_samples = 298
nbr_epochs = 100
batch_size = 64
train_step = 19
val_step = 5
n_classes = 4
multi_labels = get_multilabels()
y_labels = collections.defaultdict(dict)


for key, value in multi_labels.items():
    lable_list = [value['t1'], value['t2'], value['t3']]

    if "circle" not in value['cshape']: 
        lable_list.extend([1])
    else:
        lable_list.extend([0])
    
    y_labels[key] = lable_list   

# traing the model for 80% 20% split
train_data_dir = '../../train_split_crop'
val_data_dir = '../../val_split_crop'
best_model_file = "./ex1_model_crop_multilable_100.h5"

Ctypes = ['Type_1', 'Type_2', 'Type_3', 'C_Shape']


model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(img_width,img_height,3)))
model.add(Convolution2D(32, (3, 3), activation='relu',  kernel_regularizer=regularizers.l2(0.0005)))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.15))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(32, (3, 3), activation='relu',  kernel_regularizer=regularizers.l2(0.0005)))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.15))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(32, (3, 3), activation='relu',  kernel_regularizer=regularizers.l2(0.0005)))
model.add(ZeroPadding2D((1,1)))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.15))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu',  kernel_regularizer=regularizers.l2(0.0005)))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.15))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
model.add(ZeroPadding2D((1,1)))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.15))

model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(n_classes, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0008)))

#if weights_path:
#    print('loading weights')
#model.load_weights(weights_path)

#get the model 
optimizer = SGD(lr = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
model.summary()

# autosave best Model
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose = 1, save_best_only = True)
stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=0, mode='auto')

# train the model on the new data for a few epochs
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.05,
        zoom_range=[0.6, 1.4],
        rotation_range=360.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        vertical_flip=True)
        
# this is the augmentation configuration we will use for validation: only rescaling
val_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.02,
        zoom_range=[0.9, 1.1],
        width_shift_range=0.03,
        height_shift_range=0.03,
        horizontal_flip=True,
        vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        color_mode = 'rgb',
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        classes = Ctypes,
        class_mode = 'multilabel',
        multilabel_classes = y_labels)

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        color_mode = 'rgb',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        classes = Ctypes,
        class_mode = 'multilabel',
        multilabel_classes = y_labels)

history = model.fit_generator(
            train_generator,
            steps_per_epoch = train_step,            
            epochs = nbr_epochs,
            validation_data= validation_generator,
            validation_steps = val_step,
            callbacks = [best_model],
            verbose = 1)
 
# save the training history
import numpy as np
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
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
