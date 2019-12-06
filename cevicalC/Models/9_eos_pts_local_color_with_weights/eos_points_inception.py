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
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, AveragePooling2D, Activation
from keras.layers import Input
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np
from keras import regularizers
from keras.initializers import RandomNormal
from PIL import Image
import math
import os
import json
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
img_width = 128  #299 - Xception InceptionV3
img_height = 128 #224 - VGG19 VGG16 ResNet50
img_channel = 3
nbr_train_samples = 1184
nbr_validation_samples = 297
nbr_epochs = 1000
batch_size = 32
train_step = 37
val_step = 10
n_classes = 6
multi_labels = get_multilabels()
y_labels = collections.defaultdict(dict)
y_labels_test = collections.defaultdict(dict)

## log labels - use mse loss
for key, value in multi_labels.items():
    img = Image.open('../../data/train/'+key)
    orig_w = img.size[0]
    orig_y = img.size[1]
    x_ratio = img_width/orig_w
    y_ratio = img_height/orig_y
    lable_list=[]
    #lable_list = [value['t1'], value['t2'], value['t3']]
    label = json.loads(value['p1'])
    lable_list.extend([label["cx"]*x_ratio, label["cy"]*y_ratio])
    label = json.loads(value['mid'])
    lable_list.extend([label["cx"]*x_ratio, label["cy"]*y_ratio])
    label = json.loads(value['p2'])
    lable_list.extend([label["cx"]*x_ratio, label["cy"]*y_ratio])
    '''
    if "circle" not in value['cshape']: 
        lable_list.extend([1])
    else:
        lable_list.extend([0])
    '''
    y_labels_test['p1x/'+os.path.basename(key)] = lable_list   

    #lable_list = [i/10.0 for i in lable_list]
    y_labels['p1x/'+os.path.basename(key)] = lable_list   

            
            
f_submit = open('y_train_50.csv', 'w')
f_submit.write('image_name,Type_1,Type_2,Type_3,p1x,p1y,midx,midy,p2x,p2y,is_circle\n')
for key, value in y_labels.items():
    val = ['%.6f' % p for p in value[:]]
    f_submit.write('%s,%s\n' % (key, ','.join(val)))
f_submit.close()    

# traing the model for 80% 20% split
train_data_dir = '../../data/train_split_annotations'
val_data_dir = '../../data/val_split_annotations'
best_model_file = "./eos_points_clahe_150X150.h5"

Ctypes = ['Type_1', 'Type_2', 'Type_3', 'p1x', 'p1y', 'midx', 'midy', 'p2x', 'p2y', 'is_circle']

Ctypes = ['p1x', 'p1y', 'midx', 'midy', 'p2x', 'p2y']

##model section

def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):
    """Utility function to apply conv + BN.
    Arguments:
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        kernel_regularizer=regularizers.l2(0.01),
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

#create the model
img_input = Input(shape=(img_width,img_height,3))

if K.image_data_format() == 'channels_first':
    channel_axis = 1
else:
    channel_axis = 3
    
print(channel_axis)


#stage 1
x = conv2d_bn(img_input, 4, 3, 3, padding='same')
x = Dropout(0.3)(x)
x = conv2d_bn(x, 4, 3, 3, strides=(2, 2), padding='same')

#stage 2
x = conv2d_bn(x, 4, 3, 3, padding='same')
x = Dropout(0.3)(x)
x = conv2d_bn(x, 4, 3, 3, strides=(2, 2), padding='same')

#stage 3
x = conv2d_bn(x, 4, 3, 3, padding='same')
x = Dropout(0.3)(x)
x = conv2d_bn(x, 4, 3, 3, strides=(2, 2), padding='same')

#stage 6
x = Flatten()(x)
x = Dropout(0.)(x)
x = Dense(n_classes, 
          kernel_regularizer=regularizers.l2(0.2),
          kernel_initializer=RandomNormal(mean=0.5, stddev=0.50, seed=None)
          )(x)

inputs = img_input
# Create model.
model = Model(inputs, x, name='Cervix_pts')

'''
def actual_mse(y_true, y_pred):
    y_true = K.pow(10.0, y_true)
    y_pred = K.pow(10.0, y_pred)
    return K.mean(K.square(y_pred - y_true), axis=-1)

def actual_abs(y_true, y_pred):
    y_true = K.pow(10.0, y_true)
    y_pred = K.pow(10.0, y_pred)
    return K.mean(K.abs(y_pred - y_true), axis=-1)
'''

#get the model 
#optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000375)
#optimizer = SGD(lr=0.00001, momentum=0.0, nesterov=True)
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer = Adam(lr=0.00015, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer = Adadelta(lr=0.3, rho=0.95, epsilon=1e-08, decay=0.0018)
optimizer = Adadelta(lr=0.3, rho=0.95, epsilon=1e-08, decay=0.00001)
#optimizer = Adadelta(lr=2, rho=0.95, epsilon=1e-08, decay=0.0)
model.compile(optimizer='adadelta', loss='mean_squared_error', metrics = ['mean_squared_error','mean_absolute_error'])
model.summary()

# autosave best Model
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose = 1, save_best_only = True)
stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=0, mode='auto')
lrreduce =ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=30, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
#preprocess image 
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

# train the model on the new data for a few epochs
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        #samplewise_center=True,
        #samplewise_std_normalization=True,
        #rescale=1./255,
        shear_range=0.01,
        width_shift_range=0.01,
        height_shift_range=0.01,
        horizontal_flip=False,
        vertical_flip=False,
        preprocessing_function=preprocess_input)
        
# this is the augmentation configuration we will use for validation: only rescaling
val_datagen = ImageDataGenerator(
        #samplewise_center=True,
        #samplewise_std_normalization=True,
        #rescale=1./255,
        shear_range=0.01,
        width_shift_range=0.01,
        height_shift_range=0.01,
        horizontal_flip=False,
        vertical_flip=False,
        preprocessing_function=preprocess_input)

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

# let's visualize layer names and layer indices to see get the name of the layer for saliency map
for i, layer in enumerate(model.layers):
   print(i, layer, layer.trainable)
   
# check the learned weights for the network
for layer in model.layers:
    if(layer.name == 'conv2d_1'):
        print(layer.get_weights()) # list of numpy arrays
 
# save the training history
np.savetxt("grey_200.csv", history, delimiter=",")

#11015224128
# list all data in history
print(history.history.keys())
## summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'][0:])
plt.plot(history.history['val_loss'][0:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['mean_squared_error'][0:])
plt.plot(history.history['val_mean_squared_error'][0:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['mean_absolute_error'][0:])
plt.plot(history.history['val_mean_absolute_error'][0:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

val_training = history.history['val_mean_absolute_error']
val_training.extend(history.history['val_mean_absolute_error'])

val_abs =history.history['actual_abs']


plt.plot(val_training[50:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['val'], loc='upper left')
plt.show()