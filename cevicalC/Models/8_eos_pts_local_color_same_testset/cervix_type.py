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

#for large image size
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

#np.random.seed(2017)

# param setup
img_width = 200  #299 - Xception InceptionV3
img_height = 200 #224 - VGG19 VGG16 ResNet50
img_channel = 3
nbr_train_samples = 1184
nbr_validation_samples = 297
nbr_epochs =  200
batch_size = 32
train_step = 37
val_step = 10
n_classes = 3

# traing the model for 80% 20% split
train_data_dir = '../../data/train_split_clahe_256px_crop_auto_30_lcolor'
val_data_dir = '../../data/val_split_clahe_256px_crop_auto_30_lcolor'
best_model_file = "./ex1_model_clahe_lcolor_30wh.h5"

Ctypes = ['Type_1', 'Type_2', 'Type_3']


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
        #kernel_regularizer=regularizers.l2(0.00008),
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

x = conv2d_bn(img_input, 8, 3, 3, strides=(2, 2), padding='same')
x = conv2d_bn(x, 8, 3, 3, padding='same')
x = conv2d_bn(x, 8, 3, 3)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

x = conv2d_bn(x, 8, 1, 1, padding='same')
x = conv2d_bn(x, 8, 3, 3, padding='same')
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

# mixed 0, 1, 2: 35 x 35 x 256
branch1x1 = conv2d_bn(x, 8, 1, 1)

branch5x5 = conv2d_bn(x, 8, 1, 1)
branch5x5 = conv2d_bn(branch5x5, 8, 5, 5)

branch3x3dbl = conv2d_bn(x, 8, 1, 1)
branch3x3dbl = conv2d_bn(branch3x3dbl, 8, 3, 3)
branch3x3dbl = conv2d_bn(branch3x3dbl, 8, 3, 3)

branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_pool = conv2d_bn(branch_pool, 8, 1, 1)
x = layers.concatenate(
    [branch1x1, branch5x5, branch3x3dbl, branch_pool],
    axis=channel_axis,
    name='mixed0')

# mixed 1: 35 x 35 x 256
branch1x1 = conv2d_bn(x, 8, 1, 1)

branch5x5 = conv2d_bn(x, 8, 1, 1)
branch5x5 = conv2d_bn(branch5x5, 8, 5, 5)

branch3x3dbl = conv2d_bn(x, 8, 1, 1)
branch3x3dbl = conv2d_bn(branch3x3dbl, 8, 3, 3)
branch3x3dbl = conv2d_bn(branch3x3dbl, 8, 3, 3)

branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_pool = conv2d_bn(branch_pool, 8, 1, 1)
x = layers.concatenate(
    [branch1x1, branch5x5, branch3x3dbl, branch_pool],
    axis=channel_axis,
    name='mixed1')

# mixed 2: 35 x 35 x 256
branch1x1 = conv2d_bn(x, 8, 1, 1)

branch5x5 = conv2d_bn(x, 8, 1, 1)
branch5x5 = conv2d_bn(branch5x5, 8, 5, 5)

branch3x3dbl = conv2d_bn(x, 8, 1, 1)
branch3x3dbl = conv2d_bn(branch3x3dbl, 8, 3, 3)
branch3x3dbl = conv2d_bn(branch3x3dbl, 8, 3, 3)

branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_pool = conv2d_bn(branch_pool, 8, 1, 1)
x = layers.concatenate(
    [branch1x1, branch5x5, branch3x3dbl, branch_pool],
    axis=channel_axis,
    name='mixed2')

# mixed 3: 17 x 17 x 768
branch3x3 = conv2d_bn(x, 16, 3, 3, strides=(2, 2), padding='valid')

branch3x3dbl = conv2d_bn(x, 16, 1, 1)
branch3x3dbl = conv2d_bn(branch3x3dbl, 16, 3, 3)
branch3x3dbl = conv2d_bn(
    branch3x3dbl, 16, 3, 3, strides=(2, 2), padding='valid')

branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
x = layers.concatenate(
    [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

# mixed 4: 17 x 17 x 768
branch1x1 = conv2d_bn(x, 16, 1, 1)

branch7x7 = conv2d_bn(x, 16, 1, 1)
branch7x7 = conv2d_bn(branch7x7, 16, 1, 7)
branch7x7 = conv2d_bn(branch7x7, 16, 7, 1)

branch7x7dbl = conv2d_bn(x, 16, 1, 1)
branch7x7dbl = conv2d_bn(branch7x7dbl, 16, 7, 1)
branch7x7dbl = conv2d_bn(branch7x7dbl, 16, 1, 7)
branch7x7dbl = conv2d_bn(branch7x7dbl, 16, 7, 1)
branch7x7dbl = conv2d_bn(branch7x7dbl, 16, 1, 7)

branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_pool = conv2d_bn(branch_pool, 16, 1, 1)
x = layers.concatenate(
    [branch1x1, branch7x7, branch7x7dbl, branch_pool],
    axis=channel_axis,
    name='mixed4')

# mixed 5, 6: 17 x 17 x 768
for i in range(2):
    branch1x1 = conv2d_bn(x, 32, 1, 1)

    branch7x7 = conv2d_bn(x, 32, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 32, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 16, 7, 1)

    branch7x7dbl = conv2d_bn(x, 32, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 32, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 32, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 32, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 32, 1, 7)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 16, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed' + str(5 + i))

# mixed 7: 17 x 17 x 768
branch1x1 = conv2d_bn(x, 64, 1, 1)

branch7x7 = conv2d_bn(x, 64, 1, 1)
branch7x7 = conv2d_bn(branch7x7, 64, 1, 7)
branch7x7 = conv2d_bn(branch7x7, 64, 7, 1)

branch7x7dbl = conv2d_bn(x, 64, 1, 1)
branch7x7dbl = conv2d_bn(branch7x7dbl, 64, 7, 1)
branch7x7dbl = conv2d_bn(branch7x7dbl, 64, 1, 7)
branch7x7dbl = conv2d_bn(branch7x7dbl, 64, 7, 1)
branch7x7dbl = conv2d_bn(branch7x7dbl, 64, 1, 7)

branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
x = layers.concatenate(
    [branch1x1, branch7x7, branch7x7dbl, branch_pool],
    axis=channel_axis,
    name='mixed7')

x = GlobalAveragePooling2D()(x)

#stage 6
# dense layer
#x = Flatten()(x)
#x = Dropout(0.1)(x)
x = Dense(n_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.005))(x)

inputs = img_input
# Create model.
model = Model(inputs, x, name='Cervix_pts')
 
#get the model 

#optimizer = Adam(lr=0.00008, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
optimizer = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics = ['accuracy'])
model.summary()

# autosave best Model
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose = 1, save_best_only = True)
stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=0, mode='auto')
lrreduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

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
        zoom_range=0.05,
        width_shift_range=0.08,
        height_shift_range=0.08,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input)
        
# this is the augmentation configuration we will use for validation: only rescaling
val_datagen = ImageDataGenerator(
        shear_range=0.02,
        rotation_range=360.,
        zoom_range=0.05,
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

#class_weight = {0 : 2., 1: 1., 2: 1.5}
#class_weight = {0 : 3., 1: 1., 2: 1.9}
class_weight = {0 : 2.5, 1: 1., 2: 1.5} # applied smoothing .1

history = model.fit_generator(
            train_generator,
            steps_per_epoch = train_step,            
            epochs = nbr_epochs,
            validation_data = validation_generator,
            validation_steps = val_step,
            callbacks = [best_model, lrreduce],
            class_weight = class_weight,
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

plt.plot(train_loss[300:])
plt.plot(val_loss[300:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()