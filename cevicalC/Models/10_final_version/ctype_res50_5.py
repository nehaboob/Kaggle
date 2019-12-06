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
train_data_dir = '../../data/train_split_clahe_256px_crop_auto_40_lcolor'
val_data_dir = '../../data/val_split_clahe_256px_crop_auto_40_lcolor'
best_model_file = "./ex1_model_clahe_lcolor_40wh_5.h5"

Ctypes = ['Type_1', 'Type_2', 'Type_3']

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

#create the model
img_input = Input(shape=(img_width,img_height,3))

if K.image_data_format() == 'channels_last':
    bn_axis = 3
else:
    bn_axis = 1

x = ZeroPadding2D((3, 3))(img_input)
x = Conv2D(8, (7, 7), strides=(2, 2), name='conv1')(x)
x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

x = conv_block(x, 3, [8, 8, 8], stage=2, block='a', strides=(1, 1))
x = identity_block(x, 3, [8, 8, 8], stage=2, block='b')
x = identity_block(x, 3, [8, 8, 8], stage=2, block='c')

x = conv_block(x, 3, [8, 8, 8], stage=3, block='a')
x = identity_block(x, 3, [8, 8, 8], stage=3, block='b')
x = identity_block(x, 3, [8, 8, 8], stage=3, block='c')
x = identity_block(x, 3, [8, 8, 8], stage=3, block='d')

x = conv_block(x, 3, [8, 8, 8], stage=4, block='a')
x = identity_block(x, 3, [8, 8, 8], stage=4, block='b')
x = identity_block(x, 3, [8, 8, 8], stage=4, block='c')
x = identity_block(x, 3, [8, 8, 8], stage=4, block='d')
x = identity_block(x, 3, [8, 8, 8], stage=4, block='e')
x = identity_block(x, 3, [8, 8, 8], stage=4, block='f')

x = conv_block(x, 3, [12, 12, 12], stage=5, block='a')
x = identity_block(x, 3, [12, 12, 12], stage=5, block='b')
x = identity_block(x, 3, [12, 12, 12], stage=5, block='c')

x = AveragePooling2D((7, 7), name='avg_pool')(x)
#stage 6
# dense layer
x = Flatten()(x)
#x = Dropout(0.1)(x)
#, kernel_regularizer=regularizers.l2(0.0003)
x = Dense(n_classes, activation='softmax')(x)

inputs = img_input
# Create model.
model = Model(inputs, x, name='Cervix_pts_resnet_50')
 
#get the model 

#optimizer = Adam(lr=0.00008, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy', 'categorical_crossentropy'])
model.summary()

# autosave best Model
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose = 1, save_best_only = True)
stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=0, mode='auto')
lrreduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=35, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

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
class_weight = {0 : 2.8, 1: 1., 2: 1.6} # applied smoothing .1

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
plt.plot(history.history['categorical_crossentropy'][0:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


val_loss = history.history['val_loss']
train_loss = history.history['categorical_crossentropy']

val_loss.extend(history.history['val_loss'])
train_loss.extend(history.history['categorical_crossentropy'])

plt.plot(train_loss[0:])
plt.plot(val_loss[0:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()