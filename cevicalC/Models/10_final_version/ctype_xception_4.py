#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:43:41 2017

@author: neha
LB 0.79405
"""

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, AveragePooling2D, Conv2D, SeparableConv2D, GlobalMaxPooling2D
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
nbr_train_samples = 6607
nbr_validation_samples = 736
nbr_epochs =  100
batch_size = 32
train_step = 207
val_step = 23
n_classes = 3

# traing the model for 80% 20% split
train_data_dir = '../../data/additional/train_split_crop_auto40_all_lcolor'
val_data_dir = '../../data/additional/val_split_crop_auto40_all_lcolor'
best_model_file = "./ex1_model_clahe_lcolor_40wh_4.h5"

Ctypes = ['Type_1', 'Type_2', 'Type_3']


#create the model
img_input = Input(shape=(img_width,img_height,3))

if K.image_data_format() == 'channels_first':
    channel_axis = 1
else:
    channel_axis = 3

x = Conv2D(16, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
x = BatchNormalization(name='block1_conv1_bn')(x)
x = Activation('relu', name='block1_conv1_act')(x)
x = Conv2D(16, (3, 3), use_bias=False, name='block1_conv2')(x)
x = BatchNormalization(name='block1_conv2_bn')(x)
x = Activation('relu', name='block1_conv2_act')(x)

residual = Conv2D(16, (1, 1), strides=(2, 2),
                  padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
x = BatchNormalization(name='block2_sepconv1_bn')(x)
x = Activation('relu', name='block2_sepconv2_act')(x)
x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
x = BatchNormalization(name='block2_sepconv2_bn')(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
x = layers.add([x, residual])

residual = Conv2D(16, (1, 1), strides=(2, 2),
                  padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = Activation('relu', name='block3_sepconv1_act')(x)
x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
x = BatchNormalization(name='block3_sepconv1_bn')(x)
x = Activation('relu', name='block3_sepconv2_act')(x)
x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
x = BatchNormalization(name='block3_sepconv2_bn')(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
x = layers.add([x, residual])

residual = Conv2D(20, (1, 1), strides=(2, 2),
                  padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = Activation('relu', name='block4_sepconv1_act')(x)
x = SeparableConv2D(20, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
x = BatchNormalization(name='block4_sepconv1_bn')(x)
x = Activation('relu', name='block4_sepconv2_act')(x)
x = SeparableConv2D(20, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
x = BatchNormalization(name='block4_sepconv2_bn')(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
x = layers.add([x, residual])

for i in range(8):
    residual = x
    prefix = 'block' + str(i + 5)

    x = Activation('relu', name=prefix + '_sepconv1_act')(x)
    x = SeparableConv2D(20, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
    x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
    x = Activation('relu', name=prefix + '_sepconv2_act')(x)
    x = SeparableConv2D(20, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
    x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
    x = Activation('relu', name=prefix + '_sepconv3_act')(x)
    x = SeparableConv2D(20, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
    x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

    x = layers.add([x, residual])

residual = Conv2D(20, (1, 1), strides=(2, 2),
                  padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = Activation('relu', name='block13_sepconv1_act')(x)
x = SeparableConv2D(20, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
x = BatchNormalization(name='block13_sepconv1_bn')(x)
x = Activation('relu', name='block13_sepconv2_act')(x)
x = SeparableConv2D(20, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
x = BatchNormalization(name='block13_sepconv2_bn')(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
x = layers.add([x, residual])

x = SeparableConv2D(20, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
x = BatchNormalization(name='block14_sepconv1_bn')(x)
x = Activation('relu', name='block14_sepconv1_act')(x)

x = SeparableConv2D(20, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
x = BatchNormalization(name='block14_sepconv2_bn')(x)
x = Activation('relu', name='block14_sepconv2_act')(x)

x = GlobalMaxPooling2D()(x)
#x = SeparableConv2D(32, (7, 7), padding='same', use_bias=False, name='block14_sepconv3')(x) 
#stage 6
# dense layer
#x = Flatten()(x)
#x = Dropout(0.1)(x)
#, kernel_regularizer=regularizers.l2(0.0003)
x = Dense(n_classes, activation='softmax')(x)

inputs = img_input
# Create model.
model = Model(inputs, x, name='Cervix_pts_xception')
 
#get the model 

#optimizer = Adam(lr=0.00008, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics = ['accuracy', 'categorical_crossentropy'])
model.summary()

# autosave best Model
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose = 1, save_best_only = True)
stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=0, mode='auto')
lrreduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

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
