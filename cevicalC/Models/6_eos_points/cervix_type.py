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
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler, Callback
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.initializers import RandomNormal
#for large image size
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np



#np.random.seed(2017)

# param setup
learning_rate = 0.001
learning_rate_decay = 1e-6
img_width = 150  #299 - Xception InceptionV3
img_height = 150 #224 - VGG19 VGG16 ResNet50
img_channel = 3
nbr_train_samples = 1184
nbr_validation_samples = 297
nbr_epochs = 100
batch_size = 32
train_step = 37
val_step = 10
n_classes = 3

# traing the model for 80% 20% split
train_data_dir = '../../train_split_crop_auto'
val_data_dir = '../../val_split_crop_auto'
best_model_file = "./ex1_model_crop_auto_500ep.h5"

Ctypes = ['Type_1', 'Type_2', 'Type_3']


model = Sequential()

#stage 1
model.add(ZeroPadding2D((1,1),input_shape=(img_width,img_height,3)))
model.add(Convolution2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.002)))
model.add(Dropout(0.25))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(32, (3, 3), strides=(2,2), activation='relu', kernel_regularizer=regularizers.l2(0.002)))

#stage 2
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.002)))
model.add(Dropout(0.25))

# replace pooling by conv
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), strides=(2,2), activation='relu', kernel_regularizer=regularizers.l2(0.002)))

#stage 3
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.002)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.002)))
model.add(Dropout(0.25))

# replace pooling by conv
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), strides=(2,2), activation='relu', kernel_regularizer=regularizers.l2(0.002)))

#stage 4
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.002)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.002)))
model.add(Dropout(0.25))

# replace pooling by conv
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), strides=(2,2), activation='relu', kernel_regularizer=regularizers.l2(0.002)))

#stage 5
# replace pooling by conv
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.002)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.002)))
model.add(Dropout(0.25))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), strides=(2,2), activation='relu', kernel_regularizer=regularizers.l2(0.002)))

#stage 6
# dense layer
model.add(Flatten())
model.add(Dropout(0.6))

model.add(Dense(n_classes, activation='softmax', kernel_regularizer=regularizers.l2(1)))


#if weights_path:
#    print('loading weights')
#model.load_weights(weights_path)

#get the model 
optimizer=SGD(lr=learning_rate, momentum = 0.9, decay = learning_rate_decay, nesterov = True)
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics = ['accuracy'])
model.summary()

# autosave best Model
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose = 1, save_best_only = True)
stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=0, mode='auto')
tensor_board = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

def learning_schedule(epoch):
    if(epoch%200 ==0):
        return 0.01
    
class LrHistory(Callback):
    def on_train_begin(self, logs={}):
        self.learning_rt = []

    def on_epoch_end(self, batch, logs={}):
        self.learning_rt.append(self.model.optimizer.lr)

lrhistory = LrHistory()

lrate = LearningRateScheduler(learning_schedule)
# train the model on the new data for a few epochs
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.03,
        rotation_range=360.,
        width_shift_range=0.03,
        height_shift_range=0.03,
        horizontal_flip=True,
        vertical_flip=True)
        
# this is the augmentation configuration we will use for validation: only rescaling
val_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.02,
        rotation_range=360.,
        width_shift_range=0.02,
        height_shift_range=0.02,
        horizontal_flip=True,
        vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        color_mode = 'rgb',
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        classes = Ctypes,
        class_mode = 'categorical')

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
            callbacks = [best_model],
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
plt.plot(history.history['val_loss'][10:])
plt.plot(history.history['loss'][10:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()