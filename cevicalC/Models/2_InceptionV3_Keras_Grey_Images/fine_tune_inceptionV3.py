#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 20:42:03 2017

@author: neha
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
#for large image size
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# param setup
learning_rate = 0.001
learning_rate_decay = 1e-6
img_width = 299  #299 - Xception InceptionV3
img_height = 299 #224 - VGG19 VGG16 ResNet50
nbr_train_samples = 1183
nbr_validation_samples = 298
nbr_epochs = 40
batch_size = 32
n_classes = 3

train_data_dir = '../../train_splitG'
val_data_dir = '../../val_splitG'
best_model_file = "./InceptionV3_grey_lr001_epoch40_finetune280.h5"
Ctypes = ['Type_1', 'Type_2', 'Type_3']

# get the model and fine tune train last two layers
from keras.models import load_model
import os
root_path = '.'
weights_path = os.path.join(root_path,'InceptionV3_grey_lr001_epoch20.h5')
InceptionV3_model = load_model(weights_path)

for layer in InceptionV3_model.layers[280:]:
   layer.trainable = True

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(InceptionV3_model.layers):
   print(i, layer.name, layer.trainable)
   
#InceptionV3_model.summary()

# visualize the model -- get the last layer and train it
#from keras.utils import plot_model
#plot_model(InceptionV3_model, to_file='InceptionV3.png')

best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)

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
        classes = Ctypes,
        class_mode = 'categorical')

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        classes = Ctypes,
        class_mode = 'categorical')


# train the model
fine_tune_history = InceptionV3_model.fit_generator(
            train_generator,
            steps_per_epoch = 37,
            epochs = nbr_epochs,
            validation_data = validation_generator,
            validation_steps = 10,
            callbacks = [best_model],
            verbose = 1)

# list all data in history
print(fine_tune_history.history.keys())
## summarize history for accuracy
plt.plot(fine_tune_history.history['acc'])
plt.plot(fine_tune_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(fine_tune_history.history['loss'])
plt.plot(fine_tune_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()