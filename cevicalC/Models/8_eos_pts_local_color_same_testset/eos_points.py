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
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras import regularizers
from PIL import Image
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
img_width = 256  #299 - Xception InceptionV3
img_height = 256 #224 - VGG19 VGG16 ResNet50
img_channel = 3
nbr_train_samples = 1184
nbr_validation_samples = 297
nbr_epochs = 400
batch_size = 32
train_step = 37
val_step = 10
n_classes = 10
multi_labels = get_multilabels()
y_labels = collections.defaultdict(dict)


for key, value in multi_labels.items():
    img = Image.open('../../data/train/'+key)
    orig_w = img.size[0]
    orig_y = img.size[1]
    x_ratio = img_width/orig_w
    y_ratio = img_height/orig_y
    lable_list = [value['t1'], value['t2'], value['t3']]
    label = json.loads(value['p1'])
    lable_list.extend([label["cx"]*x_ratio, label["cy"]*y_ratio])
    label = json.loads(value['mid'])
    lable_list.extend([label["cx"]*x_ratio, label["cy"]*y_ratio])
    label = json.loads(value['p2'])
    lable_list.extend([label["cx"]*x_ratio, label["cy"]*y_ratio])
    if "circle" not in value['cshape']: 
        lable_list.extend([1])
    else:
        lable_list.extend([0])
    
    y_labels[key] = lable_list   
            
f_submit = open('y_train_50.csv', 'w')
f_submit.write('image_name,Type_1,Type_2,Type_3,p1x,p1y,midx,midy,p2x,p2y,is_circle\n')
for key, value in y_labels.items():
    val = ['%.6f' % p for p in value[:]]
    f_submit.write('%s,%s\n' % (key, ','.join(val)))
f_submit.close()    

# traing the model for 80% 20% split
train_data_dir = '../../data/train_split_lc_256px'
val_data_dir = '../../data/val_split_lc_256px'
best_model_file = "./eos_points_clahe_256X256.h5"

Ctypes = ['Type_1', 'Type_2', 'Type_3', 'p1x', 'p1y', 'midx', 'midy', 'p2x', 'p2y', 'is_circle']

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(img_width,img_height,3)))

#should I add batch normalization ?

#stage 1
model.add(Conv2D(4, (3, 3),  activation='relu', kernel_regularizer=regularizers.l2(0.003)))
model.add(Dropout(0.1))

model.add(Conv2D(4, (3, 3), strides=(2,2), activation='relu', kernel_regularizer=regularizers.l2(0.003)))

#stage 2
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(4, (3, 3),  activation='relu', kernel_regularizer=regularizers.l2(0.003)))
model.add(Dropout(0.1))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(4, (3, 3), strides=(2,2), activation='relu', kernel_regularizer=regularizers.l2(0.003)))

#stage 3
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(4, (3, 3),  activation='relu', kernel_regularizer=regularizers.l2(0.005)))
model.add(Dropout(0.1))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(4, (3, 3), strides=(2,2), activation='relu', kernel_regularizer=regularizers.l2(0.005)))

#stage 4
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(4, (3, 3),  activation='relu', kernel_regularizer=regularizers.l2(0.005)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(4, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.005)))
model.add(Dropout(0.1))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(4, (3, 3), strides=(2,2), activation='relu', kernel_regularizer=regularizers.l2(0.005)))

#stage 5
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(4, (3, 3),  activation='relu', kernel_regularizer=regularizers.l2(0.005)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(4, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.005)))
model.add(Dropout(0.1))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(4, (3, 3), strides=(2,2), activation='relu', kernel_regularizer=regularizers.l2(0.005)))
#model.add(Dropout(0.10))

model.add(Flatten())
#model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.010)))
model.add(Dropout(0.20))
model.add(Dense(n_classes, kernel_regularizer=regularizers.l2(0.025)))

#if weights_path:
#    print('loading weights')
#model.load_weights(weights_path)

#get the model 
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer = SGD(lr=0.01, momentum=0.0, decay=1e-09, nesterov=True)
model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['mae', 'mse'])
model.summary()

# autosave best Model
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose = 1, save_best_only = True)
stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=0, mode='auto')

# train the model on the new data for a few epochs
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        #samplewise_center=True,
        #samplewise_std_normalization=True,
        rescale=1./255,
        shear_range=0.01,
        width_shift_range=0.01,
        height_shift_range=0.01,
        horizontal_flip=False,
        vertical_flip=False)
        
# this is the augmentation configuration we will use for validation: only rescaling
val_datagen = ImageDataGenerator(
        #samplewise_center=True,
        #samplewise_std_normalization=True,
        rescale=1./255,
        shear_range=0.01,
        width_shift_range=0.01,
        height_shift_range=0.01,
        horizontal_flip=False,
        vertical_flip=False)

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
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'][2:])
plt.plot(history.history['val_loss'][2:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
