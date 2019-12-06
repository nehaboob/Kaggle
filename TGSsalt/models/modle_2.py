#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 10:05:15 2018

@author: neha
"""

# import the necessary packages
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import os
from unet import unet, unet_with_resnet50
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import random
import pickle
import cv2


# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric_new(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    # Jiaxin fin that if all zeros, then, the background is treated as object
    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))
#     temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))
    #print(temp1)
    intersection = temp1[0]
    #print("temp2 = ",temp1[1])
    #print(intersection.shape)
   # print(intersection)
    # Compute areas (needed for finding the union between all objects)
    #print(np.histogram(labels, bins = true_objects))
    area_true = np.histogram(labels,bins=[0,0.5,1])[0]
    #print("area_true = ",area_true)
    area_pred = np.histogram(y_pred, bins=[0,0.5,1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
  
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    intersection[intersection == 0] = 1e-9
    
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    y_pred_in = y_pred_in > 0.5 # added by sgx 20180728
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric_new(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    #print("metric = ",metric)
    return np.mean(metric)

def my_iou_metric(label, pred):
    metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float64)
    return metric_value

"""

def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
            metric.append(1)
            continue

        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = np.sum(intersection > 0) / np.sum(union > 0)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)

def iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred>0.5], tf.float64)    


def iou_test(y_true, y_pred):
    t, p = y_true, y_pred
    s = []
    iou = 0 

    if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
        sc = 0
    if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
        sc = 0
    if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
        sc = 1
    if np.count_nonzero(t) > 0 and np.count_nonzero(p) > 0:
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = np.sum(intersection > 0) / np.sum(union > 0)
        thresholds = np.arange(0.5, 1, 0.05)
        for thresh in thresholds:
            s.append(iou > thresh)
        sc = np.mean(s)
    #print("test", sc)
    return sc

def IOU_scoring(y_true, y_pred):
    iou_test(y_true, y_pred)
    IoU_thresh = []
    IoU = 0
    
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)

    if np.count_nonzero(y_true) == 0:
        if np.count_nonzero(y_pred) == 0:
            score = 1
        else:
            score = 0
    else: 
        IoU = np.sum(intersection > 0)/np.sum(union > 0 )
    
        threshholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
        IoU_thresh = [IoU > x for x in threshholds] 
        score = np.mean(IoU_thresh)
    
    #print("pred", score)               
    return score

def IOU_score(y_true, y_pred):
    #print(np.shape(y_true))
    #print(np.shape(y_pred))
    batch = y_true.shape[0]
    score_array = []
    for i in range(0, batch):
        score = IOU_scoring(y_true[i], y_pred[i]) 
        score_array.append(score)
    return np.mean(score_array)

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_IOU = []

    def on_epoch_end(self, batch, logs={}):
        #print(np.shape(self.validation_data))
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = self.model.predict(X_val)
        y_val = (y_val > 0)*1
        y_predict = (y_predict > 0.5)*1

        IOU = IOU_score(y_val, y_predict)
        self.val_IOU.append(IOU)
        print(" val IOU - ", IOU)
        return

    def get_data(self):
        return self.val_IOU

def preprocess_images(path, mask_path, mode='train'):
    # grab the image paths and randomly shuffle them
    print("[INFO] loading images...")
    imagePaths = sorted(os.listdir(path))
    random.seed(42)
    random.shuffle(imagePaths)
    
    # initialize the data and labels
    data = []
    labels = []
    
    # loop over the input images
    for img in imagePaths:
        # load the image, pre-process it, and store it in the data list
        imagePath = path+'/'+img
        image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        image = img_to_array(image)/255
        data.append(image)
        
        if(mode == 'train'):
            # extract set of class labels from the image path and update the labels list
            image_mask_path = mask_path+'/'+img
            image_mask = cv2.imread(image_mask_path, cv2.IMREAD_GRAYSCALE)
            image_mask = img_to_array(image_mask)/255
            #image_mask = image_mask.ravel()
            labels.append(image_mask)
    
    data = np.array(data, dtype="float")
    labels = np.array(labels, dtype='int')
    return data, labels
""" 

    
def train_model(trainX, testX, trainY, testY, EPOCHS, INIT_LR, BATCH_SIZE, IMAGE_DIMS):
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    	horizontal_flip=True, fill_mode="nearest")
    
    # initialize the model using a sigmoid activation as the final layer
    # in the network so we can perform multi-label classification
    print("[INFO] compiling model...")
    
    """
    model = SmallerVGGNet.build(
    	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
    	depth=IMAGE_DIMS[2], classes=n_classes,
    	finalAct="sigmoid")
    """
    
    early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode = 'max',patience=20, verbose=1)
    
    model_checkpoint = ModelCheckpoint('./unet_best1.model',monitor='val_my_iou_metric', 
                                   mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode = 'max',factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    
    """
    early_stopping = EarlyStopping(monitor='val_iou_metric', mode = 'max',patience=20, verbose=1)
    
    model_checkpoint = ModelCheckpoint('./unet_best1.model',monitor='val_iou_metric', 
                                   mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_iou_metric', mode = 'max',factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    """
    
    model = unet_with_resnet50(IMAGE_DIMS)
    #metrics = Metrics()

    print(model.summary())
    
    # initialize the optimizer
    opt = Adam()    
    # compile the model using binary cross-entropy rather than
    # categorical cross-entropy -- this may seem counterintuitive for
    # multi-label classification, but keep in mind that the goal here
    # is to treat each output label as an independent Bernoulli
    # distribution
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy", my_iou_metric])

    # train the network
    print("[INFO] training network...")
    """
    H = model.fit_generator(
    	aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    	validation_data=(testX, testY),
    	steps_per_epoch=len(trainX) // BATCH_SIZE,
    	epochs=EPOCHS, verbose=1, callbacks=[metrics])"""
    
    H = model.fit(trainX, trainY,
                    validation_data=[testX, testY], 
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    #callbacks=[metrics], 
                    callbacks=[ model_checkpoint,reduce_lr,early_stopping], 
                    verbose=2)
    
    return H       
  
def plot_training_graphs(H):
# plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    plt.show()   
    
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left") 
    plt.show()
    
    plt.plot(np.arange(0, N), H.history["my_iou_metric"], label="val_IOU")
    plt.plot(np.arange(0, N), H.history["val_my_iou_metric"], label="val_IOU")
    plt.title("Val IOU")
    plt.xlabel("Epoch #")
    plt.ylabel("IOU")
    plt.legend(loc="upper left")
    plt.show()    

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 200
INIT_LR = 1e-5
BATCH_SIZE = 32
IMAGE_DIMS = (101, 101, 1)
train_data_path='/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train/images'
train_mask_path='/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train/masks'

"""
train_data, train_lables = preprocess_images(train_data_path, train_mask_path, mode='train')
n_classes = train_lables.shape[1]


# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(train_data, train_lables, test_size=0.2, random_state=42)
H = train_model(trainX, testX, trainY, testY, EPOCHS, INIT_LR, BATCH_SIZE, IMAGE_DIMS)
plot_training_graphs(H)
"""
# Loading of training/testing ids and depths
def upsample(img):
    return img

img_size_ori = 101
img_size_target = 101
train_df = pd.read_csv("../data/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../data/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

len(train_df)

train_df["images"] = [np.array(load_img("../data/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]
train_df["masks"] = [np.array(load_img("../data/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]
train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)

def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i
        
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    train_df.coverage.values,
    train_df.z.values,
    test_size=0.2, stratify=train_df.coverage_class, random_state= 1234)

H = train_model(x_train, x_valid, y_train, y_valid, EPOCHS, INIT_LR, BATCH_SIZE, IMAGE_DIMS)
plot_training_graphs(H)
