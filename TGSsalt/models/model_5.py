#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:27:19 2018

@author: neha

Try FCN + FC ResNet as described in below paper:
https://arxiv.org/pdf/1702.05174.pdf
"""

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

ex = Experiment("TGsalt", interactive=True)
ex.observers.append(MongoObserver.create(url='127.0.0.1:27017', db_name='kaggle_TGSalt'))
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def my_config():
    epochs = 200
    batch_size = 32
    start_neurons_arg = 16
    dropout_arg = 0.3
    train_mask_threshold = 0.5
    lr_arg = 0.001
    exp_notes = "Unet + Unet with Resnet blocks. Sort of FCN + FC ResNet as described in paper"

@ex.capture
def my_metrics(_run, logs):
    print(logs.get('val_my_iou_metric'))
    _run.log_scalar("loss", float(logs.get('loss')))
    _run.log_scalar("acc", float(logs.get('acc')))
    _run.log_scalar("val_loss", float(logs.get('val_loss')))
    _run.log_scalar("val_acc", float(logs.get('val_acc')))
    _run.log_scalar("my_iou_metric", float(logs.get('my_iou_metric')))
    _run.log_scalar("val_my_iou_metric", float(logs.get('val_my_iou_metric')))
    _run.result = float(logs.get('val_my_iou_metric'))

@ex.automain
def my_main(batch_size, epochs, start_neurons_arg, dropout_arg, train_mask_threshold, lr_arg):
    import os
    import sys
    import random
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-white')
    import seaborn as sns
    sns.set_style("white")
    from sklearn.model_selection import train_test_split
    from itertools import chain
    from skimage.io import imread, imshow, concatenate_images
    from skimage.transform import resize    
    from keras.models import Model, load_model
    from keras.layers import Input,Dropout,BatchNormalization,Activation,Add,UpSampling2D,Cropping2D,ZeroPadding2D
    from keras.layers.core import Lambda
    from keras.layers.convolutional import Conv2D, Conv2DTranspose
    from keras.layers.pooling import MaxPooling2D
    from keras.layers.merge import concatenate
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
    from keras import backend as K
    from keras.optimizers import Adam
    import time 
    import tensorflow as tf
    import h5py
    from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img #,save_img
    
    # Set some parameters
    im_width = 101
    im_height = 101
    im_chan = 1
    basicpath = '/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/'
    path_train = basicpath + 'train/'
    path_test = basicpath + 'test/'
    
    path_train_images = path_train + 'images/'
    path_train_masks = path_train + 'masks/'
    path_test_images = path_test + 'images/'
    
    img_size_ori = 101
    img_size_target = 101
    
    def upsample(img):# not used
        if img_size_ori == img_size_target:
            return img
        return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
        #res = np.zeros((img_size_target, img_size_target), dtype=img.dtype)
        #res[:img_size_ori, :img_size_ori] = img
        #return res
        
    def downsample(img):# not used
        if img_size_ori == img_size_target:
            return img
        return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
        #return img[:img_size_ori, :img_size_ori]
        
    # Loading of training/testing ids and depths
    
    train_df = pd.read_csv("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    
    len(train_df)
    
    train_df["images"] = [np.array(load_img("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]
    train_df["masks"] = [np.array(load_img("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]
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
    
    def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
        x = Conv2D(filters, size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        if activation == True:
            x = Activation('relu')(x)
        return x
    
    def residual_block(blockInput, num_filters=16):
        x = Activation('relu')(blockInput)
        x = BatchNormalization()(x)
        x = convolution_block(x, num_filters, (3,3) )
        x = convolution_block(x, num_filters, (3,3), activation=False)
        x = Add()([x, blockInput])
        return x
    
    def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)
    
        return (ch1, ch2), (cw1, cw2)
    
    # Build model
    def build_model(input_layer, start_neurons, DropoutRatio = 0.5):
        concat_axis = 3

        conv1 = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv1_1')(input_layer)
        conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(12, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(12, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
        conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
        conv4 = Conv2D(20, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(20, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    
        up_conv5 = UpSampling2D(size=(2, 2))(conv5)
        ch, cw = get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = Conv2D(20, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(20, (3, 3), activation='relu', padding='same')(conv6)
    
        up_conv6 = UpSampling2D(size=(2, 2))(conv6)
        ch, cw = get_crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis) 
        conv7 = Conv2D(16, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv7)
    
        up_conv7 = UpSampling2D(size=(2, 2))(conv7)
        ch, cw = get_crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = Conv2D(12, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(12, (3, 3), activation='relu', padding='same')(conv8)
    
        up_conv8 = UpSampling2D(size=(2, 2))(conv8)
        ch, cw = get_crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv9)
    
        ch, cw = get_crop_shape(input_layer, conv9)
        conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        conv10 = Conv2D(1, (1, 1))(conv9)        
        
        # 101 -> 50
        conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(conv10)
        conv1 = residual_block(conv1,start_neurons * 1)
        conv1 = residual_block(conv1,start_neurons * 1)
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)
        pool1 = Dropout(DropoutRatio/2)(pool1)
    
        # 50 -> 25
        conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
        conv2 = residual_block(conv2,start_neurons * 2)
        conv2 = residual_block(conv2,start_neurons * 2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)
        pool2 = Dropout(DropoutRatio)(pool2)
    
        # 25 -> 12
        conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
        conv3 = residual_block(conv3,start_neurons * 4)
        conv3 = residual_block(conv3,start_neurons * 4)
        conv3 = Activation('relu')(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)
        pool3 = Dropout(DropoutRatio)(pool3)
    
        # 12 -> 6
        conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
        conv4 = residual_block(conv4,start_neurons * 8)
        conv4 = residual_block(conv4,start_neurons * 8)
        conv4 = Activation('relu')(conv4)
        pool4 = MaxPooling2D((2, 2))(conv4)
        pool4 = Dropout(DropoutRatio)(pool4)
    
        # Middle
        convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
        convm = residual_block(convm,start_neurons * 16)
        convm = residual_block(convm,start_neurons * 16)
        convm = Activation('relu')(convm)
        
        # 6 -> 12
        deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
        uconv4 = concatenate([deconv4, conv4])
        uconv4 = Dropout(DropoutRatio)(uconv4)
        
        uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
        uconv4 = residual_block(uconv4,start_neurons * 8)
        uconv4 = residual_block(uconv4,start_neurons * 8)
        uconv4 = Activation('relu')(uconv4)
        
        # 12 -> 25
        #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
        deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
        uconv3 = concatenate([deconv3, conv3])    
        uconv3 = Dropout(DropoutRatio)(uconv3)
        
        uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
        uconv3 = residual_block(uconv3,start_neurons * 4)
        uconv3 = residual_block(uconv3,start_neurons * 4)
        uconv3 = Activation('relu')(uconv3)
    
        # 25 -> 50
        deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
        uconv2 = concatenate([deconv2, conv2])
            
        uconv2 = Dropout(DropoutRatio)(uconv2)
        uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
        uconv2 = residual_block(uconv2,start_neurons * 2)
        uconv2 = residual_block(uconv2,start_neurons * 2)
        uconv2 = Activation('relu')(uconv2)
        
        # 50 -> 101
        #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
        deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        
        uconv1 = Dropout(DropoutRatio)(uconv1)
        uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
        uconv1 = residual_block(uconv1,start_neurons * 1)
        uconv1 = residual_block(uconv1,start_neurons * 1)
        uconv1 = Activation('relu')(uconv1)
        
        uconv1 = Dropout(DropoutRatio/2)(uconv1)
        output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
        
        return output_layer
    
    #Score the model and do a threshold optimization by the best IoU.
    
    # src: https://www.kaggle.com/aglotero/another-iou-metric
    def iou_metric(y_true_in, y_pred_in, print_table=False):
        labels = y_true_in
        y_pred = y_pred_in
    
        true_objects = 2
        pred_objects = 2
    
        # Jiaxin fin that if all zeros, then, the background is treated as object
        temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))
        intersection = temp1[0]

        area_true = np.histogram(labels,bins=[0,0.5,1])[0]
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
        y_pred_in = y_pred_in 
        batch_size = y_true_in.shape[0]
        metric = []
        for batch in range(batch_size):
            value = iou_metric(y_true_in[batch], y_pred_in[batch])
            metric.append(value)
        return np.mean(metric)
    
    def my_iou_metric(label, pred):
        metric_value = tf.py_func(iou_metric_batch, [label, pred > train_mask_threshold], tf.float64)
        return metric_value
    
    class LogMetrics(Callback):
        def on_epoch_end(self, _, logs={}):
            my_metrics(logs=logs)
    
    #Data augmentation
    x_train2 = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train2 = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
    print(x_train2.shape)
    print(y_valid.shape)
    
    # model
    input_layer = Input((img_size_target, img_size_target, 1))
    output_layer = build_model(input_layer, start_neurons_arg , dropout_arg)
    
    adam = Adam(lr=lr_arg)
    # del model
    model = Model(input_layer, output_layer)
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=[my_iou_metric, 'accuracy'])
    
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode = 'max',patience=20, verbose=1)
    model_checkpoint = ModelCheckpoint("./unet_best1.model",monitor='val_my_iou_metric', 
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode = 'max',factor=0.2, patience=5, min_lr=0.00001, verbose=1)
        
    history = model.fit(x_train2, y_train2,
                        validation_data=[x_valid, y_valid], 
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping, model_checkpoint, reduce_lr, LogMetrics()], 
                        verbose=2)
    
    
    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_loss.legend()
    ax_score.plot(history.epoch, history.history["my_iou_metric"], label="Train score")
    ax_score.plot(history.epoch, history.history["val_my_iou_metric"], label="Validation score")
    ax_score.legend()
    