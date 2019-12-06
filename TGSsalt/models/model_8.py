#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 20:23:31 2018

@author: neha
Notes: integrating sacred to store experment results

Resent 50 experiments
"""

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

ex = Experiment("TGsalt", interactive=True)
ex.observers.append(MongoObserver.create(url='127.0.0.1:27017', db_name='kaggle_TGSalt'))
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def my_config():
    epochs = 150
    batch_size = 32
    start_neurons_arg = 16
    dropout_arg = 0.35
    train_mask_threshold = 0.5
    lr_arg = 0.001
    exp_notes = "Resnet 50 with unetdecoder. IMG size 128. Weighted bce dice loss."

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
    from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
    from keras.layers.core import Lambda, SpatialDropout2D
    from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, Cropping2D
    from keras.layers.pooling import MaxPooling2D
    from keras.layers.merge import concatenate
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback, LearningRateScheduler
    from keras import backend as K
    from keras.optimizers import Adam
    from keras.losses import binary_crossentropy
    import time 
    import tensorflow as tf
    import h5py
    from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img #,save_img
    from keras.applications.resnet50 import ResNet50
    from imgaug import augmenters as iaa
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
    img_size_target = 128
    
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
    test_df = depths_df[~depths_df.index.isin(train_df.index)]
    
    len(train_df)
    
    train_df["images"] = [np.array(load_img("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]
    train_df["masks"] = [np.array(load_img("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    
    def cov_to_class(val):    
        for i in range(0, 11):
            if val * 10 <= i :
                return i
            
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
    
    def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
        conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
        conv = BatchNormalization(name=prefix + "_bn")(conv)
        conv = Activation('relu', name=prefix + "_activation")(conv)
        return conv

    def conv_block_simple_no_bn(prevlayer, filters, prefix, strides=(1, 1)):
        conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
        conv = Activation('relu', name=prefix + "_activation")(conv)
        return conv
    
    # Build model
    def build_model(input_layer, start_neurons, DropoutRatio = 0.5):
        resnet_base = ResNet50(include_top=False, weights='imagenet', input_tensor=input_layer)
    
        for l in resnet_base.layers:
            l.trainable = True
        
        conv1 = resnet_base.get_layer("activation_1").output                      
        conv2 = resnet_base.get_layer("activation_10").output
        conv3 = resnet_base.get_layer("activation_22").output
        conv4 = resnet_base.get_layer("activation_40").output
        conv5 = resnet_base.get_layer("activation_49").output
        
        #up6 = concatenate([Cropping2D(cropping=((0, 1), (0, 1)))(UpSampling2D()(conv5)), conv4], axis=-1)
        up6 = concatenate([UpSampling2D(interpolation='bilinear')(conv5), conv4], axis=-1)
        up6 = Dropout(DropoutRatio)(up6)
        conv6 = conv_block_simple(up6, 512, "conv6_1")
        conv6 = conv_block_simple(conv6, 512, "conv6_2")
    
        #up7 = concatenate([Cropping2D(cropping=((0, 1), (0, 1)))(UpSampling2D()(conv6)), conv3], axis=-1)
        up7 = concatenate([UpSampling2D(interpolation='bilinear')(conv6), conv3], axis=-1)
        up7 = Dropout(DropoutRatio)(up7)
        conv7 = conv_block_simple(up7, 256, "conv7_1")
        conv7 = conv_block_simple(conv7, 256, "conv7_2")
    
        up8 = concatenate([UpSampling2D(interpolation='bilinear')(conv7), conv2], axis=-1)
        up8 = Dropout(DropoutRatio)(up8)
        conv8 = conv_block_simple(up8, 128, "conv8_1")
        conv8 = conv_block_simple(conv8, 128, "conv8_2")
    
        #up9 = concatenate([Cropping2D(cropping=((0, 1), (0, 1)))(UpSampling2D()(conv8)), conv1], axis=-1)
        up9 = concatenate([UpSampling2D(interpolation='bilinear')(conv8), conv1], axis=-1)
        up9 = Dropout(DropoutRatio)(up9)
        conv9 = conv_block_simple(up9, 64, "conv9_1")
        conv9 = conv_block_simple(conv9, 64, "conv9_2")
    
        #up10 = concatenate([Cropping2D(cropping=((0, 1), (0, 1)))(UpSampling2D()(conv9)), resnet_base.input], axis=-1)
        up10 = concatenate([UpSampling2D(interpolation='bilinear')(conv9), resnet_base.input], axis=-1)
        up10 = Dropout(DropoutRatio)(up10)
        conv10 = conv_block_simple(up10, 32, "conv10_1")
        conv10 = conv_block_simple(conv10, 32, "conv10_2")
        conv10 = SpatialDropout2D(DropoutRatio)(conv10)
        output_layer_noActi = Conv2D(1, (1,1), activation=None, name='convd_1_1')(conv10)
        x = Activation('sigmoid')(output_layer_noActi)
        #x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
        model = Model(resnet_base.input, x)
        
        return model
            
    #Score the model and do a threshold optimization by the best IoU.
    
    # src: https://www.kaggle.com/aglotero/another-iou-metric
    def iou_metric(y_true_in, y_pred_in, print_table=False):
        labels = y_true_in
        y_pred = y_pred_in
    
        true_objects = 2
        pred_objects = 2
    
        # Jiaxin fin that if all zeros, then, the background is treated as object
        temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))
        #temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))
        #print(temp1)
        intersection = temp1[0]
        #print("temp2 = ",temp1[1])
        #print(intersection.shape)
        #print(intersection)
        #Compute areas (needed for finding the union between all objects)
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
    
    def jaccard_distance_loss(y_true, y_pred):
        """
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
        
        The jaccard distance loss is usefull for unbalanced datasets. This has been
        shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
        gradient.
        
        Ref: https://en.wikipedia.org/wiki/Jaccard_index
        
        @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
        @author: wassname
        """
        smooth=100
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth
    
    def bce_jaccard_loss(y_true, y_pred):
        return binary_crossentropy(y_true, y_pred) + jaccard_distance_loss(y_true, y_pred)

    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred = K.cast(y_pred, 'float32')
        y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
        intersection = y_true_f * y_pred_f
        score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
        return score
    
    def dice_loss(y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = y_true_f * y_pred_f
        score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return 1. - score

    def weighted_bce_loss(y_true, y_pred, weight):
        epsilon = 1e-7
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        logit_y_pred = K.log(y_pred / (1. - y_pred))
        loss = weight * (logit_y_pred * (1. - y_true) + 
                         K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
        return K.sum(loss) / K.sum(weight)
    
    def weighted_dice_loss(y_true, y_pred, weight):
        smooth = 1.
        w, m1, m2 = weight, y_true, y_pred
        intersection = (m1 * m2)
        score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
        loss = 1. - K.sum(score)
        return loss
    
    def weighted_bce_dice_loss(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')
        # if we want to get same size of output, kernel size must be odd
        averaged_mask = K.pool2d(
                y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
        weight = K.ones_like(averaged_mask)
        w0 = K.sum(weight)
        weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
        w1 = K.sum(weight)
        weight *= (w0 / w1)
        loss = weighted_bce_loss(y_true, y_pred, weight) + dice_loss(y_true, y_pred)
        return loss
    
    def focal_loss_fixed(y_true, y_pred):
        gamma=2.
        alpha=.25
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
        
    # learning rate schedule
    def cosine_lr(epoch):
        max_lr = 0.0001
        min_lr = 0.00000005
        lr_list = np.linspace(min_lr,max_lr,20)
        lr_list = np.concatenate([lr_list[::-1], lr_list])
        i = epoch%40
        return lr_list[i]

    def cosine_lr_2(epoch):
        max_lr = 0.0001
        min_lr = 0.00000005
        time_e = np.linspace(0, 6.28, 30)
        cos_values = (np.cos(time_e)+1)/2
        cos_values = cos_values*max_lr+min_lr
        
        return cos_values[epoch%30]

    def do_augmentation(X_train, y_train):
        seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip
        iaa.OneOf([
                iaa.Noop(),
                iaa.GaussianBlur(sigma=(0.0, 1.0)),
                iaa.Noop(),
                iaa.Affine(rotate=(-10, 10), translate_percent={"x": (-0.25, 0.25)}, mode='symmetric', cval=(0)),
                iaa.Noop(),
                iaa.PerspectiveTransform(scale=(0.04, 0.08)),
                iaa.Noop(),
                iaa.PiecewiseAffine(scale=(0.05, 0.1), mode='edge', cval=(0)),
        ]),])
        seq_det = seq.to_deterministic()
        
        
        # Move from 0-1 float to uint8 format (needed for most imgaug operators)
        X_train_aug = [(x[:,:,:] * 255.0).astype(np.uint8) for x in X_train]
        # Do augmentation
        X_train_aug = seq_det.augment_images(X_train_aug)
        # Back to 0-1 float range
        X_train_aug = [(x[:,:,:].astype(np.float64)) / 255.0 for x in X_train_aug]
    
        # Move from 0-1 float to uint8 format (needed for imgaug)
        y_train_aug = [(x[:,:,:] * 255.0).astype(np.uint8) for x in y_train]
        # Do augmentation
        y_train_aug = seq_det.augment_images(y_train_aug)
        # Make sure we only have 2 values for mask augmented
        y_train_aug = [np.where(x[:,:,:] > 0, 255, 0) for x in y_train_aug]
        # Back to 0-1 float range
        y_train_aug = [(x[:,:,:].astype(np.float64)) / 255.0 for x in y_train_aug]
        return np.array(X_train_aug), np.array(y_train_aug)
    

    # Return augmented images/masks arrays of batch size
    def generator(features, labels, batch_size):
        # create empty arrays to contain batch of features and labels
        batch_features = np.zeros((batch_size, features.shape[1], features.shape[2], features.shape[3]))
        batch_labels = np.zeros((batch_size, labels.shape[1], labels.shape[2], labels.shape[3]))
    
        while True:
            # Fill arrays of batch size with augmented data taken randomly from full passed arrays
            indexes = random.sample(range(len(features)), batch_size)
            # Perform the exactly the same augmentation for X and y
            random_augmented_images, random_augmented_labels = do_augmentation(features[indexes], labels[indexes])
            batch_features[:,:,:,:] = random_augmented_images[:,:,:,:]
            batch_labels[:,:,:,:] = random_augmented_labels[:,:,:,:]
    
            yield batch_features, batch_labels
            

    
    """
    x_t = np.array([np.transpose(x) for x in x_train]).reshape((3200, 101, 101, 1))
    y_t = np.array([np.transpose(x) for x in y_train]).reshape((3200, 101, 101, 1))

    [np.flip(x) for x in x_train],
    [np.fliplr(np.flip(x)) for x in x_train],
    x_t,
    [np.fliplr(np.flip(x)) for x in x_t],
    [np.flip(x) for x in x_t],
    [np.fliplr(x) for x in x_t]

    ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
        train_df.index.values,
        np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
        np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
        train_df.coverage.values,
        train_df.z.values,
        test_size=0.2, stratify=train_df.coverage_class, random_state= 1234)
    """
    
    t_fold = pd.Series.from_csv("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train_fold_4.csv")
    v_fold = pd.Series.from_csv("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/valid_fold_4.csv")

    #Data augmentation
    x_train = np.array(train_df[train_df.index.isin(t_fold.values)].images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    y_train = np.array(train_df[train_df.index.isin(t_fold.values)].masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    x_valid = np.array(train_df[train_df.index.isin(v_fold.values)].images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    y_valid = np.array(train_df[train_df.index.isin(v_fold.values)].masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)

    x_train2 = np.concatenate((x_train, [np.fliplr(x) for x in x_train]), axis=0)
    y_train2 = np.concatenate((y_train, [np.fliplr(x) for x in y_train]), axis=0)

    # copy the image to all 3 channels   
    x_train2 = np.array([np.stack((img.reshape((img_size_target, img_size_target)),)*3, -1) for img in x_train2])
    x_valid = np.array([np.stack((img.reshape((img_size_target, img_size_target)),)*3, -1) for img in x_valid])

    print(x_train2.shape)
    print(y_valid.shape)

    # model
    input_layer = Input((img_size_target, img_size_target, 3))
    model = build_model(input_layer, start_neurons_arg , dropout_arg)    
    adam = Adam(lr=lr_arg)  
    model.compile(loss=weighted_bce_dice_loss, optimizer=adam, metrics=[my_iou_metric, 'accuracy'])    
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode = 'max',patience=60, verbose=1)
    model_checkpoint = ModelCheckpoint("./unet_best_fold_4.model",monitor='val_my_iou_metric', 
                                       mode = 'max', save_best_only=True, verbose=1)
    #reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode = 'max',factor=0.3, patience=5, min_lr=0.000001, verbose=1)
    lrate = LearningRateScheduler(cosine_lr_2)
    st_per_epoch = x_train2.shape[0]/batch_size
    history = model.fit_generator(generator(x_train2, y_train2, batch_size),
                        validation_data=[x_valid, y_valid], 
                        epochs=epochs,
                        steps_per_epoch=st_per_epoch,
                        callbacks=[early_stopping, model_checkpoint, lrate, LogMetrics()], 
                        verbose=2)
    
    
    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_loss.legend()
    ax_score.plot(history.epoch, history.history["my_iou_metric"], label="Train score")
    ax_score.plot(history.epoch, history.history["val_my_iou_metric"], label="Validation score")
    ax_score.legend()
    
#my_main(batch_size=32, epochs=1)