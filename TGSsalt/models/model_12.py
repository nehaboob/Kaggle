#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 08:00:54 2018

@author: neha

5 flold - same as model 11
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
    start_neurons_arg = 24
    dropout_arg = 0.05
    train_mask_threshold = 0.5
    lr_arg = 0.008
    exp_notes = "Unet with resent and se blocks on folds. Similar to 2"

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
    from keras.layers import Input,Dropout,BatchNormalization,Activation,Add,Multiply
    from keras.layers.core import Lambda, Dense
    from keras.layers.convolutional import Conv2D, Conv2DTranspose
    from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
    from keras.layers.merge import concatenate
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback, LearningRateScheduler
    from keras import backend as K
    from keras.optimizers import Adam, SGD
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
    def cov_to_class(val):    
        for i in range(0, 11):
            if val * 10 <= i :
                return i
    
    def BatchActivate(x):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    
    def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
        x = Conv2D(filters, size, strides=strides, padding=padding)(x)
        if activation==True: x = BatchActivate(x)
        return x
    
    def residual_block(blockInput, num_filters=16, batch_activate=False):
        x = BatchActivate(blockInput)
        x = convolution_block(x, num_filters, (3,3))
        x = convolution_block(x, num_filters, (3,3), activation=False)
        x = Add()([x, blockInput])
        if batch_activate: x = BatchActivate(x)
        return x
   
    def se_block(in_block, ch, ratio=16):
        x = GlobalAveragePooling2D()(in_block)
        x = Dense(ch//ratio, activation='relu')(x)
        x = Dense(ch, activation='sigmoid')(x)
        return Multiply()([in_block, x])
    # Build Model
    def build_model(input_layer, start_neurons, DropoutRatio=0.5):
        # 101 -> 50
        conv1 = Conv2D(start_neurons*1, (3,3), activation=None, padding='same')(input_layer)
        conv1 = residual_block(conv1, start_neurons*1)
        conv1 = se_block(conv1, start_neurons*1)
        conv1 = residual_block(conv1, start_neurons*1, True)
        conv1 = se_block(conv1, start_neurons*1)
        pool1 = MaxPooling2D((2,2))(conv1)
        pool1 = Dropout(DropoutRatio/2)(pool1)
        
        # 50 -> 25
        conv2 = Conv2D(start_neurons*2, (3,3), activation=None, padding='same')(pool1)
        conv2 = residual_block(conv2, start_neurons*2)
        conv2 = se_block(conv2, start_neurons*2)
        conv2 = residual_block(conv2, start_neurons*2, True)
        conv2 = se_block(conv2, start_neurons*2)
        pool2 = MaxPooling2D((2,2))(conv2)
        pool2 = Dropout(DropoutRatio)(pool2)
        
        # 25 -> 12
        conv3 = Conv2D(start_neurons*4, (3,3), activation=None, padding='same')(pool2)
        conv3 = residual_block(conv3, start_neurons*4)
        conv3 = se_block(conv3, start_neurons*4)
        conv3 = residual_block(conv3, start_neurons*4, True)
        conv3 = se_block(conv3, start_neurons*4)
        pool3 = MaxPooling2D((2,2))(conv3)
        pool3 = Dropout(DropoutRatio)(pool3)
        
        # 12 -> 6
        conv4 = Conv2D(start_neurons*8, (3,3), activation=None, padding='same')(pool3)
        conv4 = residual_block(conv4, start_neurons*8)
        conv4 = se_block(conv4, start_neurons*8)
        conv4 = residual_block(conv4, start_neurons*8, True)
        conv4 = se_block(conv4, start_neurons*8)
        pool4 = MaxPooling2D((2,2))(conv4)
        pool4 = Dropout(DropoutRatio)(pool4)
        
        # Middle
        convm = Conv2D(start_neurons*16, (3,3), activation=None, padding='same')(pool4)
        convm = residual_block(convm, start_neurons*16)
        convm = se_block(convm, start_neurons*16)
        convm = residual_block(convm, start_neurons*16, True)
        convm = se_block(convm, start_neurons*16)

        # 6 -> 12
        deconv4 = Conv2DTranspose(start_neurons*8, (3,3), strides=(2,2), padding='same')(convm)
        uconv4 = concatenate([deconv4, conv4])
        uconv4 = Dropout(DropoutRatio)(uconv4)
        
        uconv4 = Conv2D(start_neurons*8, (3,3), activation=None, padding='same')(uconv4)
        uconv4 = residual_block(uconv4, start_neurons*8)
        uconv4 = se_block(uconv4, start_neurons*8)
        uconv4 = residual_block(uconv4, start_neurons*8, True)
        uconv4 = se_block(uconv4, start_neurons*8)
        
        # 12 -> 25
        deconv3 = Conv2DTranspose(start_neurons*4, (3,3), strides=(2,2), padding='valid')(uconv4)
        uconv3 = concatenate([deconv3, conv3])
        uconv3 = Dropout(DropoutRatio)(uconv3)
        
        uconv3 = Conv2D(start_neurons*4, (3,3), activation=None, padding='same')(uconv3)
        uconv3 = residual_block(uconv3, start_neurons*4)
        uconv3 = se_block(uconv3, start_neurons*4)
        uconv3 = residual_block(uconv3, start_neurons*4, True)
        uconv3 = se_block(uconv3, start_neurons*4)
        
        # 25 -> 50
        deconv2 = Conv2DTranspose(start_neurons*2, (3,3), strides=(2,2), padding='same')(uconv3)
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = Dropout(DropoutRatio)(uconv2)
        
        uconv2 = Conv2D(start_neurons*2, (3,3), activation=None, padding='same')(uconv2)
        uconv2 = residual_block(uconv2, start_neurons*2)
        uconv2 = se_block(uconv2, start_neurons*2)
        uconv2 = residual_block(uconv2, start_neurons*2, True)
        uconv2 = se_block(uconv2, start_neurons*2)
        
        # 50 -> 101
        deconv1 = Conv2DTranspose(start_neurons*1, (3,3), strides=(2,2), padding='valid')(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = Dropout(DropoutRatio)(uconv1)
        
        uconv1 = Conv2D(start_neurons*1, (3,3), activation=None, padding='same')(uconv1)
        uconv1 = residual_block(uconv1, start_neurons*1)
        uconv1 = se_block(uconv1, start_neurons*1)
        uconv1 = residual_block(uconv1, start_neurons*1, True)
        uconv1 = se_block(uconv1, start_neurons*1)
        
        output_layer_noActi = Conv2D(1, (1,1), padding='same', activation=None)(uconv1)
        output_layer = Activation('sigmoid')(output_layer_noActi)
        
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
        batch_size = y_true_in.shape[0]
        metric = []
        for batch in range(batch_size):
            value = iou_metric(y_true_in[batch], y_pred_in[batch])
            metric.append(value)
        return np.mean(metric)
    
    def my_iou_metric(label, pred):
        metric_value = tf.py_func(iou_metric_batch, [label, pred> train_mask_threshold], tf.float64)
        return metric_value
    
    def iou_metric_batch_2(y_true_in, y_pred_in):
        ## Scoring for last model, choose threshold by validation data 
        thresholds_ori = np.linspace(0.3, 0.7, 31)
        # Reverse sigmoid function: Use code below because the  sigmoid activation was removed
        thresholds = np.log(thresholds_ori/(1-thresholds_ori))  
        ious = np.array([iou_metric_batch(y_true_in, y_pred_in > threshold) for threshold in thresholds])
        return ious.max()
        
    def my_iou_metric_2(label, pred):
        return tf.py_func(iou_metric_batch_2, [label, pred], tf.float64)
    
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
    

    # code download from: https://github.com/bermanmaxim/LovaszSoftmax
    def lovasz_grad(gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        gts = tf.reduce_sum(gt_sorted)
        intersection = gts - tf.cumsum(gt_sorted)
        union = gts + tf.cumsum(1. - gt_sorted)
        jaccard = 1. - intersection / union
        jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
        return jaccard
    
    # --------------------------- BINARY LOSSES ---------------------------
    
    def lovasz_hinge(logits, labels, per_image=True, ignore=None):
        """
        Binary Lovasz hinge loss
          logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
          labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
          per_image: compute the loss per image instead of per batch
          ignore: void class id
        """
        if per_image:
            def treat_image(log_lab):
                log, lab = log_lab
                log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
                log, lab = flatten_binary_scores(log, lab, ignore)
                return lovasz_hinge_flat(log, lab)
            losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
            loss = tf.reduce_mean(losses)
        else:
            loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
        return loss
    
    def lovasz_hinge_flat(logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
          ignore: label to ignore
        """
    
        def compute_loss():
            labelsf = tf.cast(labels, logits.dtype)
            signs = 2. * labelsf - 1.
            errors = 1. - logits * tf.stop_gradient(signs)
            errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
            gt_sorted = tf.gather(labelsf, perm)
            grad = lovasz_grad(gt_sorted)
            loss = tf.tensordot(tf.nn.elu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
            return loss
    
        # deal with the void prediction case (only void pixels)
        loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                       lambda: tf.reduce_sum(logits) * 0.,
                       compute_loss,
                       strict=True,
                       name="loss"
                       )
        return loss
    
    
    def flatten_binary_scores(scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = tf.reshape(scores, (-1,))
        labels = tf.reshape(labels, (-1,))
        if ignore is None:
            return scores, labels
        valid = tf.not_equal(labels, ignore)
        vscores = tf.boolean_mask(scores, valid, name='valid_scores')
        vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
        return vscores, vlabels
    
    def lovasz_loss(y_true, y_pred):
        y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
        #logits = K.log(y_pred / (1. - y_pred))
        logits = y_pred #Jiaxin
        loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
        return loss

    train_df = pd.read_csv("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)    
    len(train_df)
    
    train_df["images"] = [np.array(load_img("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]
    train_df["masks"] = [np.array(load_img("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
    
    # trainng for one fold
    fold = 0
    t_fold = pd.Series.from_csv("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train_fold_"+str(fold)+".csv")
    v_fold = pd.Series.from_csv("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/valid_fold_"+str(fold)+".csv")

    #Data augmentation
    x_train = np.array(train_df[train_df.index.isin(t_fold.values)].images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    y_train = np.array(train_df[train_df.index.isin(t_fold.values)].masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    x_valid = np.array(train_df[train_df.index.isin(v_fold.values)].images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    y_valid = np.array(train_df[train_df.index.isin(v_fold.values)].masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)

    #Data augmentation
    x_train2 = np.concatenate((x_train, [np.fliplr(x) for x in x_train]), axis=0)
    y_train2 = np.concatenate((y_train, [np.fliplr(x) for x in y_train]), axis=0)
    print(x_train2.shape)
    print(y_valid.shape)
    

    # model
    input_layer = Input((img_size_target, img_size_target, 1))
    output_layer = build_model(input_layer, start_neurons_arg, dropout_arg)
    
    model = Model(input_layer, output_layer)
    
    #c = Adam(lr = lr_arg)
    c = SGD(lr=lr_arg, momentum=0.9, decay=0.0001, nesterov=False)

    model.compile(loss="binary_crossentropy", optimizer=c, metrics=[my_iou_metric, 'accuracy'])
    
    model.summary()
    save_model_name = "unet_resent_12_fold"+str(fold)+".model" 
    early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode = 'max',patience=15, verbose=1)
    model_checkpoint = ModelCheckpoint(save_model_name, monitor='val_my_iou_metric', mode='max',
                                       save_best_only=True, verbose=1)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode='max', factor=0.5, patience=5,
                                  min_lr=0.000001, verbose=1)
    
    t_model1_start = time.time()
    model.fit(x_train2, y_train2,
                        validation_data=[x_valid, y_valid], 
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping, model_checkpoint, reduce_lr, LogMetrics()], 
                        verbose=2)
    t_model1_end = time.time()
    print(f"Run time = {(t_model1_end-t_model1_start)/3600} hours")

    """
    save_model_name = "/home/neha/Desktop/code/ML/Kaggle/TGSsalt/models/unet_resent_12_fold"+str(fold)+".model"
    
    # retrain using other loss
    model = load_model(save_model_name, custom_objects={'my_iou_metric':my_iou_metric, 'jaccard_distance_loss': jaccard_distance_loss})
    # remove activation layer and use lovasz loss
    input_x = model.layers[0].input
    
    output_layer = model.layers[-1].input
    model = Model(input_x, output_layer)
    c = Adam(lr=0.01)
    
    model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2, 'accuracy', my_iou_metric])
    
    model.summary()
    save_model_name = "/home/neha/Desktop/code/ML/Kaggle/TGSsalt/models/unet_resent_12_hl_fold"+str(fold)+".model" 

    early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=30, verbose=1)
    model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric_2', 
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.5, patience=5, 
                                  min_lr=0.00005, verbose=1)
    epochs = 120
    batch_size = 128
    
    t_model2_start = time.time()
    history = model.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid], 
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[ model_checkpoint,reduce_lr,early_stopping, LogMetrics()], 
                        verbose=2)
    t_model2_end = time.time()
    print(f"Run time = {(t_model2_end-t_model2_start)/3600} hours")
    """