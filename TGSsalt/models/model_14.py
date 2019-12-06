#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 14:45:31 2018

@author: neha

fine tune model 8 further
"""

from __future__ import print_function, division
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

ex = Experiment("TGsalt", interactive=True)
ex.observers.append(MongoObserver.create(url='127.0.0.1:27017', db_name='kaggle_TGSalt'))
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def my_config():
    epochs = 24
    batch_size = 16
    start_neurons_arg = 16
    dropout_arg = 0.2
    train_mask_threshold = 0.5
    lr_arg = 0.001
    exp_notes = "Resnet 50 with unetdecoder. IMG size 128. Weighted bce dice loss."

@ex.capture
def my_metrics(_run, logs):
    #print(logs.get('val_my_iou_metric'))
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
    from keras.optimizers import Adam, SGD
    from keras.losses import binary_crossentropy
    import time 
    import tensorflow as tf
    import h5py
    from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img #,save_img
    from keras.applications.resnet50 import ResNet50
    from keras.applications.vgg16 import VGG16
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
        
        up6 = concatenate([UpSampling2D(interpolation='bilinear')(conv5), conv4], axis=-1)
        conv6 = conv_block_simple(up6, 768, "conv6_1")
        conv6 = conv_block_simple(conv6, 768, "conv6_2")
    
        up7 = concatenate([UpSampling2D(interpolation='bilinear')(conv6), conv3], axis=-1)
        conv7 = conv_block_simple(up7, 512, "conv7_1")
        conv7 = conv_block_simple(conv7, 512, "conv7_2")
    
        up8 = concatenate([UpSampling2D(interpolation='bilinear')(conv7), conv2], axis=-1)
        conv8 = conv_block_simple(up8, 256, "conv8_1")
        conv8 = conv_block_simple(conv8, 256, "conv8_2")
    
        up9 = concatenate([UpSampling2D(interpolation='bilinear')(conv8), conv1], axis=-1)
        conv9 = conv_block_simple(up9, 128, "conv9_1")
        conv9 = conv_block_simple(conv9, 128, "conv9_2")
    
        #up10 = concatenate([UpSampling2D(interpolation='bilinear')(conv9), resnet_base.input], axis=-1)
        vgg = VGG16(input_shape=(img_size_target, img_size_target, 3), input_tensor=resnet_base.input, include_top=False)
        for l in vgg.layers:
            l.trainable = False
        vgg_first_conv = vgg.get_layer("block1_conv2").output
        up10 = concatenate([UpSampling2D()(conv9), resnet_base.input, vgg_first_conv], axis=-1)
        conv10 = conv_block_simple(up10, 64, "conv10_1")
        conv10 = conv_block_simple(conv10, 64, "conv10_2")
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
    
    def my_iou_metric_2(label, pred):
        metric_value = tf.py_func(iou_metric_batch, [label, pred > 0], tf.float64)
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
    
    
    class LRFinder(Callback):
        '''
        A simple callback for finding the optimal learning rate range for your model + dataset. 
        
        # Usage
            ```python
                lr_finder = LRFinder(min_lr=1e-5, 
                                     max_lr=1e-2, 
                                     steps_per_epoch=np.ceil(epoch_size/batch_size), 
                                     epochs=3)
                model.fit(X_train, Y_train, callbacks=[lr_finder])
                
                lr_finder.plot_loss()
            ```
        
        # Arguments
            min_lr: The lower bound of the learning rate range for the experiment.
            max_lr: The upper bound of the learning rate range for the experiment.
            steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
            epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient. 
            
        # References
            Blog post: jeremyjordan.me/nn-learning-rate
            Original paper: https://arxiv.org/abs/1506.01186
        '''
        
        def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
            super().__init__()
            
            self.min_lr = min_lr
            self.max_lr = max_lr
            self.total_iterations = steps_per_epoch * epochs
            self.iteration = 0
            self.history = {}
            
        def clr(self):
            '''Calculate the learning rate.'''
            x = self.iteration / self.total_iterations 
            return self.min_lr + (self.max_lr-self.min_lr) * x
            
        def on_train_begin(self, logs=None):
            '''Initialize the learning rate to the minimum value at the start of training.'''
            logs = logs or {}
            K.set_value(self.model.optimizer.lr, self.min_lr)
            
        def on_batch_end(self, epoch, logs=None):
            '''Record previous batch statistics and update the learning rate.'''
            logs = logs or {}
            self.iteration += 1
    
            self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
            self.history.setdefault('iterations', []).append(self.iteration)
    
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)
                
            K.set_value(self.model.optimizer.lr, self.clr())
     
        def plot_lr(self):
            '''Helper function to quickly inspect the learning rate schedule.'''
            plt.plot(self.history['iterations'], self.history['lr'])
            plt.yscale('log')
            plt.xlabel('Iteration')
            plt.ylabel('Learning rate')
            
        def plot_loss(self):
            '''Helper function to quickly observe the learning rate experiment results.'''
            plt.plot(self.history['lr'], self.history['loss'])
            plt.xscale('log')
            plt.xlabel('Learning rate')
            plt.ylabel('Loss')
    
    class Cosine_Scheduler(Callback):
        '''Cosine annealing learning rate scheduler with periodic restarts.
        # Usage
            ```python
                schedule = SGDRScheduler(min_lr=1e-5,
                                         max_lr=1e-2,
                                         steps_per_epoch=np.ceil(epoch_size/batch_size),
                                         lr_decay=0.9,
                                         cycle_length=5,
                                         mult_factor=1.5)
                model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
            ```
        # Arguments
            min_lr: The lower bound of the learning rate range for the experiment.
            max_lr: The upper bound of the learning rate range for the experiment.
            steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
            lr_decay: Reduce the max_lr after the completion of each cycle.
                      Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
            cycle_length: Initial number of epochs in a cycle.
            mult_factor: Scale epochs_to_restart after each full cycle completion.
        # References
            Blog post: jeremyjordan.me/nn-learning-rate
            Original paper: http://arxiv.org/abs/1608.03983
        '''
        def __init__(self,
                     min_lr,
                     max_lr,
                     steps_per_epoch,
                     lr_decay=1,
                     cycle_length=10,
                     mult_factor=2):
    
            self.min_lr = min_lr
            self.max_lr = max_lr
            self.lr_decay = lr_decay
    
            self.batch_since_restart = 0
            self.next_restart = cycle_length
    
            self.steps_per_epoch = steps_per_epoch
    
            self.cycle_length = cycle_length
            self.mult_factor = mult_factor
    
            self.history = {}
    
        def clr(self):
            '''Calculate the learning rate.'''
            fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
            return lr
    
        def on_train_begin(self, logs={}):
            '''Initialize the learning rate to the minimum value at the start of training.'''
            logs = logs or {}
            K.set_value(self.model.optimizer.lr, self.max_lr)
    
        def on_batch_end(self, batch, logs={}):
            '''Record previous batch statistics and update the learning rate.'''
            logs = logs or {}
            self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)
    
            self.batch_since_restart += 1
            K.set_value(self.model.optimizer.lr, self.clr())
    
        def on_epoch_end(self, epoch, logs={}):
            '''Check for end of current cycle, apply restarts when necessary.'''
            #print(self.clr())
            if epoch + 1 == self.next_restart:
                self.batch_since_restart = 0
                self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
                self.next_restart += self.cycle_length
                self.max_lr *= self.lr_decay
                self.min_lr *= self.lr_decay

    def do_augmentation(X_train, y_train):
        seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontally flip
            iaa.OneOf([
                    iaa.Noop(),
                    iaa.Affine(scale=(0.95, 1.05), rotate=(-8, 8), translate_percent={"x": (-0.10, 0.10)}, mode='symmetric', cval=(0)),
                    iaa.Noop(),
                    iaa.Affine(rotate=(-8, 8), mode='symmetric', cval=(0)),
                    iaa.Noop(),
                    iaa.Affine(translate_percent={"x": (-0.10, 0.10)}, mode='symmetric', cval=(0)),
                    iaa.Noop(),
                    iaa.Affine(scale=(0.95, 1.05), mode='symmetric', cval=(0)),
                    iaa.Noop(),
                    iaa.PerspectiveTransform(scale=(0.04, 0.08)),
                    iaa.Noop()
                ]),
            iaa.OneOf([
                     iaa.Noop(),
                     iaa.Add((-10, 10), per_channel=0.5),
                     iaa.Noop(),
                     iaa.Multiply((0.8, 1.2), per_channel=0.2),
                     iaa.Noop(),
                     iaa.GaussianBlur(sigma=(0.0, 1.0)),
                     iaa.Noop(),
                     iaa.ContrastNormalization((0.8, 1.20)),
                     iaa.Noop()
            ])
        ])
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
            
    fold = 7
    load_model_file = "/home/neha/Desktop/code/ML/Kaggle/TGSsalt/models/unet_resent_model14_new_lh_"+str(fold)+".model"
    save_model_file = "/home/neha/Desktop/code/ML/Kaggle/TGSsalt/models/unet_resent_model14_new_lh_cos_"+str(fold)+".model"
    t_fold = pd.Series.from_csv("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train_fold_new_"+str(fold)+".csv")
    v_fold = pd.Series.from_csv("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/valid_fold_new_"+str(fold)+".csv")

    #Data augmentation
    x_train = np.array(train_df[train_df.index.isin(t_fold.values)].images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    y_train = np.array(train_df[train_df.index.isin(t_fold.values)].masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    x_valid = np.array(train_df[train_df.index.isin(v_fold.values)].images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    y_valid = np.array(train_df[train_df.index.isin(v_fold.values)].masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)

    # copy the image to all 3 channels   
    x_train = np.array([np.stack((img.reshape((img_size_target, img_size_target)),)*3, -1) for img in x_train])
    x_valid = np.array([np.stack((img.reshape((img_size_target, img_size_target)),)*3, -1) for img in x_valid])

    print(x_train.shape)
    print(y_valid.shape)

    """
    # initial model
    """
    """
    # model
    #input_layer = Input((img_size_target, img_size_target, 3))
    #model = build_model(input_layer, start_neurons_arg , dropout_arg)    
    model = load_model(load_model_file,custom_objects={'my_iou_metric': my_iou_metric, 'weighted_bce_dice_loss':weighted_bce_dice_loss, 'lovasz_loss':lovasz_loss, 'my_iou_metric_2':my_iou_metric_2})
    sgd= SGD(lr=lr_arg)
    model.compile(loss=weighted_bce_dice_loss, optimizer=sgd, metrics=[my_iou_metric, 'accuracy'])    
    model.summary()
    
    
    # model
    st_per_epoch = x_train.shape[0]//batch_size
    early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode = 'max', patience=25, verbose=1)
    model_checkpoint = ModelCheckpoint(save_model_file,monitor='val_my_iou_metric', mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode = 'max',factor=0.3, patience=6, min_lr=0.0000001, verbose=1)

    model.fit_generator(generator(x_train, y_train, batch_size),
                        validation_data=[x_valid, y_valid], 
                        epochs=epochs,
                        steps_per_epoch=st_per_epoch,
                        callbacks=[model_checkpoint, early_stopping, reduce_lr, LogMetrics()], 
                        verbose=2)
    """
    """
    lovasz loss
    """
    """
    model = load_model(load_model_file,custom_objects={'my_iou_metric': my_iou_metric, 'weighted_bce_dice_loss':weighted_bce_dice_loss, 'lovasz_loss':lovasz_loss, 'my_iou_metric_2':my_iou_metric_2})
    #input_x = model.layers[0].input
    #output_layer = model.layers[-1].input
    #model = Model(input_x, output_layer)
    sgd= SGD(lr=lr_arg)
    model.compile(loss=lovasz_loss, optimizer=sgd, metrics=[my_iou_metric, 'accuracy', my_iou_metric_2])    
    model.summary()
    
    # model
    st_per_epoch = x_train.shape[0]//batch_size
    early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max', patience=25, verbose=1)
    model_checkpoint = ModelCheckpoint(save_model_file,monitor='val_my_iou_metric_2', mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.5, patience=6, min_lr=1e-10, verbose=1)

    model.fit_generator(generator(x_train, y_train, batch_size),
                        validation_data=[x_valid, y_valid], 
                        epochs=epochs,
                        steps_per_epoch=st_per_epoch,
                        callbacks=[model_checkpoint, early_stopping, reduce_lr, LogMetrics()], 
                        verbose=2)
    """
    #cos lr

    model = load_model(load_model_file,custom_objects={'my_iou_metric': my_iou_metric, 'weighted_bce_dice_loss':weighted_bce_dice_loss, 'lovasz_loss':lovasz_loss, 'my_iou_metric_2':my_iou_metric_2})
    sgd= SGD(lr=lr_arg)
    model.compile(loss=lovasz_loss, optimizer=sgd, metrics=[my_iou_metric, 'accuracy', my_iou_metric_2])    
    model.summary()
    # model
    st_per_epoch = x_train.shape[0]//batch_size
    #early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max', patience=25, verbose=1)
    model_checkpoint = ModelCheckpoint(save_model_file,monitor='val_my_iou_metric_2', mode = 'max', save_best_only=True, verbose=1)
    lrate = Cosine_Scheduler(min_lr=0.001, max_lr=0.01, steps_per_epoch=st_per_epoch,lr_decay=0.9, cycle_length=50, mult_factor=1)

    model.fit_generator(generator(x_train, y_train, batch_size),
                        validation_data=[x_valid, y_valid], 
                        epochs=epochs,
                        steps_per_epoch=st_per_epoch,
                        callbacks=[model_checkpoint, lrate, LogMetrics()], 
                        verbose=2)
    