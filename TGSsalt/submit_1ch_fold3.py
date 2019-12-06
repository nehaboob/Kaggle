#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 17:43:18 2018

@author: neha

code for sumbission file creation
"""

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
from skimage.util import pad, crop
from keras.models import Model, load_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add, Flatten, Dense, multiply
from keras.layers.core import Lambda, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, Cropping2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback, LearningRateScheduler
from keras import backend as K
from keras.backend import tf as ktf
from keras.optimizers import Adam, SGD
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
img_size_target = 101
train_mask_threshold = 0.5

submission_file = 'submission_20.csv'

def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def conv_block_simple_no_bn(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

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

# Build model
def build_model(input_layer, start_neurons, DropoutRatio = 0.5):
    input_layer = Input((img_size_target, img_size_target, 3))

    # 101 -> 50
    conv1 = Conv2D(start_neurons*1, (3,3), activation=None, padding='same')(input_layer)
    conv1 = residual_block(conv1, start_neurons*1)
    conv1 = residual_block(conv1, start_neurons*1, True)
    pool1 = MaxPooling2D((2,2))(conv1)
    pool1 = Dropout(DropoutRatio/2)(pool1)
    
    # 50 -> 25
    conv2 = Conv2D(start_neurons*2, (3,3), activation=None, padding='same')(pool1)
    conv2 = residual_block(conv2, start_neurons*2)
    conv2 = residual_block(conv2, start_neurons*2, True)
    pool2 = MaxPooling2D((2,2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)
    
    # 25 -> 12
    conv3 = Conv2D(start_neurons*4, (3,3), activation=None, padding='same')(pool2)
    conv3 = residual_block(conv3, start_neurons*4)
    conv3 = residual_block(conv3, start_neurons*4, True)
    pool3 = MaxPooling2D((2,2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)
    
    # 12 -> 6
    conv4 = Conv2D(start_neurons*8, (3,3), activation=None, padding='same')(pool3)
    conv4 = residual_block(conv4, start_neurons*8)
    conv4 = residual_block(conv4, start_neurons*8, True)
    pool4 = MaxPooling2D((2,2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)
    
    # Middle
    convm = Conv2D(start_neurons*16, (3,3), activation=None, padding='same')(pool4)
    convm = residual_block(convm, start_neurons*16)
    convm = residual_block(convm, start_neurons*16, True)
    
    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons*8, (3,3), strides=(2,2), padding='same')(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)
    
    uconv4 = Conv2D(start_neurons*8, (3,3), activation=None, padding='same')(uconv4)
    uconv4 = residual_block(uconv4, start_neurons*8)
    uconv4 = residual_block(uconv4, start_neurons*8, True)
    
    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons*4, (3,3), strides=(2,2), padding='valid')(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)
    
    uconv3 = Conv2D(start_neurons*4, (3,3), activation=None, padding='same')(uconv3)
    uconv3 = residual_block(uconv3, start_neurons*4)
    uconv3 = residual_block(uconv3, start_neurons*4, True)
    
    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons*2, (3,3), strides=(2,2), padding='same')(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)
    
    uconv2 = Conv2D(start_neurons*2, (3,3), activation=None, padding='same')(uconv2)
    uconv2 = residual_block(uconv2, start_neurons*2)
    uconv2 = residual_block(uconv2, start_neurons*2, True)
    
    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons*1, (3,3), strides=(2,2), padding='valid')(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)
    
    uconv1 = Conv2D(start_neurons*1, (3,3), activation=None, padding='same')(uconv1)
    uconv1 = residual_block(uconv1, start_neurons*1)
    uconv1 = residual_block(uconv1, start_neurons*1, True)
    
    image_pool = AveragePooling2D(pool_size=6)(convm)
    image_pool = Conv2D(32, 1)(image_pool)
    classification = Flatten()(image_pool)
    classification = Dense(1, activation='sigmoid', name='classification')(classification)
    #output_layer = multiply([output_layer_noActi, classification])
    
    hypercolumn = concatenate([
        uconv1,
        Lambda(lambda image: ktf.image.resize_images(image, (img_size_target, img_size_target)))(uconv2),
        Lambda(lambda image: ktf.image.resize_images(image, (img_size_target, img_size_target)))(uconv3),
        Lambda(lambda image: ktf.image.resize_images(image, (img_size_target, img_size_target)))(uconv4)
    ])

    up_image_pool = UpSampling2D(size=img_size_target)(image_pool)
    
    fusion = concatenate([hypercolumn, up_image_pool])
    fusion = Conv2D(1, (3,3), padding='same', name='fusion_1')(fusion)
    fusion = Activation('sigmoid', name='fusion')(fusion)

    hypercolumn = Conv2D(1, (3,3), padding='same', name='hypercolumn_1')(hypercolumn)
    hypercolumn = Activation('sigmoid', name='hypercolumn')(hypercolumn)

    deep_model = Model(inputs=input_layer, outputs=[classification, hypercolumn, fusion])

    return deep_model
            
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
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

def my_iou_metric(label, pred):
    metric_value = tf.py_func(iou_metric_batch, [label, pred > train_mask_threshold ], tf.float64)
    return metric_value

    
def my_iou_metric_2(label, pred):
    return tf.py_func(iou_metric_batch, [label, pred > 0], tf.float64)

    
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

def upsample(img):# not used
    if img_size_ori == img_size_target:
        return img
    else:
        return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)

    
def downsample(img):# not used
    if img_size_ori == img_size_target:
        return img
    else:
        return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
   
def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i    

"""
used for converting the decoded image to rle mask
Fast compared to previous one
"""
def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
    x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
    (classi,hypcol,preds_test) = model.predict(x_test)
    preds_test = preds_test.reshape(-1, img_size_target, img_size_target)
    (classi,hyp,preds_test2_refect) = model.predict(x_test_reflect)
    preds_test2_refect = preds_test2_refect.reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
    return preds_test/2

def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

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

#def bce_jaccard_loss(y_true, y_pred):
#    return binary_crossentropy(y_true, y_pred) + jaccard_distance_loss(y_true, y_pred)
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

def bce_dice_loss_non_empty(y_true, y_pred):
    return K.max(K.max(y_true,axis=0),axis=1)*(binary_crossentropy(y_true, y_pred)-K.log(dice_coef(y_true, y_pred)))
 
def lovasz_loss_no_empty(y_true, y_pred):
    return lovasz_loss(y_true, y_pred)
    
        
############### best threshhold
# Loading of training/testing ids and depths
train_df = pd.read_csv("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]
train_df["images"] = [np.array(load_img("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]
train_df["masks"] = [np.array(load_img("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]
train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

x_test = np.array([upsample((np.array(load_img("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/test/images/{}.png".format(idx), grayscale = True))) / 255) for idx in test_df.index]).reshape(-1, img_size_target, img_size_target, 1)
x_test = np.array([np.stack((img.reshape((img_size_target, img_size_target)),)*3, -1) for img in x_test])

i=3
v_file = "valid_fold_"+str(i)+".csv"
v_fold = pd.Series.from_csv("/home/neha/Desktop/code/ML/Kaggle/TGSsalt/data/"+v_file)

model_name = "urestnet_deeps_16_lz_2_cy_3.model"
model_file = "/home/neha/Desktop/code/ML/Kaggle/TGSsalt/models/"+model_name

#Data augmentation
x_valid = np.array(train_df[train_df.index.isin(v_fold.values)].images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
y_valid = np.array(train_df[train_df.index.isin(v_fold.values)].masks.map(upsample).map(np.round).tolist()).reshape(-1, img_size_target, img_size_target, 1)
x_valid = np.array([np.stack((img.reshape((img_size_target, img_size_target)),)*3, -1) for img in x_valid])
y_valid_no_empty = np.sum(x_valid, axis=(1,2,3))
y_valid_no_empty = np.array((y_valid_no_empty>0)*1) 

input_layer = Input((img_size_target, img_size_target, 3))
model = build_model(input_layer, 16 , 0.1)
input_x = model.layers[0].input
classification = model.get_layer('classification').output
hypercolumn = model.get_layer('hypercolumn').output
fusion = model.get_layer('fusion').input      
model = Model(inputs=input_x, outputs=[classification, hypercolumn, fusion])
model.load_weights(model_file)
    
sgd = SGD(lr=0.001)
model.compile(optimizer=sgd,loss=['binary_crossentropy', bce_dice_loss_non_empty, lovasz_loss], loss_weights=[0.05, 0.1, 1.0], metrics=['accuracy', my_iou_metric])

m = model.evaluate(x_valid, [y_valid_no_empty, y_valid, y_valid])
print(pd.Series(index=model.metrics_names, data=m))


preds_valid = predict_result(model,x_valid,img_size_target)
preds_test = predict_result(model,x_test,img_size_target)

np.save('preds_valid_urestnet_deeps_16_lz_cy_'+str(i)+'.npy', preds_valid)
np.save('y_valid_urestnet_deeps_16_lz_cy_'+str(i)+'.npy', y_valid)
np.save('preds_test_urestnet_deeps_16_lz_cy_'+str(i)+'.npy', preds_test)

  
preds_valid3 = np.load('preds_valid_urestnet_deeps_16_lz_cy_3.npy')       
y_valid3 = np.load('y_valid_urestnet_deeps_16_lz_cy_3.npy')       
preds_test3 = np.load('preds_test_urestnet_deeps_16_lz_cy_3.npy')    

final_prediction = preds_valid3
final_truth = y_valid3
sumbit_prediction = preds_test3

final_prediction = np.array(final_prediction)
final_truth =np.array(final_truth)
sumbit_prediction = np.array(sumbit_prediction)

## Scoring for last model, choose threshold by validation data 
thresholds_ori = np.linspace(0.3, 0.7, 31)
thresholds_ori = np.linspace(0.2, 0.9, 40)

# Reverse sigmoid function: Use code below because the  sigmoid activation was removed
thresholds = np.log(thresholds_ori/(1-thresholds_ori)) 

ious = np.array([iou_metric_batch(final_truth, final_prediction > threshold) for threshold in thresholds])
print(ious)

# instead of using default 0 as threshold, use validation data to find the best threshold.
threshold_best_index = np.argmax(ious) 
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend()

############### submission file

t1 = time.time()
pred_dict = {idx: rle_encode(np.round(downsample(sumbit_prediction[i]) > threshold_best)) for i, idx in enumerate(test_df.index.values)}
t2 = time.time()

print("Usedtime = {t2-t1} s")

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']

sub.to_csv(submission_file)
