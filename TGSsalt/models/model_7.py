#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 13:01:27 2018

@author: neha

Notes: deeplabv3
https://github.com/bonlime/keras-deeplab-v3-plus/blob/master/model.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

ex = Experiment("TGsalt", interactive=True)
ex.observers.append(MongoObserver.create(url='127.0.0.1:27017', db_name='kaggle_TGSalt'))
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def my_config():
    epochs = 100
    batch_size = 32
    start_neurons_arg = 18
    dropout_arg = 0.3
    train_mask_threshold = 0.5
    lr_arg = 0.001
    exp_notes = "Deeplabv3"

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
    from keras.layers import Input,Dropout,BatchNormalization,Activation,Add,Concatenate
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
    from keras.models import Model
    from keras import layers
    from keras.layers import Input
    from keras.layers import Activation
    from keras.layers import Concatenate
    from keras.layers import Add
    from keras.layers import Dropout
    from keras.layers import BatchNormalization
    from keras.layers import Conv2D
    from keras.layers import DepthwiseConv2D
    from keras.layers import ZeroPadding2D
    from keras.layers import AveragePooling2D
    from keras.engine import Layer
    from keras.engine import InputSpec
    from keras.engine.topology import get_source_inputs
    from keras import backend as K
    from keras.applications import imagenet_utils
    from keras.utils import conv_utils
    from keras.utils.data_utils import get_file
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
    
    
    ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
        train_df.index.values,
        np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
        np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
        train_df.coverage.values,
        train_df.z.values,
        test_size=0.2, stratify=train_df.coverage_class, random_state= 1234)



    class BilinearUpsampling(Layer):
        """Just a simple bilinear upsampling layer. Works only with TF.
           Args:
               upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
               output_size: used instead of upsampling arg if passed!
        """
    
        def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):
    
            super(BilinearUpsampling, self).__init__(**kwargs)
    
            self.data_format = K.normalize_data_format(data_format)
            self.input_spec = InputSpec(ndim=4)
            if output_size:
                self.output_size = conv_utils.normalize_tuple(
                    output_size, 2, 'output_size')
                self.upsampling = None
            else:
                self.output_size = None
                self.upsampling = conv_utils.normalize_tuple(
                    upsampling, 2, 'upsampling')
    
        def compute_output_shape(self, input_shape):
            if self.upsampling:
                height = self.upsampling[0] * \
                    input_shape[1] if input_shape[1] is not None else None
                width = self.upsampling[1] * \
                    input_shape[2] if input_shape[2] is not None else None
            else:
                height = self.output_size[0]
                width = self.output_size[1]
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])
    
        def call(self, inputs):
            if self.upsampling:
                return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                           inputs.shape[2] * self.upsampling[1]),
                                                  align_corners=True)
            else:
                return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                           self.output_size[1]),
                                                  align_corners=True)
    
        def get_config(self):
            config = {'upsampling': self.upsampling,
                      'output_size': self.output_size,
                      'data_format': self.data_format}
            base_config = super(BilinearUpsampling, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))


    def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
        """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
            Implements right "same" padding for even kernel sizes
            Args:
                x: input tensor
                filters: num of filters in pointwise convolution
                prefix: prefix before name
                stride: stride at depthwise conv
                kernel_size: kernel size for depthwise convolution
                rate: atrous rate for depthwise convolution
                depth_activation: flag to use activation between depthwise & poinwise convs
                epsilon: epsilon to use in BN layer
        """
    
        if stride == 1:
            depth_padding = 'same'
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = ZeroPadding2D((pad_beg, pad_end))(x)
            depth_padding = 'valid'
    
        if not depth_activation:
            x = Activation('relu')(x)
        x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                            padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
        x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
        if depth_activation:
            x = Activation('relu')(x)
        x = Conv2D(filters, (1, 1), padding='same',
                   use_bias=False, name=prefix + '_pointwise')(x)
        x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
        if depth_activation:
            x = Activation('relu')(x)
    
        return x


    def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
        """Implements right 'same' padding for even kernel sizes
            Without this there is a 1 pixel drift when stride = 2
            Args:
                x: input tensor
                filters: num of filters in pointwise convolution
                prefix: prefix before name
                stride: stride at depthwise conv
                kernel_size: kernel size for depthwise convolution
                rate: atrous rate for depthwise convolution
        """
        if stride == 1:
            return Conv2D(filters,
                          (kernel_size, kernel_size),
                          strides=(stride, stride),
                          padding='same', use_bias=False,
                          dilation_rate=(rate, rate),
                          name=prefix)(x)
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = ZeroPadding2D((pad_beg, pad_end))(x)
            return Conv2D(filters,
                          (kernel_size, kernel_size),
                          strides=(stride, stride),
                          padding='valid', use_bias=False,
                          dilation_rate=(rate, rate),
                          name=prefix)(x)


    def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                        rate=1, depth_activation=False, return_skip=False):
        """ Basic building block of modified Xception network
            Args:
                inputs: input tensor
                depth_list: number of filters in each SepConv layer. len(depth_list) == 3
                prefix: prefix before name
                skip_connection_type: one of {'conv','sum','none'}
                stride: stride at last depthwise conv
                rate: atrous rate for depthwise convolution
                depth_activation: flag to use activation between depthwise & pointwise convs
                return_skip: flag to return additional tensor after 2 SepConvs for decoder
                """
        residual = inputs
        for i in range(3):
            residual = SepConv_BN(residual,
                                  depth_list[i],
                                  prefix + '_separable_conv{}'.format(i + 1),
                                  stride=stride if i == 2 else 1,
                                  rate=rate,
                                  depth_activation=depth_activation)
            if i == 1:
                skip = residual
        if skip_connection_type == 'conv':
            shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                    kernel_size=1,
                                    stride=stride)
            shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
            outputs = layers.add([residual, shortcut])
        elif skip_connection_type == 'sum':
            outputs = layers.add([residual, inputs])
        elif skip_connection_type == 'none':
            outputs = residual
        if return_skip:
            return outputs, skip
        else:
            return outputs


    def relu6(x):
        return K.relu(x, max_value=6)
    
    
    def _make_divisible(v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


    def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
        in_channels = inputs._keras_shape[-1]
        pointwise_conv_filters = int(filters * alpha)
        pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
        x = inputs
        prefix = 'expanded_conv_{}_'.format(block_id)
        if block_id:
            # Expand
    
            x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                       use_bias=False, activation=None,
                       name=prefix + 'expand')(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                   name=prefix + 'expand_BN')(x)
            x = Activation(relu6, name=prefix + 'expand_relu')(x)
        else:
            prefix = 'expanded_conv_'
        # Depthwise
        x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                            use_bias=False, padding='same', dilation_rate=(rate, rate),
                            name=prefix + 'depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'depthwise_BN')(x)
    
        x = Activation(relu6, name=prefix + 'depthwise_relu')(x)
    
        # Project
        x = Conv2D(pointwise_filters,
                   kernel_size=1, padding='same', use_bias=False, activation=None,
                   name=prefix + 'project')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'project_BN')(x)
    
        if skip_connection:
            return Add(name=prefix + 'add')([inputs, x])
    
        # if in_channels == pointwise_filters and stride == 1:
        #    return Add(name='res_connect_' + str(block_id))([inputs, x])
    
        return x


    def Deeplabv3(weights='pascal_voc', input_tensor=None, input_shape=(512, 512, 3), classes=21, backbone='mobilenetv2', OS=8, alpha=1.):
        """ Instantiates the Deeplabv3+ architecture
        Optionally loads weights pre-trained
        on PASCAL VOC. This model is available for TensorFlow only,
        and can only be used with inputs following the TensorFlow
        data format `(width, height, channels)`.
        # Arguments
            weights: one of 'pascal_voc' (pre-trained on pascal voc)
                or None (random initialization)
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: shape of input image. format HxWxC
                PASCAL VOC model was trained on (512,512,3) images
            classes: number of desired classes. If classes != 21,
                last layer is initialized randomly
            backbone: backbone to use. one of {'xception','mobilenetv2'}
            OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
                Used only for xception backbone.
            alpha: controls the width of the MobileNetV2 network. This is known as the
                width multiplier in the MobileNetV2 paper.
                    - If `alpha` < 1.0, proportionally decreases the number
                        of filters in each layer.
                    - If `alpha` > 1.0, proportionally increases the number
                        of filters in each layer.
                    - If `alpha` = 1, default number of filters from the paper
                        are used at each layer.
                Used only for mobilenetv2 backbone
        # Returns
            A Keras model instance.
        # Raises
            RuntimeError: If attempting to run this model with a
                backend that does not support separable convolutions.
            ValueError: in case of invalid argument for `weights` or `backbone`
        """
    
        if not (weights in {'pascal_voc', None}):
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `pascal_voc` '
                             '(pre-trained on PASCAL VOC)')
    
        if K.backend() != 'tensorflow':
            raise RuntimeError('The Deeplabv3+ model is only available with '
                               'the TensorFlow backend.')
    
        if not (backbone in {'xception', 'mobilenetv2'}):
            raise ValueError('The `backbone` argument should be either '
                             '`xception`  or `mobilenetv2` ')
    
        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor
    
        if backbone == 'xception':
            if OS == 8:
                entry_block3_stride = 1
                middle_block_rate = 2  # ! Not mentioned in paper, but required
                exit_block_rates = (2, 4)
                atrous_rates = (12, 24, 36)
            else:
                entry_block3_stride = 2
                middle_block_rate = 1
                exit_block_rates = (1, 2)
                atrous_rates = (6, 12, 18)
    
            x = Conv2D(32, (3, 3), strides=(2, 2),
                       name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
            x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
            x = Activation('relu')(x)
    
            x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
            x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
            x = Activation('relu')(x)
    
            x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                                skip_connection_type='conv', stride=2,
                                depth_activation=False)
            x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                                       skip_connection_type='conv', stride=2,
                                       depth_activation=False, return_skip=True)
    
            x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                                skip_connection_type='conv', stride=entry_block3_stride,
                                depth_activation=False)
            for i in range(16):
                x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                                    skip_connection_type='sum', stride=1, rate=middle_block_rate,
                                    depth_activation=False)
    
            x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                                skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                                depth_activation=False)
            x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                                skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                                depth_activation=True)
    
        else:
            OS = 8
            first_block_filters = _make_divisible(32 * alpha, 8)
            x = Conv2D(first_block_filters,
                       kernel_size=3,
                       strides=(2, 2), padding='same',
                       use_bias=False, name='Conv')(img_input)
            x = BatchNormalization(
                epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
            x = Activation(relu6, name='Conv_Relu6')(x)
    
            x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                                    expansion=1, block_id=0, skip_connection=False)
    
            x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                                    expansion=6, block_id=1, skip_connection=False)
            x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                                    expansion=6, block_id=2, skip_connection=True)
    
            x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                                    expansion=6, block_id=3, skip_connection=False)
            x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                    expansion=6, block_id=4, skip_connection=True)
            x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                    expansion=6, block_id=5, skip_connection=True)
    
            # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
            x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                                    expansion=6, block_id=6, skip_connection=False)
            x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                    expansion=6, block_id=7, skip_connection=True)
            x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                    expansion=6, block_id=8, skip_connection=True)
            x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                    expansion=6, block_id=9, skip_connection=True)
    
            x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                    expansion=6, block_id=10, skip_connection=False)
            x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                    expansion=6, block_id=11, skip_connection=True)
            x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                    expansion=6, block_id=12, skip_connection=True)
    
            x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                                    expansion=6, block_id=13, skip_connection=False)
            x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                    expansion=6, block_id=14, skip_connection=True)
            x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                    expansion=6, block_id=15, skip_connection=True)
    
            x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                                    expansion=6, block_id=16, skip_connection=False)
    
        # end of feature extractor
    
        # branching for Atrous Spatial Pyramid Pooling
    
        # Image Feature branch
        #out_shape = int(np.ceil(input_shape[0] / OS))
        b4 = AveragePooling2D(pool_size=(int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(x)
        b4 = Conv2D(256, (1, 1), padding='same',
                    use_bias=False, name='image_pooling')(b4)
        b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
        b4 = Activation('relu')(b4)
        b4 = BilinearUpsampling((int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(b4)
    
        # simple 1x1
        b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
        b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
        b0 = Activation('relu', name='aspp0_activation')(b0)
    
        # there are only 2 branches in mobilenetV2. not sure why
        if backbone == 'xception':
            # rate = 6 (12)
            b1 = SepConv_BN(x, 256, 'aspp1',
                            rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
            # rate = 12 (24)
            b2 = SepConv_BN(x, 256, 'aspp2',
                            rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
            # rate = 18 (36)
            b3 = SepConv_BN(x, 256, 'aspp3',
                            rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)
    
            # concatenate ASPP branches & project
            x = Concatenate()([b4, b0, b1, b2, b3])
        else:
            x = Concatenate()([b4, b0])
    
        x = Conv2D(256, (1, 1), padding='same',
                   use_bias=False, name='concat_projection')(x)
        x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = Dropout(0.1)(x)
    
        # DeepLab v.3+ decoder
    
        if backbone == 'xception':
            # Feature projection
            # x4 (x2) block
            x = BilinearUpsampling(output_size=(int(np.ceil(input_shape[0] / 4)),
                                                int(np.ceil(input_shape[1] / 4))))(x)
            dec_skip1 = Conv2D(48, (1, 1), padding='same',
                               use_bias=False, name='feature_projection0')(skip1)
            dec_skip1 = BatchNormalization(
                name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
            dec_skip1 = Activation('relu')(dec_skip1)
            x = Concatenate()([x, dec_skip1])
            x = SepConv_BN(x, 256, 'decoder_conv0',
                           depth_activation=True, epsilon=1e-5)
            x = SepConv_BN(x, 256, 'decoder_conv1',
                           depth_activation=True, epsilon=1e-5)
    
        # you can use it with arbitary number of classes
        if classes == 21:
            last_layer_name = 'logits_semantic'
        else:
            last_layer_name = 'custom_logits_semantic'
    
        x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
        x = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(x)
        x = Activation('sigmoid')(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input
    
        model = Model(inputs, x, name='deeplabv3plus')
    
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
        y_pred_in = y_pred_in > train_mask_threshold 
        batch_size = y_true_in.shape[0]
        metric = []
        for batch in range(batch_size):
            value = iou_metric(y_true_in[batch], y_pred_in[batch])
            metric.append(value)
        return np.mean(metric)
    
    def my_iou_metric(label, pred):
        metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float64)
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
    #input_layer = Input((img_size_target, img_size_target, 1))
    #output_layer = build_model(input_layer, start_neurons_arg , dropout_arg)
    
    adam = Adam(lr=lr_arg)
    #  backbone='xception' OS, alpha
    model = Deeplabv3(input_shape=(101,101,1), classes=1, alpha=0.8)  
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=[my_iou_metric, 'accuracy'])
    
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode = 'max',patience=20, verbose=1)
    model_checkpoint = ModelCheckpoint("./unet_best1.model",monitor='val_my_iou_metric', 
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode = 'max',factor=0.2, patience=5, min_lr=0.0000001, verbose=1)
        
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
    
    
#my_main(batch_size=32, epochs=1)