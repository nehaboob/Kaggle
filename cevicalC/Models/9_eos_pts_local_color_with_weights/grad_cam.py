#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 21:53:50 2017

@author: neha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 19:30:19 2017

@author: neha

https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py

this code does not work properly
"""

from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os


img_width = 100
img_height = 100
weights_path = 'eos_points_deepsense_2_ep50_100X100.h5'
img_path = '../../train_crop/Type_1/1273.jpg'
nb_classes = 10
K.set_learning_phase(0)

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = path
    img = image.load_img(img_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    ix = np.array([img_to_array(img)])
    print(x.shape)
    print(ix.shape)
    #preprocess image
    x = x*(1./255)
    return x

def init_model(weights_path):
    root_path = '.'
    weights_path = os.path.join(root_path,weights_path)
    model = load_model(weights_path)
    print('Model loaded.')
    print(model.input)
    return model

def grad_cam(input_model, image, category_index, layer_name, pred_val, nb_classes):
    model = Sequential()
    model.add(input_model)

    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer,output_shape = target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    
    conv_output =  [layer for idx, layer in enumerate(model.layers[0].layers) if layer.name == layer_name][0].output

    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))

    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (img_width, img_height))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    
    heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_SPRING)
    heatmap_colored[np.where(heatmap <= 0.2)] = 0
    heatmap_colored = heatmap_colored.astype(np.float32)
    
    cam = cv2.addWeighted(image[0]*255., 1, heatmap_colored, 0.5, 0)
    
    return cam, heatmap_colored

# get the imput image
preprocessed_input = load_image(img_path)

# load the model
model = init_model(weights_path)
model.summary()

predictions = model.predict(preprocessed_input)
print('Predicted class:')
print(predictions)

predicted_class = np.argmax(predictions)
predicted_score = predictions[0][predicted_class]
cam, heatmap_colored = grad_cam(model, preprocessed_input, predicted_class, "conv2d_5", predicted_score, nb_classes)
cv2.imwrite("gradcam.jpg", cam)

