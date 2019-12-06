#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:25:28 2017

@author: neha
"""

import cv2
import numpy as np

from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from vis.utils import utils
from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam
from vis.visualization import visualize_activation, get_num_filters
from keras.models import load_model
import os

#for large image size
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# get the model for checking saliency map
def init_model(weights_path):
    root_path = '.'
    weights_path = os.path.join(root_path,weights_path)
    model = load_model(weights_path)
    print('Model loaded.')
    print(model.input)
    return model

def grad_cam(root_path, out_img, img_width, img_height, model, layer_idx,penultimate_layer_idx):
    items = os.listdir(root_path)
    
    image_paths = []
    for names in items:
        if names.endswith(".jpg"):
            image_paths.append(root_path+"/"+names)
    print(image_paths)
    
    heatmaps = []
    for path in image_paths:
        # Predict the corresponding class for use in `visualize_saliency`.
        keras_img = image.load_img(path, target_size=(img_width, img_height))
        seed_img = np.array([img_to_array(keras_img)])
        img_array = seed_img*(1./255)
        pred_array = model.predict(img_array)
        pred_class = np.argmax(pred_array)
        seed_img = img_array[0]
        #seed_img = utils.load_img(path, target_size=(300,300))
        #print(pred_class)
        print(os.path.basename(path)+':Type_'+str(pred_class))
        print(pred_array)
        #print(pred_class)
        # Here we are asking it to show attention such that prob of `pred_class` is maximized.
        #heatmap = visualize_saliency(model, layer_idx, [pred_class], seed_img, text=pred_class)
        # Here we are asking it to show attention such that prob of `pred_class` is maximized.
        heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img, penultimate_layer_idx, text=os.path.basename(path)+':Type_'+str(pred_class))
       # cv2.circle(heatmap,(pred_array[0][3],pred_array[0][4]), 3, (0,0,255), -1)
       # cv2.circle(heatmap,(pred_array[0][5],pred_array[0][6]), 3, (0,0,255), -1)
       # cv2.circle(heatmap,(pred_array[0][7],pred_array[0][8]), 3, (0,0,255), -1)

        heatmaps.append(heatmap)
    
    #cv2.imshow("Saliency map", utils.stitch_images(heatmaps))
    cv2.imwrite(out_img,utils.stitch_images(heatmaps))

def conv_filters(out_img, model, layer_idx):
    # Visualize all filters in this layer.
    filters = np.arange(get_num_filters(model.layers[layer_idx]))

    # Generate input image for each filter. Here `text` field is used to overlay `filter_value` on top of the image.
    vis_images = [visualize_activation(model, layer_idx, filter_indices=idx, text=str(idx)) for idx in filters]

    # Generate stitched image pallette with 10 cols.
    cv2.imwrite(out_img, utils.stitch_images(vis_images, cols=8))
    
def dense_layer(out_img, model, layer_idx, class_index):
    # Generate three different images of the same output index.
    vis_images = [visualize_activation(model, layer_idx, filter_indices=idx, text=str(idx), max_iter=500) for idx in [class_index, class_index, class_index]]
    cv2.imwrite(out_img, utils.stitch_images(vis_images))


root_path = "../../data/cam_samples"
img_width = 200
img_height = 200
weights_path = 'ex1_model_clahe_new.h5'

# load the model
model = init_model(weights_path)
#model.summary()

# let's visualize layer names and layer indices to see get the name of the layer for saliency map
for i, layer in enumerate(model.layers):
   print(i, layer.name, layer.trainable)

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'dense_1'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]
class_idx = 2
penultimate_layer_idx = 39

conv_filters('conv2d_1.png', model, 11)
dense_layer('type_2_input.png',model, layer_idx, class_idx)
grad_cam(root_path, 'auto_crop_adam.png', img_width, img_height, model, layer_idx, penultimate_layer_idx)

