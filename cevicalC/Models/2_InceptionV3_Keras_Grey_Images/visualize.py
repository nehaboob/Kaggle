#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:25:28 2017

@author: neha
"""

import cv2
import numpy as np

from keras.preprocessing.image import img_to_array
from vis.utils import utils
from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam
import os

#for large image size
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# get the model for checking saliency map
from keras.models import load_model
root_path = '.'
weights_path = os.path.join(root_path,'InceptionV3_Pop_lr001_epoch40.h5')
model = load_model(weights_path)
print('Model loaded.')
print(model.input)

# let's visualize layer names and layer indices to see get the name of the layer for saliency map
for i, layer in enumerate(model.layers):
   print(i, layer.name, layer.trainable)

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'dense_2'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Images corresponding to a type
#image_paths = ["../../train/Type_1/48.jpg","../../train/Type_2/15.jpg"]

root_path = "../../train/Type_1"
items = os.listdir(root_path)

image_paths = []
for names in items:
    if names.endswith(".jpg"):
        image_paths.append(root_path+"/"+names)
print(image_paths)

heatmaps = []
for path in image_paths:
    # Predict the corresponding class for use in `visualize_saliency`.
    seed_img = utils.load_img(path, target_size=(299, 299))
    pred_class = np.argmax(model.predict(np.array([img_to_array(seed_img)])))
    print(pred_class)

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    #heatmap = visualize_saliency(model, layer_idx, [pred_class], seed_img, text=pred_class)
    
    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img)
    heatmaps.append(heatmap)

#cv2.imshow("Saliency map", utils.stitch_images(heatmaps))
cv2.imwrite('type1_cam.png',utils.stitch_images(heatmaps))


