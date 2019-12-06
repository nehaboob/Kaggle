#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 13:33:10 2017

@author: neha
"""

#subtract local color
import numpy as np
import cv2
from skimage import color
from skimage.exposure import equalize_hist

# 0 10 81 129 109

#gaussian blurr
img = cv2.imread('../../data/train_split_clahe_256px_crop_auto_30/Type_1/81.jpg')
img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 30), -4, 128)
cv2.imwrite('test.jpg', img)


#median blurr
img = cv2.imread('../../data/train_split_clahe_256px_crop_auto_30/Type_1/129.jpg')
img = cv2.addWeighted(img, 10, cv2.medianBlur(img, 100), -10, 128)
cv2.imwrite('test.jpg', cv2.medianBlur(img, 11))
cv2.imwrite('test1.jpg', cv2.blur(img,(11,11)))

cv2.imwrite('test.jpg', (cv2.medianBlur(img, 11)/2+cv2.blur(img,(11,11))/2))
cv2.imwrite('test1.jpg', cv2.blur(img,(11,11)))

cv2.imwrite('test1.jpg',cv2.GaussianBlur(img, (0,0), 5))

#get the green channel
img = cv2.imread('../../data/train_split_clahe_256px_crop_auto_30/Type_1/129.jpg')
img[:,:,2] = 0
img[:,:,0] = 0
cv2.imwrite('test.jpg', img)
   

#reduce number of colors
from sklearn.cluster import MiniBatchKMeans
image = cv2.imread('../../data/train_split_clahe_256px_crop_auto_30/Type_1/129.jpg')
#image = cv2.medianBlur(img, 11)
(h, w) = image.shape[:2]
 
# convert the image from the RGB color space to the L*a*b*
# color space -- since we will be clustering using k-means
# which is based on the euclidean distance, we'll use the
# L*a*b* color space where the euclidean distance implies
# perceptual meaning
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
 
# reshape the image into a feature vector so that k-means
# can be applied
image = image.reshape((image.shape[0] * image.shape[1], 3))
 
# apply k-means using the specified number of clusters and
# then create the quantized image based on the predictions
clt = MiniBatchKMeans(n_clusters = 2)
labels = clt.fit_predict(image)
quant = clt.cluster_centers_.astype("uint8")[labels]
 
# reshape the feature vectors to images
quant = quant.reshape((h, w, 3))
image = image.reshape((h, w, 3))
 
# convert from L*a*b* to RGB
quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
 
# two color image

two_color = np.hstack([image, quant])

from sklearn.cluster import MiniBatchKMeans
img = cv2.imread('../../data/train_split_clahe_256px_crop_auto_30/Type_1/129.jpg')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

img = equalize_hist(img)
thresh = np.percentile(img, 5)

binary = img > thresh
ret = np.empty((200, 200, 3), dtype=bool)
ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  np.invert(binary)
   
quant[ret] = 0

# display the images and wait for a keypress
cv2.imshow("image", np.hstack([image, quant]))
cv2.waitKey(0)



import numpy as np
from skimage import data
import matplotlib.pyplot as plt

#to check histogram of the image px
coins = data.coins()
hist, bin_edges = np.histogram(coins, bins=np.arange(0, 256))
plt.bar(bin_edges[:-1], hist, width = 1)
plt.xlim(min(bin_edges), max(bin_edges))
plt.show()

#convert image to grey scale
from skimage import color
from skimage import io

img_color = io.imread('../../81.jpg')
img = color.rgb2gray(io.imread('../../81.jpg'))
io.imsave("gery.jpg", img)
io.imshow(img)


# Find contours at a constant value of 0.8
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
from skimage import measure

img_color = color.rgb2gray(io.imread('../../data/train/Type_1/81.jpg'))

contours = measure.find_contours(img_color, 0.3)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(img_color, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()


#edge detection

from skimage.filters import roberts, sobel, scharr, prewitt
image = color.rgb2gray(io.imread('../../data/train/Type_1/81.jpg'))

edge_roberts = roberts(image)
edge_sobel = sobel(image)

fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                       figsize=(8, 4))

ax[0].imshow(edge_roberts, cmap=plt.cm.gray)
ax[0].set_title('Roberts Edge Detection')

ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
ax[1].set_title('Sobel Edge Detection')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()


#edge detection
from skimage import feature
image = color.rgb2gray(io.imread('../../data/train/Type_1/81.jpg'))

edges1 = feature.canny(image)
io.imshow(edges1)


#active countor map

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# Test scipy version, since active contour is only possible
# with recent scipy version
import scipy
split_version = scipy.__version__.split('.')
if not(split_version[-1].isdigit()): # Remove dev string if present
        split_version.pop()
scipy_version = list(map(int, split_version))
new_scipy = scipy_version[0] > 0 or (scipy_version[0] == 0 and scipy_version[1] >= 14)

img = color.rgb2gray(io.imread('../../data/train/Type_1/81.jpg'))

s = np.linspace(0, 2*np.pi, 400)
x = 220 + 100*np.cos(s)
y = 100 + 100*np.sin(s)
init = np.array([x, y]).T

if not new_scipy:
    print('You are using an old version of scipy. '
          'Active contours is implemented for scipy versions '
          '0.14.0 and above.')

if new_scipy:
    snake = active_contour(gaussian(img, 3),
                           init, alpha=0.015, beta=10, gamma=0.001)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    plt.gray()
    ax.imshow(img)
    ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

#equalized image 

from skimage.exposure import equalize_hist
equalized_image = equalize_hist(img)
io.imshow(equalized_image)

from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert

# Invert the horse image
image = color.rgb2gray(io.imread('../../data/train/Type_1/81.jpg'))

# perform skeletonization
skeleton = skeletonize(image)

# display results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})

ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

fig.tight_layout()
plt.show()


# thresholding

from skimage.filters import threshold_mean
from skimage.exposure import equalize_hist
from skimage.filters import threshold_minimum
import numpy as np

image = color.rgb2gray(io.imread('../../data/train/Type_3/69.jpg'))
#image = equalize_hist(image)
thresh = np.percentile(image, 5)

binary = image > thresh


fig, axes = plt.subplots(ncols=2, figsize=(8, 3))
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Original image')

ax[1].imshow(binary, cmap=plt.cm.gray)
ax[1].set_title('Result')

for a in ax:
    a.axis('off')

plt.show()




# corner detection
from skimage.feature import corner_harris,corner_peaks


def show_corners(corners,image,title=None):
    """Display a list of corners overlapping an image"""
    fig = plt.figure()
    plt.imshow(image)
    # Convert coordinates to x and y lists
    y_corner,x_corner = zip(*corners)
    plt.plot(x_corner,y_corner,'o') # Plot corners
    if title:
        plt.title(title)
    plt.xlim(0,image.shape[1])
    plt.ylim(image.shape[0],0) # Images use weird axes
    fig.set_size_inches(np.array(fig.get_size_inches()) * 1.5)
    plt.show()
    print("Number of corners:",len(corners))


# Run Harris
checkers_corners = corner_peaks(corner_harris(equalized_image),min_distance=2)
show_corners(checkers_corners,equalized_image)

#edge detection
from skimage import feature
edges1 = feature.canny(equalized_image)
io.imshow(edges1)
