#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:10:57 2017

@author: neha
"""

from PIL import Image, ImageDraw

path = '../../data/train/Type_1/1134.jpg'
r_tuple = (829, 1106)

img = Image.open(path)
img = img.resize(r_tuple)

draw = ImageDraw.Draw(img)
x = 1160
y = 2462
r = 5

x1=249.15478928	
y1=357.30633664	
x2=301.0702552	
y2=355.43082816	
x3=346.82363232	
y3=352.31144896

#draw.point((163, 142), 'black')
draw.ellipse((x1-r, y1-r, x1+r, y1+r), fill=(0,0,0,0))
draw.ellipse((x2-r, y2-r, x2+r, y2+r), fill=(0,0,0,0))
draw.ellipse((x3-r, y3-r, x3+r, y3+r), fill=(0,0,0,0))

#draw.point((1403, 1551), 'white')
img.save('test.png')



from PIL import Image, ImageDraw

path = '../../data/train/Type_1/1134.jpg'
r_tuple = (300, 300)

img = Image.open(path)
#img = img.resize(r_tuple)

draw = ImageDraw.Draw(img)
x = 1160
y = 2462
r = 13

x1=961.15478928	
y1=1374.30633664	
x2=1158.0702552	
y2=1366.43082816	
x3=1333.82363232	
y3=1356.31144896

#draw.point((163, 142), 'black')
draw.ellipse((x1-r, y1-r, x1+r, y1+r), fill=(0,0,0,0))
draw.ellipse((x2-r, y2-r, x2+r, y2+r), fill=(0,0,0,0))
draw.ellipse((x3-r, y3-r, x3+r, y3+r), fill=(0,0,0,0))

#draw.point((1403, 1551), 'white')
img.save('test.png')




# crop image using PIL
x = 1
y = 1
w = 100
h = 100

im = Image.open('0.png').convert('L')
im = im.crop((x-(w/2), y-(h/2), x+(w/2), y+(h/2)))
im.save('_0.png')


# do not use ....its very slow
import cv2

img = cv2.imread('../../train/Type_1/0.jpg')
resized_image = cv2.resize(img, (300, 300)) 
cv2.circle(resized_image,(122,190), 3, (0,0,0), -1)
cv2.imshow('img',resized_image)
cv2.waitKey()


# remove local color form image
# first blur the image and then subtract blurred image from original image
from PIL import Image
from PIL import ImageFilter
from PIL import ImageChops
path = '../../train/Type_2/305.jpg'
img = Image.open(path)

im1 = img.filter(ImageFilter.GaussianBlur(radius=50))

im_new = ImageChops.difference(img, im1)
im_new.save('test.png')



# using cv2 
import cv2
img = cv2.imread('../../train_crop_auto/Type_1/81.jpg')
img = cv2.resize(img, (300,300))
img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 20), -4, 128)
cv2.imwrite('test.png', img)


# histogram equalization of color images
b, g, r = cv2.split(img)
red = cv2.equalizeHist(r)
green = cv2.equalizeHist(g)
blue = cv2.equalizeHist(b)
img = cv2.merge((blue, green, red))


