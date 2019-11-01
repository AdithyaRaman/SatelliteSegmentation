# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 20:49:23 2017

@author: Simran
"""

import numpy as np
from PIL import Image
from PIL import ImageStat
import math

# define color constants
class_color=[[0, 119, 204],
             [0,170,170],
             [76,153,0],
             [210, 210,50],
             [255,255,255],
             [153,76,0],
             [221,170,0]]

# read training data
index = np.genfromtxt("Data/TRAINING WINDOW INDEXES.dat")
index = index.astype(int)
images = ["RED BAND.tif","BLUE BAND.tif","GREEN BAND.tif","INFRARED BAND.tif"]
class_mean = np.empty((7,4))
train_data = np.empty((7,4,484))
for m in range(0, 7):
    band_mean = []
    for i in range(0,4):
        img = images[i]
        im = Image.open("Data/"+img)
        crop_rectangle = (index[m][0], index[m][1], index[m][2], index[m][3])
        cropped_im = im.crop(crop_rectangle)
        train_data[m,i,:] = list(cropped_im.getdata())
        stat = ImageStat.Stat(cropped_im)
        class_mean[m,i]=stat.mean[0]
        im.close()

#calculate class varinace-covariance matrix
class_covariance = np.empty((7,4,4))
class_determinant = np.empty((7,1))
class_inverse = np.empty((7,4,4))
for i in range(0,7):
    class_covariance[i] = np.cov(train_data[i])
    class_inverse[i] = np.linalg.inv(class_covariance[i])
    class_determinant[i] = np.linalg.det(class_covariance[i])

image_vector = np.empty((1024,1024,4))
for i in range(0,len(images)):
    im = Image.open("Data/"+images[i])
    im = np.asarray(im)
    image_vector[:,:,i] = im
            
#calculate Mahalabonis distance!
image_data = np.empty((1024,1024,3),dtype=np.uint8)
prob = np.empty((7,1))
b = class_mean
c = class_inverse
d = class_determinant
for i in range(0,1024):
    for j in range(0,1024):
        a = image_vector[i,j,:]
        e = a-b
        for k in range(0,7):
            prob[k] = np.exp(np.dot(np.transpose(e[k]),np.dot(c[k],e[k]))*(-0.5))/(math.sqrt(d[k]))
#        print(class_color[np.argmax(prob)])
        image_data[i,j,:] = class_color[np.argmax(prob)]
#               
#print(image_data)
#print(np.unique(image_data))                    
#
classified_img = Image.fromarray(image_data, 'RGB')
classified_img.save('Data/CLASSIFIED IMAGE.TIF')
classified_img.show()