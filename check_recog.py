#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 06:07:52 2020

@author: welberth
"""

#____________________Required libraries
#!pip3 --proxy 'proxyapp.santanderbr.corp:80' install opencv-contrib-python

#____________________Importing libraries

import numpy as np
import matplotlib.pyplot as plt
#import argparse
import cv2

#____________________Argument parser

#parser=argparse.ArgumentParser()
#parser.add_argument('-i','--image', required=True, help='Path to the image')
#args=vars(parser.parse_args())

#____________________Loading and resizing image

img_original=cv2.imread('itaug2.png')

hg = 500
ratio = hg/img_original.shape[0]
dim = (int(img_original.shape[1]*ratio),hg)
img_original=cv2.resize(img_original,dim,interpolation=cv2.INTER_AREA)

img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)  #converting image to grayscale color scheme

#____________________Aplying a mask

mask = np.zeros(img.shape[:], dtype = "uint8")
cv2.rectangle(mask, (0, int(hg*0.84)), (img.shape[1], int(hg*0.46)), 255, -1)
cv2.rectangle(mask, (int(img.shape[1]*0.42), int(hg*0.58)), (img.shape[1], 0), 0, -1)
masked=cv2.bitwise_and(img,img,mask=mask)

#____________________Defining kernels

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 9))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

#____________________Image filtering and manipulation

masked_blur=cv2.medianBlur(masked, 5)
blackhat = cv2.morphologyEx(masked_blur, cv2.MORPH_BLACKHAT, sqKernel)
sobelx = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
sobelx = np.uint8(np.absolute(sobelx)) # used to evidence gradients with negative sign

#sobelx = np.absolute(sobelx)   
(minVal, maxVal) = (np.min(sobelx), np.max(sobelx))
sobelx = (255 * ((sobelx - minVal) / (maxVal - minVal))).astype("uint8")

thresh = cv2.morphologyEx(sobelx, cv2.MORPH_CLOSE, rectKernel)
#thresh = cv2.morphologyEx(sobelx, cv2.MORPH_CLOSE, rectKernel)

thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh = cv2.erode(thresh,sqKernel,iterations = 1)
thresh = cv2.dilate(thresh,sqKernel,iterations = 1)

#____________________Finding contours and ploting ROI boxes

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

roi=[]
for (i,c) in enumerate(cnts):
    (a,b,l,h)= cv2.boundingRect(c)
    
    a_ratio=l/(float(h))
    area=float(l)*float(h)
    
    bufv=5
    bufh=20
    (a_b, b_b, l_b, h_b)=(a-bufh, b-bufv, a+l+bufh, b+h+bufv)
    if a_ratio>1.2 and area>600:
        cv2.rectangle(img_original,(a_b,b_b),(l_b,h_b),(0,255,0),2)
        roi.append((a_b,b_b,l_b, h_b))
        
        
#____________________Process checker
#plt.imshow(blackhat,cmap='gray')
#plt.imshow(sobelx,cmap='gray')
#plt.imshow(thresh,cmap='gray')

#Find a way to extract each image

roi = sorted(roi, key=lambda x:x[0])

plt.imshow(img_original)
cv2.imwrite('ocr_image.jpg',img_original)

crop_img = img[295:379,182:408].copy() 
plt.imshow(crop_img,cmap='gray')
