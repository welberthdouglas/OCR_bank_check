#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 06:20:50 2020

@author: welberth
"""

#____________________________importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Lambda,Dense,Flatten
from keras.utils.np_utils import to_categorical

#____________________________loading Data

train_data=pd.read_csv('train.csv',delimiter=',')
test_data=pd.read_csv('test.csv',delimiter=',')
#train_data.head()

#____________________________Partitioning data as images and labels / transforming into arrays

train_label=train_data.iloc[:,0].values
train_images=train_data.iloc[:,1:].values

test_images=test_data.values
test_images=test_images.reshape(test_images.shape[0],28,28) #reshaping to a 3D array of images 28X28
train_images=train_images.reshape(train_images.shape[0],28,28) 

#____________________________Visualizing images

#random_image=np.random.choice(train_images.shape[0],4) #chosing 4 random images from the training data


#for i in range(0,4):   #ploting images and labels
#    plt.subplot(2,2,(i+1))
#    plt.imshow(train_images[random_image[i]],cmap='binary')
#    plt.title(train_label[random_image[i]])
#    plt.subplots_adjust(top=1.3)
#    plt.axis('off')

#____________________________Normalizing data (important to put put inputs and outputs in the same magnitude )
mean=train_images.mean()   
std=train_images.std()

def normalize(x): 
    return (x-mean)/std
  
#____________________________Setting random seed
np.random.seed(8)

#changing labels to match the number of output neurons (ex: 4 -> [0,0,0,0,1,0,0,0,0,0])
train_label= to_categorical(train_label)

#____________________________Creating an ANN model

model=Sequential()
model.add(Lambda(normalize,input_shape=(28,28)))
model.add(Flatten())
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='sigmoid'))

print("input shape ",model.input_shape)   #input and output shapes
print("output shape ",model.output_shape)

#____________________________Compiling and training model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_images, train_label, epochs=10, batch_size=64)


prediction=model.predict_classes(test_images)
#prediction[10]
#plt.imshow(test_images[10],cmap='binary')

#____________________________Visualizing predictions vs test images

random_test=np.random.choice(test_images.shape[0],20) #chosing 20 random images from training data

for i in range(0,20):   #ploting images and predictions
    plt.subplot(5,5,(i+1))
    plt.imshow(test_images[random_test[i]],cmap='binary')
    plt.title(prediction[random_test[i]])
    plt.subplots_adjust(top=1.5)
    plt.axis('off')