#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 23:06:10 2017

@author: sourish
"""

#from keras.applications.vgg16 import VGG16
from keras.preprocessing import image # for image preprocessing
import pandas as pd # for data manupulation


#Reading the filelist of training images
train_list=pd.read_table('train.txt',header=None, names= ['Filename'])

#Reading the filelist of testing images
test_list=pd.read_table('test.txt',header=None, names= ['Filename'])

parent_dir='images'
path_separator='/' # for unix like systems
file_format='.jpg'
train_images = [] #declare training image array

for i in range (len(train_list)) :
    
    temp_image = image.load_img(parent_dir+path_separator+train_list['Filename'][i]+file_format,target_size=(224,224))
    
    temp_image = image.img_to_array(temp_image)
    
    train_images.append(temp_image)
    
import numpy as np #for data manupulation
    
train_images=np.array(train_images) 


from keras.applications.vgg16 import preprocess_input

train_images=preprocess_input(train_images)



test_images = [] #declare test image array

for i in range (len(test_list)) :
    
    temp_image = image.load_img(parent_dir+path_separator+test_list['Filename'][i]+file_format,target_size=(224,224))
    
    temp_image = image.img_to_array(temp_image)
    
    test_images.append(temp_image)
    
#import numpy as np #for data manupulation
    
test_images=np.array(test_images) 

test_images=preprocess_input(test_images)


#from keras.applications.vgg16 import preprocess_input


from keras.models import Model

from keras.applications.vgg16 import VGG16

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation



def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):

    model = VGG16(weights='imagenet', include_top=True)

    model.layers.pop() #get rid of prediction layer
    model.layers.pop() #get rid of fc2 layer
    model.layers.pop() #get rid of fc1 layer
    

    model.outputs = [model.layers[-1].output]

    model.layers[-1].outbound_nodes = []

    #x=Dense(num_classes, activation='softmax')(model.output)

    model1=Model(model.input,model.outputs)
    
    x=Dense(num_classes, activation='softmax')(model1.output)
    
    model1=Model(model1.input,x)

#To set the first 8 layers to non-trainable (weights will not be updated)

    for layer in model1.layers[:17]:

        layer.trainable = False

# Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model1.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model1

#Processing the labels
train_labels = np.asarray(pd.read_table('train.txt',delimiter='/',header=None,names=['Label','Sub_Dir','Filename'])['Label'])

test_labels = np.asarray(pd.read_table('test.txt',delimiter='/',header=None,names=['Label','Sub_Dir','Filename'])['Label'])

from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

le = LabelEncoder()

train_labels = le.fit_transform(train_labels)

train_labels = to_categorical(train_labels)

train_labels = np.array(train_labels)

test_labels = le.fit_transform(test_labels)

test_labels = to_categorical(test_labels)

test_labels = np.array(test_labels)


img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3
num_classes = 15
batch_size = 256
nb_epoch = 1

model = vgg16_model(img_rows, img_cols, channel, num_classes)

model.summary()


model.fit(train_images, train_labels,batch_size=batch_size,epochs=nb_epoch,shuffle=True,verbose=1,validation_data=(test_images, test_labels))

model.save('ApparelVGG16.h5')

sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(train_images, train_labels,batch_size=batch_size,epochs=nb_epoch,shuffle=True,verbose=1,validation_data=(test_images, test_labels))

model.save('ApparelVGG16V2.h5')

model.summary()

outputs = model.get_layer('flatten').output
inputs = model.input 

intermediate_layer_model = Model (inputs,outputs)

intermediate_layer_model.summary()

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

result = sc_X.fit_transform(intermediate_layer_model.predict(train_images))

#test_result = sc_X.transform(intermediate_layer_model.predict(test_image))

from sklearn.decomposition import PCA

pca = PCA(n_components=None)

result = pca.fit_transform (result)

explain = pca.explained_variance_ratio_

#test_result = pca.transform(test_result)

from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(result)

#distances,indices = nbrs.kneighbors(test_result)

from sklearn.externals import joblib

filename = 'neighbourModel.pkl'

joblib.dump(nbrs,filename)


