# -*- coding: utf-8 -*-"""Spyder EditorThis is a temporary script file."""# Import library to load modelfrom keras.models import load_model#latest_trained_model = load_model('ApparelVGG16.h5')from keras.preprocessing.image import ImageDataGeneratortrain_datagen = ImageDataGenerator(rescale = 1./255,                                   shear_range = 0.2,                                   zoom_range = 0.2,horizontal_flip = True)test_datagen = ImageDataGenerator(rescale = 1./255)from sklearn.externals import joblib#train_X = train_datagen.flow(joblib.load('trainImage.pkl'),#                             joblib.load('trainLabels.pkl'),#                             batch_size=32)train_images = joblib.load('trainImage.pkl')train_labels = joblib.load('trainLabels.pkl')from keras.models import Modelfrom keras.applications.vgg16 import VGG16from keras.models import Sequentialfrom keras.optimizers import SGDfrom keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activationdef vgg16_model(img_rows, img_cols, channel=1, num_classes=None):    model = VGG16(weights='imagenet', include_top=True)    model.layers.pop() #get rid of prediction layer    model.layers.pop() #get rid of fc2 layer    model.layers.pop() #get rid of fc1 layer        model.outputs = [model.layers[-1].output]    model.layers[-1].outbound_nodes = []    #x=Dense(num_classes, activation='softmax')(model.output)    model1=Model(model.input,model.outputs)        x=Dense(num_classes, activation='softmax')(model1.output)        model1=Model(model1.input,x)#To set the first 8 layers to non-trainable (weights will not be updated)    for layer in model1.layers[:14]:        layer.trainable = False# Learning rate is changed to 0.001    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)    model1.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])    return model1img_rows, img_cols = 224, 224 # Resolution of inputschannel = 3num_classes = 15batch_size = 32nb_epoch = 8model = vgg16_model(img_rows, img_cols, channel, num_classes)model.summary()model.fit_generator(train_datagen.flow(train_images,                             train_labels,                             batch_size=32), steps_per_epoch = 222,                     epochs = nb_epoch)model.save('VGG16.h5')model.summary()outputs = model.get_layer('flatten').outputinputs = model.input intermediate_layer_model = Model (inputs,outputs)intermediate_layer_model.summary()#Feature Scalingfrom sklearn.preprocessing import StandardScalersc_X = StandardScaler()#X_train = sc_X.fit_transform(X_train)#X_test = sc_X.transform(X_test)result = sc_X.fit_transform(intermediate_layer_model.predict_generator(train_datagen.flow(train_images,batch_size=32),steps=20))from sklearn.decomposition import PCApca = PCA(n_components=234)result = pca.fit_transform (result)explain = pca.explained_variance_ratio_# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Import library to load model
from keras.models import load_model

#latest_trained_model = load_model('ApparelVGG16.h5')

from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


from sklearn.externals import joblib
#train_X = train_datagen.flow(joblib.load('trainImage.pkl'),
#                             joblib.load('trainLabels.pkl'),
#                             batch_size=32)


train_images = joblib.load('trainImage.pkl')
train_labels = joblib.load('trainLabels.pkl')




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

    for layer in model1.layers[:14]:

        layer.trainable = False

# Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model1.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model1

img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3
num_classes = 15
batch_size = 32
nb_epoch = 8

model = vgg16_model(img_rows, img_cols, channel, num_classes)

model.summary()

model.fit_generator(train_datagen.flow(train_images,
                             train_labels,
                             batch_size=32), steps_per_epoch = 222, 
                    epochs = nb_epoch)

model.save('VGG16.h5')


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

result = sc_X.fit_transform(intermediate_layer_model.predict_generator(train_datagen.flow(train_images,batch_size=32),steps=20))


from sklearn.decomposition import PCA
#234 components explained > 65% (67.6) of variance
pca = PCA(n_components=234)

result = pca.fit_transform (result)

explain = pca.explained_variance_ratio_

from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(result)
