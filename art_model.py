#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 18:17:16 2018
@author: Michael Samon

Art Classifier in Keras
"""

import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models



# define image folder locations
basedir = "/Users/user/Desktop/RRCC/ArtClassifier/"

train_dir = os.path.join(basedir, "images/train/")
test_dir = os.path.join(basedir, "images/test/")

# read in images
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(250,250),
        batch_size=5
        )

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(250,250),
        batch_size=5
        )



def art_model(epochs=10):
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (5, 5), activation='tanh',
                            input_shape=(250, 250, 3)))
    
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (5, 5), activation='tanh'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='tanh'))
    model.add(layers.Conv2D(128, (3, 3), activation='tanh'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='tanh'))
    model.add(layers.MaxPooling2D((2, 2)))    
    model.add(layers.Conv2D(128, (3, 3), activation='tanh'))
    model.add(layers.Conv2D(128, (3, 3), activation='tanh'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(8, activation='softmax')) 
    
    model.summary()
    
    model.compile(loss=keras.losses.categorical_crossentropy, 
                  optimizer=keras.optimizers.Adam(),
                  metrics=['acc']
                  )
    
    callbacks = [
        keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5),
        keras.callbacks.ModelCheckpoint('./'+'deeep'+'.h5',
                                        monitor='acc'),
        keras.callbacks.TensorBoard(log_dir="./logs/deeeep/")
                ]
    
    model.fit_generator(
      train_generator,
      steps_per_epoch=40,
      epochs=epochs,
      validation_data=test_generator,
      validation_steps=40,
      callbacks=callbacks)



art_model(epochs=50)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
