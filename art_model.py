#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 18:17:16 2018
@author: Michael Samon

Art Classifier in Keras
"""

import os
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import layers, models
from random import randint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageChops
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)


# define image folder locations
basedir = "/Users/user/Desktop/RRCC/Github Projects/ArtClassifier"

train_dir = os.path.join(basedir, "transforms/")
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
        rescale=1./255)

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


def strong_aug(p=0.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=p),
        OneOf([
            MotionBlur(p=p),
        ], p=p),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=p),
        OneOf([
            OpticalDistortion(p=p),
            GridDistortion(p=p),
            IAAPiecewiseAffine(p=p),
        ], p=p),
        OneOf([
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=p),
        HueSaturationValue(p=p),
    ], p=p)


# this function found here: 
# https://stackoverflow.com/questions/10615901/trim-whitespace-using-pil
def crop_whitespace():
    # remove '.DS_Store' from list
    artists = sorted(os.listdir(train_dir))[1:]
    for artist in artists:
        img_paths = [os.path.join(train_dir + artist, fname) for fname in os.listdir(train_dir + artist)]

        for img_path in img_paths:
            im = Image.open(img_path)
            bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
            diff = ImageChops.difference(im, bg)
            diff = ImageChops.add(diff, diff, 2.0, -100)
            bbox = diff.getbbox()
            if bbox:
                im = im.crop(bbox)
                im.save(img_path)


def create_wack_transforms(artist):
    img_names = os.listdir(train_dir + artist)
    fnames = [os.path.join(train_dir + artist, fname) for fname in os.listdir(train_dir + artist)]

    for j, img_path in enumerate(fnames):
        img_name = img_names[j]
        img = Image.open(img_path)

        img = img.resize(
            (300, 300), Image.ANTIALIAS
        )  # ANTIALIAS resizes with best quality

        img = np.array(img)

        augmentor = strong_aug(p=0.9) 

        # create 8 transforms from each image
        for i in range(8):
            augmented_img = augmentor(image=img, p=0.9)["image"]

            fig = plt.figure(i)

            # remove axis and scale content correctly
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            ax.imshow(augmented_img)
            fig.savefig('transforms/'+ artist + " " + str(i) + " " + img_name, bbox_inches='tight')
 

def visualize_transforms(artist):
    fnames = [os.path.join(train_dir + artist, fname) for fname in os.listdir(train_dir + artist)]
    img_names = os.listdir(train_dir + artist)

    for j, img_path in enumerate(fnames):
        img_name = img_names[j]
        img = image.load_img(img_path, target_size=(250, 250))

        x = image.img_to_array(img)

        i = 0
        for batch in train_datagen.flow(x, batch_size=1):
            fig = plt.figure(i)

            # remove axis and scale content correctly
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            ax.imshow(image.array_to_img(batch[0]))
            fig.savefig('transforms/'+ artist + " " + str(i) + " " + img_name, bbox_inches='tight')

            i += 1
            if i % 8 == 0:
                break

        #plt.show()


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


def view_activations(layer_index, artist, img_index=None):
    fnames = [os.path.join(train_dir + artist, fname) for fname in os.listdir(train_dir + artist)]

    rand_index = 0
    if img_index is None:
        rand_index = randint(0, len(fnames) - 1)
    else:
        rand_index = img_index

    img_path = fnames[rand_index]

    view_img = mpimg.imread(img_path)
    print(img_path, rand_index)
    plt.imshow(view_img)
    img = image.load_img(img_path, target_size=(250, 250))

    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x /= 255

    layer_outputs = [layer.output for layer in model.layers]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(x)[layer_index]

    # first_activations = activations[0]
    # end_activations = activations[10]

    images_per_row = 16

    n_features = activations.shape[-1]
    size = activations.shape[1]

    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = activations[0, :, :, col * images_per_row + row]

            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,
            row * size: (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title("Layer {}".format(layer_index))
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto')


if __name__ == "__main__":
    pass