# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:49:01 2019

@author: shivy
"""

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


def build_classifier(input_shape=(64, 64, 3), optimizer='adam'):
    model = Sequential()

    # conv-pool-flatten step
    model.add(Convolution2D(
        32, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())

    # classifier feed-forward dense layers
    model.add(Dense(128, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

    # compile the model
    model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    "./cats_and_dogs/training_set",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    "./cats_and_dogs/test_set",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

classifier = build_classifier()
classifier.fit_generator(train_generator,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_generator,
                         validation_steps=2000)
