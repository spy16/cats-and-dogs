import sys

import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, ResNet50


def run(image_path, model = ResNet50()):
    img = image.load_img(image_path, target_size=(64, 64, 3))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    prediction = model.predict(img_preprocessed)
    print(prediction)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: resnet50.py <image_path>")
        sys.exit(1)
    run(sys.argv[1], ResNet50())
