import os
import sys
import numpy as np

from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import keras

def load_model(_dir, prefix='model'):
    desc_file = os.path.join(_dir,  "{}_{}.json".format(prefix, "spec"))
    weights_file = os.path.join(_dir, "{}_{}.h5".format(prefix, "weights"))

    model = None
    with open(desc_file, "r") as spec_file:
        model = model_from_json(spec_file.read())
    model.load_weights(weights_file)
    return model


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: run_model.py model_dir model_prefix image_path")
        sys.exit(1)
    model_dir = sys.argv[1]
    model_prefix = sys.argv[2]
    model = load_model(model_dir, model_prefix)
    model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

    image_path = sys.argv[3]
    img = image.load_img(image_path, target_size=(64, 64, 3))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img, axis=0)

    prediction = model.predict(img_batch / 255)
    print(prediction)
