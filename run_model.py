import os
from keras.models import model_from_json
import sys
import numpy as np


def load_model(_dir, prefix='model'):
    desc_file = os.path.join(_dir,  "{}_{}.json".format(prefix, "spec"))
    weights_file = os.path.join(_dir, "{}_{}.h5".format(prefix, "weights"))

    model = None
    with open(desc_file, "r") as spec_file:
        model = model_from_json(spec_file.read())
    model.load_weights(weights_file)
    return model


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run_model.py model_dir model_prefix")
        sys.exit(1)
    model_dir = sys.argv[1]
    model_prefix = sys.argv[2]
    model = load_model(model_dir, model_prefix)
