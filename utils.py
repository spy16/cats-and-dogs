import os


def save_model(model, _dir, prefix='model'):
    os.mkdir(_dir)

    desc_file = os.path.join(_dir,  "{}_{}.json".format(prefix, "spec"))
    weights_file = os.path.join(_dir, "{}_{}.h5".format(prefix, "weights"))

    model_json = model.to_json()
    with open(desc_file, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weights_file)