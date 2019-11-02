# cats-and-dogs

Cats and dogs is a Convolutional Neural Network for classifying cats and dog pictures.

## Problem

Given an image, output a scalar value indicating wether the image contains a cat or a dog.

Data set is available at <https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/P16-Convolutional-Neural-Networks.zip>

```shell
curl -o download.zip https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/P16-Convolutional-Neural-Networks.zip
unzip -o download.zip
mv ./Convolutional_Neural_Networks/dataset/ cats_and_dogs/
```

## Solution

Solution is based on a Convolutional Neural Network.

1. `classifier_v1.py`:

    * Conv 2D layer with 3x3 kernel size, stride 2 and `ReLU` activation.
    * 2D Max pooling layer and a flatten step.
    * 1 Dense hidden layer with 128 units and `ReLU` activation.
    * 1 output with `Sigmoid` activation.
