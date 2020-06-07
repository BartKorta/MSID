# Fashion Mnist - Author: Bartosz Korta
## Date: 07.06.2020

### Introduction
Task to do: Classify fashion-mnist dataset 
Available input data: 60 000 images of clothes with corresponding labels < training data,   
10 000 images and labels < test data
The model which classify test images to corresponding labels, has been taught only with training dataset.
Every image is black and white, with 28x28 pixels in frame. It means that we can consider an image as matrix 28x28 in which every cell corresponds to its color. (From 0 to 255, 0-white, 255-black).
Output: The accuracy of the model. (The percentage of correctly classifed images). Also a plot which illustrates examples of model classification.
The implemented solution is called Convolutional Neural Network.

### Methods
```
model = keras.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        keras.layers.Conv2D(filters=128, kernel_size=4, activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Flatten(input_shape=(14, 14)),
        keras.layers.Dense(312, activation='relu'),
        keras.layers.Dropout(rate=0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(rate=0.3),
        keras.layers.Dense(10, activation='relu')
    ])
```
My model containes a few layers. First three layers are responsible for feature extraction.
Convolution layer detectes shapes and edges from image. How does it work?
Firstly, let's explain parameters.
* filters - number of matrices
* kernel_size - number which represents matrix size, ex. 3 stands for 3x3 matrix
* activaction relu - it means that every number in cell is going to be replaced with max(0, number). Simply every negative number is going to be replaced with 0.

```

