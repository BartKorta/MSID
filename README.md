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

So, first Convolution layer has 64 different matrices which finds different shapes in the image. This layer passes analyzed matrices to second Convolution layer which has 128 matrices, which analyze images. This matrices size is bigger (4x4). It finds another shapes on previously examined matrices (images).

<<photo>>
        
After two convolution layers, received matrices go to Pooling Layer.
Pool_size=(2,2) means that we divide our 28x28 matrix on: 196 2x2 matrices. And from every matrix we rewrite only the biggest number, to a new matrix (14x14)
Pooling layer simply "compresses" matrices. It receives 28x28 matrices and and returs 14x14 more dense matrix, with the most relevant pixels. So in the result calculations are faster (less cells to analyze later, reduced complexity of calculations).

<<photo>>
        
Dropout is subsequent layer, which drops out some comnnections in the network. The rate which is settled, corresponds to probabilty of removing a connection between neurons from previous layer to next layer.
Why we do it? To avoid overfitting. This layer forces matrices to find another path to succes. (Because some connections won't be available.) But, you must be aware of setting to high rate, model won't be able to learn properly. (Underfitting)
        
Flatten layer simply flattens our 14x14 matrix to vector which size is equal 196.

