# Fashion Mnist - Author: Bartosz Korta
## Date: 07.06.2020

### Introduction
Classification of fashion-mnist dataset 
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
My sequential model containes a few layers. First three layers are responsible for feature extraction.
Convolution layer detectes shapes and edges from image. How does it work?
Firstly, let's explain parameters.
* filters - number of matrices
* kernel_size - number which represents matrix size, ex. 3 stands for 3x3 matrix
* activaction relu - it means that every number in cell is going to be replaced with max(0, number). Simply every negative number is going to be replaced with 0.

So, first Convolution layer has 64 different matrices which finds different shapes in the image. This layer passes analyzed matrices to second Convolution layer which has 128 matrices, which analyze images. This matrices size is bigger (4x4). It finds another shapes on previously examined matrices (images). Also padding is enable so our matrix size will be always equal 28x28.

https://github.com/BartKorta/MSID/blob/master/images/conv.png
        
After two convolution layers, received matrices go to Pooling Layer.
Pool_size=(2,2) means that we divide our 28x28 matrix on: 196 2x2 matrices. And from every matrix we rewrite only the biggest number, to a new matrix (14x14)
Pooling layer simply "compresses" matrices. It receives 28x28 matrices and and returs 14x14 more dense matrix, with the most relevant pixels. So in the result calculations are faster (less cells to analyze later, reduced complexity of calculations).

<<photo>>
        
Dropout is subsequent layer, which drops out some comnnections in the network. The rate which is settled, corresponds to probabilty of removing a connection between neurons from previous layer to next layer.
Why we do it? To avoid overfitting. This layer forces matrices to find another path to succes. (Because some connections won't be available.) But, you must be aware of setting to high rate, model won't be able to learn properly. (Underfitting)
        
Flatten layer simply flattens our 14x14 matrix to vector which size is equal 196. That layer starts blocks of "clasiffication layers"/

Dense layer is just a typical layer in neural network. It connects neurons from its prevoius layer. Between layer dense layer no.2 and no.3  I've also implemented dropout layer to avoid overfitting.
The last Dense layer has size equals 10. It is our output layer. The output layer must be exactly the same size as amount of differnet lables.

### Results
The accuracy of the model is very satisfying. > 92.57% correctly clasiffied images.
<<photo>>
As you can see on attached screenshot (similar methods implemented), my model seems to be more precise.
An average time of learning was about 45 minutes. (4-5 minutes per one epoch).
### Usage
In order to run and compile the code you have to install Pyhton3 and the following libraries: tensorflow, keras, matploid and numpy.
Also I've implemented to methods which run the program:
* restore_model() - this method reads optimal weights of the model. Later it makes prediction for the test dataset, based on the loaded weights.
* new_model() - you can run this function to learn again the model, for example if you wnat to verify my solution.

Boths methods are called in the bottom section of the code. Remeber to run only one in the same time! (Comment a line you won't use.)
Example:
```
restore_model(test_images,test_labels)
#new_model(train_images, train_labels, test_images, test_labels)
```
If you install keras and tensorflow then dataset (training images, labels and test images,labels) loads automatically.
Also to reload optimal weigths you must put a "fashion-mnist-results3" directory (you can find it in this github repo) with the code in the samo directory.
