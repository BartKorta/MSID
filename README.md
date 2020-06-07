# Fashion Mnist - Author: Bartosz Korta
## Date: 07.06.2020

### Introduction
Classification of fashion-mnist dataset 

Available input data: 60 000 images of clothes with corresponding labels < training data,   

10 000 images of clothes and labels < test data

The model which classify test images to corresponding labels, has been taught only with training dataset.

Every image is black and white, with 28x28 pixels in frame. It means that we can consider an image as matrix 28x28 in which every cell corresponds to its color. (From 0 to 255, 0-white, 255-black).

Output: The accuracy of the model. (The percentage of correctly classifed images). Also a plot which illustrates examples of how model predicts what an image contains.
The solution that I've used is called Convolutional Neural Network.

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
Convolution layer detectes shapes and edges from images. How does it work?
Firstly, let's explain parameters.
* filters - number of matrices
* kernel_size - number which represents matrix size, ex. 3 stands for 3x3 matrix
* activaction relu - it means that every number in cell is going to be replaced with max(0, number). Simply every negative number is going to be replaced with 0.

So, first Convolution layer has 64 different matrices which finds different shapes and edges in the image. This layer passes analyzed matrices to second Convolution layer which has 128 matrices, which analyze images. This matrices size is bigger (4x4). It finds another shapes on previously examined matrices (images). Also padding is enable, so our matrix size will be always equal 28x28.
Here is an example of how convolution layer works:

<img src="https://github.com/BartKorta/MSID/blob/master/images/conv.png">
        
After two convolution layers, received matrices go to Pooling Layer.
Pool_size=(2,2) means that we divide our 28x28 matrix on: 196 2x2 matrices. And from every matrix we rewrite only the biggest number, to a new matrix (14x14)
Pooling layer simply "compresses" matrices. It receives 28x28 matrices and and returs 14x14 more dense matrix, with the most relevant pixels. So in the result calculations are faster (less cells to analyze later, reduced complexity of calculations).
Example, below:

<img src="https://github.com/BartKorta/MSID/blob/master/images/pooling.png">
        
Dropout is a subsequent layer, which drops out some connnections in the network. The rate which is settled, corresponds to probabilty of removing a connection between neurons from the previous layer and the next layer.
Why we do it? To avoid overfitting. This layer forces matrices to find another path to succes. (Not always the same for identical images. Because some connections won't be available.) But, you must be aware of setting to high rate. Model won't be able to learn properly. (Underfitting)
        
Flatten layer simply flattens our 14x14 matrix to vector which size is equal 196. That layer starts classification part.

Dense layer is just a typical layer in neural network. It connects neurons from its prevoius layer. Between layer dense layer no.1 and no.2  I've also implemented dropout layer to avoid overfitting.
The last Dense layer has size equals 10. It is our output layer. The output layer must be exactly the same size as the amount of different lables.

I've set batch size=500 and epochs=10 which means that every 500 processed images the model will try to minimaize a classification error, and optimize weights to classify correctly. 10 epochs means that it will go through our dataset 10 times, so it should be able to notice more relevent details every epoch.

### Results
<img src="https://github.com/BartKorta/MSID/blob/master/images/acc.png">
The accuracy of the model is very satisfying. > 92.57% correctly clasiffied images.
<img src="https://github.com/BartKorta/MSID/blob/master/images/res.png">
As you can see on attached screenshot (similar methods implemented), my model seems to be more precise.
An average time of model's learning was about 45 minutes. (4-5 minutes per one epoch).
<img src="https://github.com/BartKorta/MSID/blob/master/images/resPlot.png">

### Usage
In order to run and compile the code you have to install Pyhton3 and the following libraries: tensorflow, keras, matploid and numpy.
Also I've implemented two methods which run the program:
* restore_model() - this method reads optimal weights of the model. Later it makes prediction for the test dataset, based on the loaded weights.
* new_model() - you can run this function to learn a model again, for example if you wnat to verify my solution.

Both methods are called in the bottom section of the code. Remeber to run only one in the same time! (Comment a line you won't use!)
Example:
```
restore_model(test_images,test_labels)
#new_model(train_images, train_labels, test_images, test_labels)
```
If you install keras and tensorflow then dataset (training images, labels and test images,labels) loads automatically.
Also to reload optimal weigths you must put a "fashion-mnist-results3" directory (it contains files which are neccesary to restore my model, you can find a link to google drive in my github repo, over 25MB so I counldn't uploaded on github) with the code in the same directory.

Remeber to download fashion_mnist.py file which containes code and fashion_mnist_result3 directory with model's weights<- LINK TO IT IN "WEIGHTS - GOOGLE DRIVE LINK" ON THIS GITHUB REPOSITORY.

I also recommend running the code as an administrator to avoid "Permission denied" Error.
