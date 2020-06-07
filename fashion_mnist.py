# TensorFlow and tf.keras and mtploid
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#call this function if you want to restore your model's weights
def restore_model(test_images, test_labels):
    model = create_model()
    model.load_weights('fashion_mnist_results3/cp.ckpt')
    get_results(model,test_images,test_labels)
    model.summary()
    pass

#call this function if you want to learn a new model
def new_model(train_images, train_labels, test_images, test_labels):
    model = create_model()
    #checkpoints needed to save your results
    checkpoint_path = "new_model_weigths/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    #
    model.fit(train_images,
              train_labels,
              epochs=10,
              batch_size=500,
              callbacks=[cp_callback])  # Pass callback to training

    #display the summary
    model.summary()
    get_results(model,test_images,test_labels)
    pass

def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        keras.layers.Conv2D(filters=128, kernel_size=4, activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Flatten(input_shape=(14, 14)),
        keras.layers.Dense(312, activation='relu'),
        keras.layers.Dropout(rate=0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(rate=0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='relu')
    ])
    #compile the model with nadam optimizer
    model.compile(optimizer='nadam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
    )
    return model
    pass

def get_results(model, test_images, test_labels):
    #test your model with new data
    loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
    #Display the accuracy
    print("Accuracy: {:5.2f}%".format(100*acc))
    print("Test data size: ")
    print(test_images.shape[0])
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)

    #draw a plot
    test_images=test_images.reshape(10000,28,28)
    num_rows = 5
    num_cols = 4
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()
    pass

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],100*np.max(predictions_array),class_names[true_label]),color=color)
    pass

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    pass

if __name__ == "__main__":
    #loading photos by keras
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images=train_images.reshape(train_images.shape[0],28,28,1)
    test_images=test_images.reshape(test_images.shape[0],28,28,1)
    #Scaling values to a ronge of 0,1
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    restore_model(test_images,test_labels)
    #new_model(train_images, train_labels, test_images, test_labels)
