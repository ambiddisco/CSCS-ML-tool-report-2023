# Before starting to log metrics to W&B, you need to start its server from a terminal window.
# Open a terminal and input "wandb server start"

import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import wandb

# The MNIST classifier dataset is a set of images, made up of 28x28 pixels, of value between 0 and 255.
# First of all we need to split the 70'000 images into a training set and a testing one:
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# For simpler use of the data, we want each value to be between 0 and 1.
X_train_divided = X_train / 255
X_test_divided = X_test / 255

# This next codee block is for a simple way to create models with varying accuracy and loss.
first_layer_nodes = [100, 50, 100, 50, 100, 50]
first_layer_activation = ["relu", "sigmoid", "softmax", "relu", "sigmoid", "softmax"]
second_layer_nodes = [10, 15, 25, 25, 15, 10]
second_layer_activation = ["sigmoid", "softmax", "relu", "relu", "sigmoid", "softmax"]

# To log anything to the wandb.ai server, you need to start it from a terminal with "wandb server start", then 
# you also need to login. Either you can log in with "wand.login()" and then navigate to wandb.ai/authorize to 
# copy your API key, so that you can paste it when promted in the terminal, or, if you already have your API key, 
# you can login with "wandb.login(key='your key goes here')". The key here is a string and must go inside quotes.
wandb.login()

for run_number in range(6):
    # Each run logged with wandb should be initialised and then ended to close the connection with the server.
    wandb.init(project = "wandb MNIST test",
               config = { # these are all hyperparameters
                   "architecture": "CNN",
                   "dataset":  "keras MNIST dataset",
                   "first layer nodes": first_layer_nodes[run_number],
                   "first layer activation": first_layer_activation[run_number],
                   "second layer nodes": second_layer_nodes[run_number],
                   "second layer activation": second_layer_activation[run_number],
                   "optimizer": "adam",
                   "loss calculation": "sparse_categorical_crossentropy",
                   "epochs": 5,
               },
               name = "run_{}".format(run_number + 1))
    
    # Create, compile and train the model:
    model = Sequential([
        keras.layers.Flatten(input_shape = (28,28)),
        keras.layers.Dense(first_layer_nodes[run_number], first_layer_activation[run_number]),
        keras.layers.Dense(second_layer_nodes[run_number], second_layer_activation[run_number])
    ])
    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    history = model.fit(X_train_divided, y_train, epochs = 5, verbose = 1)

    # For each run we want to log the accuracy and loss after each epoch.
    for i in range(5):
         wandb.log({"acc": history.history['accuracy'][i], "loss": history.history['loss'][i]})
    
    wandb.finish()
