# Before running this code make sure you downloaded the requirements ("pip install tensorflow mlflow")
# and run "mlflow server" in a terminal window. This will start the localhost server for visualisation and
# model repository upload.


import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
import mlflow
from mlflow import log_params

# Separating the MNIST classifier dataset into training and testing sets
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# All values are between 0 and 255, dividing them to have values from 0 to 1:
X_train_divided = X_train / 255
X_test_divided = X_test / 255

# Setting a few lists of variables to make varying the model output easy. 
first_layer_nodes = [100, 50, 100, 50, 100, 50]
first_layer_activation = ["relu", "sigmoid", "softmax", "relu", "sigmoid", "softmax"]
second_layer_nodes = [10, 15, 25, 25, 15, 10]
second_layer_activation = ["sigmoid", "softmax", "relu", "relu", "sigmoid", "softmax"]

# Setting up client and localhost. Remember to run "mlflow server" in a terminal.
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.tracking.MlflowClient()

# Creating and naming experiment...
experiment_name = "MLflow MNIST test"
experiment_id = mlflow.create_experiment(experiment_name)

for i in range(6):
    with mlflow.start_run(experiment_id = experiment_id) as run:
        log_params({ # Logging hyperparameters
            "architecture": "CNN",
            "dataset":  "keras MNIST dataset",
            "first layer nodes": first_layer_nodes[i],
            "first layer activation": first_layer_activation[i],
            "second layer nodes": second_layer_nodes[i],
            "second layer activation": second_layer_activation[i],
            "optimizer": "adam",
            "loss calculation": "sparse_categorical_crossentropy",
            "epochs": 5,
        })

        # Autologging, yay!
        mlflow.tensorflow.autolog()

        # Model creation, compilation and training.
        model = Sequential([
            keras.layers.Flatten(input_shape = (28,28)),
            keras.layers.Dense(first_layer_nodes[i], first_layer_activation[i]),
            keras.layers.Dense(second_layer_nodes[i], second_layer_activation[i])
        ])
        model.compile(
            optimizer = 'adam',
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
        )
        model.fit(X_train_divided, y_train, epochs = 5, verbose = 1)
        model.evaluate(X_test_divided, y_test)

# Once you're done, close the server with CTRL + C in the terminal