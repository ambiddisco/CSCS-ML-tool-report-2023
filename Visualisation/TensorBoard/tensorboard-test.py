import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation 
from keras.utils import np_utils 

# Launching a TensorBoard session can be done with some code editors through the command line
import tensorboard

# To avoid errors, if intending to run the same tests multiple times, old logs should be removed
# unless each one is named differently. That can be done by navigating to the location of this file
# in the terminal, and then running the following command:
# rm -rf ./TensorBoard_test_logs/


# Separating the data into training and testing sets
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# All values are between 0 and 255, dividing them to have values from 0 to 1:
X_train_divided = X_train / 255
X_test_divided = X_test / 255

# Setting a few lists of variables to make varying the model output easy. 
first_layer_nodes = [100, 50, 100, 50, 100, 50]
first_layer_activation = ["relu", "sigmoid", "softmax", "relu", "sigmoid", "softmax"]
second_layer_nodes = [10, 15, 25, 25, 15, 10]
second_layer_activation = ["sigmoid", "softmax", "relu", "relu", "sigmoid", "softmax"]

for i in range(6):
    # Naming the runs and saving them in the same directory, so that we can upload the whole directory later.
    log_dir = "TensorBoard_test_logs/run_{}".format(i+1)
    tensorboard_callback= tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

     # Model creation, compilation and training
    model = Sequential([
        keras.layers.Flatten(input_shape = (28,28), name = 'layers_flatten'),
        keras.layers.Dense(first_layer_nodes[i], first_layer_activation[i], name = 'layers_dense_1'),
        keras.layers.Dense(second_layer_nodes[i], second_layer_activation[i], name = 'layers_dense_2')
    ])
    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    model.fit(
        X_train_divided,
        y_train, 
        epochs = 5, 
        validation_data = (X_test_divided, y_test), 
        callbacks = [tensorboard_callback]
    )

# After creating the experiment and logs, to visualise them run this in the terminal:
# tensorboard --logdir TensorBoard_test_logs/