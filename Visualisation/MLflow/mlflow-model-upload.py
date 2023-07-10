import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist     
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation 
from keras.utils import np_utils 
import mlflow

from mlflow.models.signature import infer_signature

# This code is for upload of a trained model to the model registry. Since we have run multiple
# visualisations of the 6 model variations in the other scripts, here we will upload only the one
# that has the best accuracy, which is model 1:

# Separating the data into training and testing sets
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# All values are between 0 and 255, dividing them to have values from 0 to 1:
X_train_divided = X_train / 255
X_test_divided = X_test / 255

# Model creation, compilation and training, using the best hyperparameters
model = Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(100, 'relu'),
    keras.layers.Dense(10, 'sigmoid')
])
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
history = model.fit(X_train_divided, y_train, epochs = 5, verbose = 1)
score = model.evaluate(X_test_divided, y_test)

# Model upload
signature = infer_signature(X_test_divided, model.predict(X_test_divided))
mlflow.tensorflow.log_model(model, "MNIST_best_model", signature=signature)
