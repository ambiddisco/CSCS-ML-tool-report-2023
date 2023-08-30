# Before running this code make sure you downloaded the requirements ("pip install tensorflow mlflow")
# and run "mlflow server" in a terminal window. This will start the localhost server for visualisation 
# and model repository upload.


# Imports for model creation:
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist     
from keras.models import Sequential  
from keras.layers import Dense, Dropout, Activation 
from keras.utils import np_utils 

# Imports for upload to the MLflow model registry:
import mlflow
from mlflow.models.signature import infer_signature

# This code is for upload of a trained model to the model registry. Since we have run multiple
# visualisations of the 6 model variations in the other scripts, here we will upload only the one with 
# the best accuracy, which is model 1:

# Separating the data into training and testing sets
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# All values are between 0 and 255, dividing them to have values from 0 to 1:
X_train_divided = X_train / 255
X_test_divided = X_test / 255

# Model creation, compilation and training, using the best hyperparameters:
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
model.fit(X_train_divided, y_train, epochs = 5, verbose = 1)
model.evaluate(X_test_divided, y_test)

# Before uploading the model, the URL should e defined. By default, localhost:5000 is used
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.tracking.MlflowClient()

# Creating and naming experiment...
experiment_name = "MNIST model upload"
experiment_id = mlflow.create_experiment(experiment_name)

with mlflow.start_run(experiment_id = experiment_id, run_name = "model-upload") as run:
    # Model upload:
    signature = infer_signature(X_test_divided, model.predict(X_test_divided))
    mlflow.tensorflow.log_model(model, "MNIST_model", registered_model_name = "best_model", signature=signature)
