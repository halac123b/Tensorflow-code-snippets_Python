import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

print(tf.__version__)

# Load and prepare the MNIST dataset (database of handwritten digits)
mnist = tf.keras.datasets.mnist

# The pixel values of the images range from 0 through 255.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize the input image so that each pixel value is between 0 and 1.
x_train, x_test = x_train / 255.0, x_test / 255.0

# A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
model = tf.keras.models.Sequential(
    [
        # Flatten layer: used to convert multidimensional data into a one-dimensional array
        # It is commonly used as the first layer in a neural network model when dealing with input data that has multiple dimensions, such as images.
        # input_shape: The shape of the input data (images with dimensions of 28px by 28px)
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # Fully connected layer
        # 128: number of neurons
        # relu: activation function: f(x) = max(0, x)
        tf.keras.layers.Dense(128, activation="relu"),
        # Dropout layer: used to prevent overfitting
        # 0.2: fraction of the input units to drop, 20% of the input units will be randomly set to zero during each training iteration
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ]
)
