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

# For each example the model returns a vector of "logits" or "log-odds" scores, one for each class.
# logit: raw (non-normalized) predictions that a classification model generates, then passed to a normalization function
# log-odds: the logarithm of the odds of some event.
# odds: refers to the ratio of the probability of success (p) to the probability of failure (1-p)
predictions = model(x_train[:1]).numpy()
print(predictions)

# The tf.nn.softmax (normalization function) converts these logits to "probabilities" for each class
tf.nn.softmax(predictions).numpy()
# It is possible to bake the tf.nn.softmax function into the activation function for the last layer of the network
# but not recommended: provide an exact and numerically stable loss calculation for all models when using a softmax output

# The loss function takes a vector of ground truth values and a vector of logits and returns a scalar loss for each example.
# This untrained model gives probabilities close to random
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

# Adam là một thuật toán tối ưu hóa được sử dụng phổ biến trong huấn luyện mạng neural. Adam là viết tắt của "Adaptive Moment Estimation". Thuật toán này kết hợp cả gradient descent với momentum và RMSprop để cập nhật trọng số của mạng neural.
# Before you start training, configure and compile the model
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

#  train neural network model
model.fit(x_train, y_train, epochs=5)

# check the model's performance, usually on a validation set or test set.
# verbose: This parameter controls the verbosity mode, which determines how much information is displayed during evaluation. It can take on three possible values:
# 0: silent, 1: progress bar, 2: Detailed information
model.evaluate(x_test, y_test, verbose=2)

# Another model to return a probability, by wrap the trained model, and attach the softmax to it
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
print(probability_model(x_test[:5]))
