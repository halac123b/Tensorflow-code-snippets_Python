import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Load a keras dataset
# Mnist: dataset of 70k images of 10 clothe categories
fashion_mnist = tf.keras.datasets.fashion_mnist
