import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Load a keras dataset
# Mnist: dataset of 70k images of 10 clothe categories
fashion_mnist = tf.keras.datasets.fashion_mnist

# Load data từ model, 1 phần để train, 1 phần để test
# Images: 28x28 numpy arrays, pixel values from 0 to 255
# Labels: array of integers from 0 to 9, index tương ứng với class
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Label chỉ đang được lưu dưới dạng index nên khó nhận biết, nên ta tạo list lưu tên class
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Print the format of the dataset before training the model
# (60000, 28, 28): 60000 images, 28x28 pixels
print(train_images.shape)
# Tương ứng với 60000 image, ta có 60000 label thuộc 1 trong 9 class
print(len(train_labels))
