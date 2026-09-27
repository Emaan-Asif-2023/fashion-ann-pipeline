import tensorflow as tf
import os

os.makedirs("data/raw", exist_ok=True)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

import numpy as np

np.save("data/raw/x_train.npy", x_train)
np.save("data/raw/y_train.npy", y_train)
np.save("data/raw/x_test.npy", x_test)
np.save("data/raw/y_test.npy", y_test)

print("Fashion-MNIST raw data saved successfully.")