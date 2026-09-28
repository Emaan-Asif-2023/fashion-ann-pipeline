import numpy as np
import tensorflow as tf
import json
import os

model = tf.keras.models.load_model("models/model.h5")

x_test = np.load("data/processed/x_test.npy")
y_test = np.load("data/processed/y_test.npy")

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

metrics = {
    "test_loss": float(loss),
    "test_accuracy": float(accuracy)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Evaluation completed successfully.")
print(f"Test accuracy: {accuracy:.4f}")