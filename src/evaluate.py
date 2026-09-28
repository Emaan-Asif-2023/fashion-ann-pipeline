import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
import os

model = tf.keras.models.load_model("models/model.h5")

x_test = np.load("data/processed/x_test.npy")
y_test = np.load("data/processed/y_test.npy")

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(x_test), axis=1)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.savefig("confusion_matrix.png")
plt.close()

metrics = {
    "test_loss": float(loss),
    "test_accuracy": float(accuracy)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Evaluation completed successfully.")
print(f"Test accuracy: {accuracy:.4f}")