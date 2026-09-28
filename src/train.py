import numpy as np
import tensorflow as tf
import os
import yaml
import pandas as pd

os.makedirs("models", exist_ok=True)

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

x_train = np.load("data/processed/x_train.npy")
y_train = np.load("data/processed/y_train.npy")
x_val = np.load("data/processed/x_val.npy")
y_val = np.load("data/processed/y_val.npy")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(params["model"]["hidden_units"], activation="relu"),
    tf.keras.layers.Dropout(params["model"]["dropout"]),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=params["training"]["epochs"],
    batch_size=params["training"]["batch_size"]
)

model.save("models/model.h5")

pd.DataFrame(history.history).to_csv("models/history.csv", index=False)

print("Model training completed successfully.")