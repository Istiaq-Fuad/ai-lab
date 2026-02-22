import tensorflow as tf
from tensorflow.keras import layers, models


model = models.Sequential(
    [
        layers.Input(shape=(5,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ]
)


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


model.summary()
