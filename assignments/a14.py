import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


os.makedirs("assets", exist_ok=True)


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)


def build_model(activation_fn, loss_fn):
    model = keras.Sequential(
        [
            layers.Dense(128, activation=activation_fn, input_shape=(784,)),
            layers.Dense(64, activation=activation_fn),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    return model


activation_functions = ["relu", "tanh", "sigmoid"]
loss_functions = ["categorical_crossentropy", "mean_squared_error"]

results = []

for activation in activation_functions:
    for loss in loss_functions:
        print(f"\nTraining with Activation: {activation} | Loss: {loss}")

        model = build_model(activation, loss)

        model.fit(
            x_train,
            y_train_cat,
            epochs=5,
            batch_size=128,
            validation_split=0.2,
            verbose=0,
        )

        test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)

        results.append(
            {
                "Activation": activation,
                "Loss": loss,
                "Test Accuracy": test_acc,
                "Test Loss": test_loss,
            }
        )

results_df = pd.DataFrame(results)
print("\nFinal Results:")
print(results_df)


plt.figure()

for loss in loss_functions:
    subset = results_df[results_df["Loss"] == loss]
    plt.plot(subset["Activation"], subset["Test Accuracy"], marker="o", label=loss)

plt.xlabel("Activation Function")
plt.ylabel("Test Accuracy")
plt.title("Effect of Activation & Loss on Classifier Performance")
plt.legend()
plt.tight_layout()
plt.savefig("assets/combined_comparison.png", dpi=300)
plt.close()


for loss in loss_functions:
    plt.figure()
    subset = results_df[results_df["Loss"] == loss]
    plt.plot(subset["Activation"], subset["Test Accuracy"], marker="o")
    plt.xlabel("Activation Function")
    plt.ylabel("Test Accuracy")
    plt.title(f"Effect of Activation ({loss})")
    plt.tight_layout()
    plt.savefig(f"assets/{loss}_comparison.png", dpi=300)
    plt.close()

print("\nAll plots saved inside 'assets/' folder.")
