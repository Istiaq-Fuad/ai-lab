import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


os.makedirs("results", exist_ok=True)


np.random.seed(42)
tf.random.set_seed(42)


x = np.linspace(-10, 10, 1000)
x = x.reshape(-1, 1)


def linear(x):
    return 5 * x + 10


def quadratic(x):
    return 3 * x**2 + 5 * x + 10


def cubic(x):
    return 4 * x**3 + 3 * x**2 + 5 * x + 10


experiments = {"linear": linear, "quadratic": quadratic, "cubic": cubic}


for name, func in experiments.items():

    print(f"\n===== Training for {name} equation ======")

    y = func(x)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.3, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(64, activation="relu", input_shape=(1,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="mse",
        metrics=["mae"],
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=32,
        verbose=0,
    )

    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
    print("Test MSE:", test_loss)
    print("Test MAE:", test_mae)

    y_pred = model.predict(x, verbose=0)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label="Original y", s=10)
    plt.plot(x, y_pred, color="red", label="Predicted y", linewidth=2)
    plt.title(f"{name.capitalize()} Equation: Original vs Predicted")
    plt.legend()
    plt.savefig(f"results/{name}_prediction.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{name.capitalize()} Equation: Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(f"results/{name}_loss.png")
    plt.close()

print("\nAll experiment images saved inside 'results/' folder.")
