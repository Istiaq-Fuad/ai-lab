import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


X = np.linspace(-10, 10, 500).reshape(-1, 1)
y = 5 * X**2 + 10 * X - 2


model = keras.Sequential(
    [
        layers.Dense(32, activation="relu", input_shape=(1,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ]
)

model.compile(optimizer="adam", loss="mse")


history = model.fit(X, y, epochs=100, verbose=0)


plt.plot(history.history["loss"])
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.savefig("loss_plot.png")
plt.close()


y_pred = model.predict(X)

plt.plot(X, y, label="Original f(x)")
plt.plot(X, y_pred, label="Predicted f(x)", linestyle="dashed")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Original vs Predicted Function")
plt.legend()
plt.savefig("fx_comparison.png")
plt.close()
