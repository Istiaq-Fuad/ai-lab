import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)

X = np.random.rand(2000, 10)
y = (np.sum(X, axis=1) > 5).astype(int)


X_train, X_val = X[:1600], X[1600:]
y_train, y_val = y[:1600], y[1600:]


model = models.Sequential(
    [
        layers.Dense(64, activation="relu", input_shape=(10,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)


checkpoint = ModelCheckpoint(
    "best_model.keras", monitor="val_loss", save_best_only=True
)

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1,
)


plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Training vs Validation Loss with Callbacks")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Training Loss", "Validation Loss"])

plt.savefig("training_with_callbacks_plot.png")
plt.close()

print("Training complete.")
print("Best model saved as: best_model.keras")
print("Plot saved as: training_with_callbacks_plot.png")
