import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


x_val = x_train[-5000:]
y_val = y_train[-5000:]
x_train = x_train[:-5000]
y_train = y_train[:-5000]


os.makedirs("results", exist_ok=True)


def build_baseline():
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model


def build_dropout():
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model


def build_aug_dropout():
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    model = models.Sequential(
        [
            data_augmentation,
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model


def compile_and_train(model, name):
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    print(f"\nTraining {name} model...\n")

    history = model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=64,
        validation_data=(x_val, y_val),
        verbose=1,
    )

    return history


history_baseline = compile_and_train(build_baseline(), "Baseline")
history_dropout = compile_and_train(build_dropout(), "Dropout")
history_aug = compile_and_train(build_aug_dropout(), "Dropout + Augmentation")


plt.figure()
plt.plot(history_baseline.history["val_accuracy"], label="Baseline")
plt.plot(history_dropout.history["val_accuracy"], label="Dropout")
plt.plot(history_aug.history["val_accuracy"], label="Dropout + Aug")

plt.title("Validation Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.savefig("results/accuracy_comparison.png")
plt.close()


plt.figure()
plt.plot(history_baseline.history["val_loss"], label="Baseline")
plt.plot(history_dropout.history["val_loss"], label="Dropout")
plt.plot(history_aug.history["val_loss"], label="Dropout + Aug")

plt.title("Validation Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.legend()
plt.savefig("results/loss_comparison.png")
plt.close()


print("\nTraining complete!")
print("Plots saved in 'results/' folder.")
