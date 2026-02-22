import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


def preprocess_data(X, y):
    X = X.astype("float32") / 255.0

    if len(X.shape) == 3:
        X = np.expand_dims(X, -1)

    if X.shape[-1] == 1:
        X = np.repeat(X, 3, axis=-1)

    X = tf.image.resize(X, (32, 32)).numpy()

    if len(y.shape) > 1:
        y = y.flatten()

    return X, y


def build_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def plot_accuracy(history, dataset_name):
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(f"{dataset_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"])
    plt.savefig(f"{RESULT_DIR}/{dataset_name}_accuracy.png")
    plt.close()


def plot_loss(history, dataset_name):
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(f"{dataset_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])
    plt.savefig(f"{RESULT_DIR}/{dataset_name}_loss.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, dataset_name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    tick_marks = np.arange(10)
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center")

    plt.savefig(f"{RESULT_DIR}/{dataset_name}_confusion_matrix.png")
    plt.close()


def train_dataset(dataset_name, loader):
    print(f"\nTraining on {dataset_name}")

    (X_train, y_train), (X_test, y_test) = loader()

    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    model = build_model()

    history = model.fit(
        X_train, y_train, epochs=15, batch_size=64, validation_split=0.1, verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"{dataset_name} Test Accuracy: {test_acc:.4f}")

    plot_accuracy(history, dataset_name)
    plot_loss(history, dataset_name)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    plot_confusion_matrix(y_test, y_pred, dataset_name)


if __name__ == "__main__":
    train_dataset("MNIST", tf.keras.datasets.mnist.load_data)
    train_dataset("Fashion_MNIST", tf.keras.datasets.fashion_mnist.load_data)
    train_dataset("CIFAR10", tf.keras.datasets.cifar10.load_data)

    print("\nAll results saved inside 'results/' folder.")
