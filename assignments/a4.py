import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report


EPOCHS = 20
BATCH_SIZE = 128
HIDDEN_LAYERS = [512, 256, 128]
SAVE_DIR = "assets"

os.makedirs(SAVE_DIR, exist_ok=True)


def build_fcfnn(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))

    for units in HIDDEN_LAYERS:
        model.add(layers.Dense(units, activation="relu"))

    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def plot_history(history, dataset_name):
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{dataset_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/{dataset_name}_accuracy.png")
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{dataset_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/{dataset_name}_loss.png")
    plt.close()


def plot_confusion(y_true, y_pred, dataset_name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{SAVE_DIR}/{dataset_name}_confusion_matrix.png")
    plt.close()


def run_experiment(dataset_name, load_function):
    print(f"\nRunning experiment on {dataset_name}")

    (x_train, y_train), (x_test, y_test) = load_function()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    num_classes = len(np.unique(y_train))

    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    model = build_fcfnn(x_train.shape[1:], num_classes)

    history = model.fit(
        x_train,
        y_train_cat,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)

    print(f"{dataset_name} Test Accuracy: {test_acc:.4f}")

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    plot_history(history, dataset_name)
    plot_confusion(y_test, y_pred_classes, dataset_name)

    report = classification_report(y_test, y_pred_classes)

    with open(f"{SAVE_DIR}/{dataset_name}_classification_report.txt", "w") as f:
        f.write(report)

    return test_acc


if __name__ == "__main__":

    results = {}

    results["FashionMNIST"] = run_experiment(
        "FashionMNIST", tf.keras.datasets.fashion_mnist.load_data
    )

    results["MNIST"] = run_experiment("MNIST", tf.keras.datasets.mnist.load_data)

    def load_cifar():
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        return (x_train, y_train), (x_test, y_test)

    results["CIFAR10"] = run_experiment("CIFAR10", load_cifar)

    with open(f"{SAVE_DIR}/summary_results.txt", "w") as f:
        for dataset, acc in results.items():
            f.write(f"{dataset}: {acc:.4f}\n")

    print("\nAll experiments completed!")
    print("Results saved in 'assets/' directory.")
