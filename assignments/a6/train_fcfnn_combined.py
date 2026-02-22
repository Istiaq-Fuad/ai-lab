import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical


HIDDEN_LAYERS = [256, 64, 128]


def load_handwritten_digits(folder_path):
    images = []
    labels = []

    for digit in range(10):
        digit_folder = os.path.join(folder_path, str(digit))
        if not os.path.isdir(digit_folder):
            continue

        for file_name in os.listdir(digit_folder):
            file_path = os.path.join(digit_folder, file_name)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            if image.shape != (28, 28):
                image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

            images.append(image)
            labels.append(digit)

    if len(images) == 0:
        raise ValueError(
            f"No handwritten digit images were found in '{folder_path}'. "
            "Run make_dataset.py first to generate extracted digits."
        )

    x = np.array(images, dtype="float32") / 255.0
    y = np.array(labels, dtype="int32")
    return x, y


def build_fcfnn(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())

    for units in HIDDEN_LAYERS:
        model.add(layers.Dense(units, activation="relu"))

    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def plot_accuracy(history, save_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("FCFNN Accuracy (MNIST + Handwritten)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "combined_accuracy.png"))
    plt.close()


def plot_loss(history, save_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("FCFNN Loss (MNIST + Handwritten)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "combined_loss.png"))
    plt.close()


def plot_confusion(y_true, y_pred, save_dir):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title("Confusion Matrix (MNIST + Handwritten Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "combined_confusion_matrix.png"))
    plt.close()


def save_dataset_summary(
    save_dir, hw_train_count, hw_test_count, mnist_train_count, mnist_test_count
):
    with open(os.path.join(save_dir, "dataset_summary.txt"), "w") as f:
        f.write(f"Handwritten train samples: {hw_train_count}\n")
        f.write(f"Handwritten test samples: {hw_test_count}\n")
        f.write(f"MNIST train samples: {mnist_train_count}\n")
        f.write(f"MNIST test samples: {mnist_test_count}\n")
        f.write(f"Combined train samples: {hw_train_count + mnist_train_count}\n")
        f.write(f"Combined test samples: {hw_test_count + mnist_test_count}\n")


def main(handwritten_folder, output_dir, test_size, random_state, epochs, batch_size):
    os.makedirs(output_dir, exist_ok=True)

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    x_hw, y_hw = load_handwritten_digits(handwritten_folder)

    x_hw_train, x_hw_test, y_hw_train, y_hw_test = train_test_split(
        x_hw,
        y_hw,
        test_size=test_size,
        random_state=random_state,
        stratify=y_hw,
    )

    (x_mnist_train, y_mnist_train), (x_mnist_test, y_mnist_test) = (
        tf.keras.datasets.mnist.load_data()
    )
    x_mnist_train = x_mnist_train.astype("float32") / 255.0
    x_mnist_test = x_mnist_test.astype("float32") / 255.0

    x_train = np.concatenate([x_mnist_train, x_hw_train], axis=0)
    y_train = np.concatenate([y_mnist_train, y_hw_train], axis=0)

    x_test = np.concatenate([x_mnist_test, x_hw_test], axis=0)
    y_test = np.concatenate([y_mnist_test, y_hw_test], axis=0)

    train_perm = np.random.permutation(x_train.shape[0])
    test_perm = np.random.permutation(x_test.shape[0])
    x_train, y_train = x_train[train_perm], y_train[train_perm]
    x_test, y_test = x_test[test_perm], y_test[test_perm]

    num_classes = 10
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    model = build_fcfnn(x_train.shape[1:], num_classes)

    history = model.fit(
        x_train,
        y_train_cat,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Combined Test Accuracy: {test_accuracy:.4f}")
    print(f"Combined Test Loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    plot_accuracy(history, output_dir)
    plot_loss(history, output_dir)
    plot_confusion(y_test, y_pred, output_dir)

    report = classification_report(y_test, y_pred)
    with open(os.path.join(output_dir, "combined_classification_report.txt"), "w") as f:
        f.write(report)

    save_dataset_summary(
        output_dir,
        hw_train_count=len(x_hw_train),
        hw_test_count=len(x_hw_test),
        mnist_train_count=len(x_mnist_train),
        mnist_test_count=len(x_mnist_test),
    )

    model.save(os.path.join(output_dir, "fcfnn_combined_model.keras"))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    HANDWRITTEN_FOLDER = "extracted_digits"
    OUTPUT_DIR = "results"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    EPOCHS = 10
    BATCH_SIZE = 128

    main(
        handwritten_folder=HANDWRITTEN_FOLDER,
        output_dir=OUTPUT_DIR,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
