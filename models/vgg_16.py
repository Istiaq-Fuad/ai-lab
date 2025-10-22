import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical


def build_cnn_model(input_shape=(32, 32, 3), num_classes=10):

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(
                32, (3, 3), activation="relu", padding="same", name="block1_conv1"
            ),
            layers.Conv2D(
                32, (3, 3), activation="relu", padding="same", name="block1_conv2"
            ),
            layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool"),
            layers.Conv2D(
                64, (3, 3), activation="relu", padding="same", name="block2_conv1"
            ),
            layers.Conv2D(
                64, (3, 3), activation="relu", padding="same", name="block2_conv2"
            ),
            layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool"),
            layers.Conv2D(
                128, (3, 3), activation="relu", padding="same", name="block3_conv1"
            ),
            layers.Conv2D(
                128, (3, 3), activation="relu", padding="same", name="block3_conv2"
            ),
            layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool"),
            layers.Flatten(name="flatten"),
            layers.Dense(512, activation="relu", name="fc1"),
            layers.Dropout(0.5, name="dropout1"),
            layers.Dense(num_classes, activation="softmax", name="predictions"),
        ],
        name="simple_cnn_model",
    )

    return model


if __name__ == "__main__":
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    IMG_HEIGHT = 32
    IMG_WIDTH = 32
    IMG_CHANNELS = 3
    NUM_CLASSES = 10

    # --- 3. Build the Model ---
    simple_model = build_cnn_model(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=NUM_CLASSES
    )

    print("Compiling model...")
    simple_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    print("CNN Model Architecture:")
    simple_model.summary()

    print("\nTraining model...")
    history = simple_model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=64,
        validation_data=(x_test, y_test),
    )

    print("\nEvaluating model on the test set:")
    test_loss, test_acc = simple_model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
