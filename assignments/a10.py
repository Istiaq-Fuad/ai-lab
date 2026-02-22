# !git clone https://github.com/Istiaq-Fuad/pen-pencil-classifier.git

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


dataset_path = Path("pen-pencil-classifier/processed_dataset")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-5
SEED = 42


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    str(dataset_path),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
    seed=SEED,
)

validation_generator = train_datagen.flow_from_directory(
    str(dataset_path),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
    seed=SEED,
)

print("\nClass indices:", train_generator.class_indices)


def build_model(fine_tune_option="partial"):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    if fine_tune_option == "full":
        print("\nFine-tuning FULL VGG16...")
        for layer in base_model.layers:
            layer.trainable = True

    elif fine_tune_option == "partial":
        print("\nFine-tuning PARTIAL VGG16 (last 4 layers)...")
        for layer in base_model.layers[-4:]:
            layer.trainable = True

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(option):
    model = build_model(option)

    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=1,
    )

    return history


history_full = train_model("full")
history_partial = train_model("partial")


def save_plot(history, title, filename):

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"])

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


save_plot(history_full, "Full Fine-Tuning", "full_finetune.png")
save_plot(history_partial, "Partial Fine-Tuning", "partial_finetune.png")

print("\nTraining complete.")
print("Saved: full_finetune.png")
print("Saved: partial_finetune.png")
