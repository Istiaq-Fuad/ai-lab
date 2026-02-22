# !git clone https://github.com/Istiaq-Fuad/pen-pencil-classifier.git

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

data_dir = "pen-pencil-classifier/processed_dataset"

results = {}


def create_model():
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


baseline_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_gen = baseline_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
)

val_gen = baseline_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
)

print("Training baseline model...")
model = create_model()
history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)

results["Baseline"] = history.history["val_accuracy"]

plt.figure()
plt.plot(history.history["val_accuracy"])
plt.title("Baseline")
plt.savefig("baseline.png")
plt.close()


augmentations = {
    "Rotation": dict(rotation_range=20),
    "Shift": dict(width_shift_range=0.1, height_shift_range=0.1),
    "Flip": dict(horizontal_flip=True),
    "Zoom": dict(zoom_range=0.1),
    "Combined": dict(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    ),
}


for name, aug_params in augmentations.items():

    print(f"\nTraining with {name} augmentation...")

    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2, **aug_params)

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
    )

    model = create_model()
    history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)

    results[name] = history.history["val_accuracy"]

    plt.figure()
    plt.plot(history.history["val_accuracy"])
    plt.title(name)
    plt.savefig(f"{name.lower()}.png")
    plt.close()


plt.figure()
for name, acc in results.items():
    plt.plot(acc, label=name)

plt.title("Effect of Data Augmentation on Pen-Pencil CNN")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.savefig("augmentation_comparison.png")
plt.show()

print("\nFinal Validation Accuracies:")
for name in results:
    print(f"{name}: {results[name][-1]:.4f}")
