import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def plot_history(history, metric_key, title, ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history[metric_key], label=f"Training {ylabel}", marker="o")
    plt.plot(
        history.history[f"val_{metric_key}"], label=f"Validation {ylabel}", marker="s"
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_PATH = Path("pen-pencil-classifier/processed_dataset")

classes = ["pen", "pencil"]
for class_name in classes:
    class_path = DATASET_PATH / class_name
    num_images = len(list(class_path.glob("*.jpg")))
    print(f"{class_name.capitalize()}: {num_images} images")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Sample Images from Dataset", fontsize=16, fontweight="bold")

for i, class_name in enumerate(classes):
    class_path = DATASET_PATH / class_name
    image_files = list(class_path.glob("*.jpg"))[:5]

    for j, img_path in enumerate(image_files):
        img = load_img(img_path, target_size=(224, 224))
        axes[i, j].imshow(img)
        axes[i, j].axis("off")
        axes[i, j].set_title(f"{class_name}", fontsize=10)

plt.tight_layout()
plt.show()

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

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

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    str(DATASET_PATH),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
    seed=SEED,
)

validation_generator = train_datagen.flow_from_directory(
    str(DATASET_PATH),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
    seed=SEED,
)

test_generator = test_datagen.flow_from_directory(
    str(DATASET_PATH),
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode="binary",
    shuffle=False,
)

print(f"\nClass indices: {train_generator.class_indices}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Test samples: {test_generator.samples}")

model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(*IMG_SIZE, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

print("\nCustom CNN Model Summary:")
model.summary()

checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)

early_stop = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
)

callbacks = [checkpoint, early_stop, reduce_lr]

print("Starting training...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

print("\nTraining completed!")

plot_history(history, "accuracy", "Model Accuracy Over Epochs", "Accuracy")
plot_history(history, "loss", "Model Loss Over Epochs", "Loss")
plot_history(history, "auc", "Model AUC Over Epochs", "AUC")

test_loss, test_accuracy, test_auc = model.evaluate(test_generator, verbose=1)

print(f"Test Results:\n")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

test_generator.reset()

y_pred_proba = model.predict(test_generator, verbose=1)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()
y_true = test_generator.classes

class_names = list(test_generator.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={"label": "Count"},
    annot_kws={"size": 16},
)
plt.title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
plt.ylabel("True Label", fontsize=12)
plt.xlabel("Predicted Label", fontsize=12)
plt.tight_layout()
plt.show()

print(f"\n\nTrue Negatives (Pen predicted as Pen): {cm[0][0]}")
print(f"False Positives (Pen predicted as Pencil): {cm[0][1]}")
print(f"False Negatives (Pencil predicted as Pen): {cm[1][0]}")
print(f"True Positives (Pencil predicted as Pencil): {cm[1][1]}")

precision = precision_score(y_true, y_pred, average="binary")
recall = recall_score(y_true, y_pred, average="binary")
f1 = f1_score(y_true, y_pred, average="binary")

metrics = ["Precision", "Recall", "F1-Score", "Accuracy"]
values = [precision, recall, f1, test_accuracy]

plt.figure(figsize=(10, 6))
bars = plt.bar(
    metrics,
    values,
    color=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"],
    alpha=0.8,
    edgecolor="black",
)
plt.ylim(0, 1.1)
plt.ylabel("Score", fontsize=12)
plt.title("Model Performance Metrics", fontsize=16, fontweight="bold", pad=20)
plt.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, values):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.02,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=11,
    )

plt.tight_layout()
plt.show()

num_samples = 16
indices = np.random.choice(len(test_generator.filenames), num_samples, replace=False)

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
fig.suptitle(
    "Sample Predictions with Confidence Scores", fontsize=18, fontweight="bold", y=0.995
)

for idx, ax in zip(indices, axes.flatten()):
    img_path = DATASET_PATH / test_generator.filenames[idx]
    img = load_img(img_path, target_size=IMG_SIZE)

    true_label = class_names[y_true[idx]]
    pred_label = class_names[y_pred[idx]]
    confidence = y_pred_proba[idx][0] if y_pred[idx] == 1 else 1 - y_pred_proba[idx][0]

    ax.imshow(img)
    ax.axis("off")

    color = "green" if true_label == pred_label else "red"
    title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}"
    ax.set_title(title, fontsize=10, color=color, fontweight="bold")

plt.tight_layout()
plt.show()
