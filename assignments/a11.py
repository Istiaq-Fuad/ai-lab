import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os, gc


IMG_SIZE = 96
BATCH_SIZE = 64
SAMPLE_SIZE = 500
EPOCHS = 1
SAVE_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


def preprocess_image_and_label(image, label):

    image = tf.image.resize(image[..., None], (IMG_SIZE, IMG_SIZE))
    image = tf.image.grayscale_to_rgb(image)
    image = image / 255.0
    return image, label


train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(10000)
    .map(preprocess_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
test_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .map(preprocess_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)


def extract_features(model, dataset, sample_size):
    features = []
    labels = []
    count = 0

    for images, lbls in dataset:
        feats = model(images, training=False)
        features.append(feats.numpy())
        labels.append(lbls.numpy())
        count += images.shape[0]
        if count >= sample_size:
            break

    return np.vstack(features)[:sample_size], np.hstack(labels)[:sample_size]


features_before, labels = extract_features(base_model, test_ds, SAMPLE_SIZE)


base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds)


feature_extractor_after = tf.keras.Model(inputs=model.input, outputs=x)

features_after, _ = extract_features(feature_extractor_after, test_ds, SAMPLE_SIZE)


tf.keras.backend.clear_session()
gc.collect()


def reduce_and_plot(features, labels, title, filename):

    pca_50 = PCA(n_components=50)
    features_50 = pca_50.fit_transform(features)

    pca_2 = PCA(n_components=2)
    reduced_pca = pca_2.fit_transform(features)

    plt.figure()
    plt.scatter(reduced_pca[:, 0], reduced_pca[:, 1], c=labels, s=10)
    plt.title(f"PCA - {title}")
    plt.savefig(f"{SAVE_DIR}/pca_{filename}.png")
    plt.show()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_tsne = tsne.fit_transform(features_50)

    plt.figure()
    plt.scatter(reduced_tsne[:, 0], reduced_tsne[:, 1], c=labels, s=10)
    plt.title(f"t-SNE - {title}")
    plt.savefig(f"{SAVE_DIR}/tsne_{filename}.png")
    plt.show()


reduce_and_plot(features_before, labels, "Before Transfer Learning", "before")
reduce_and_plot(features_after, labels, "After Transfer Learning", "after")

print("Plots saved in:", SAVE_DIR)
