import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobile_preprocess,
)
from tensorflow.keras.preprocessing import image
import cv2


img_path = "pen_0063.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)


def build_grid(feature_map, max_channels=16):
    fmap = feature_map[0]
    num_channels = min(max_channels, fmap.shape[-1])
    grid_size = int(np.ceil(np.sqrt(num_channels)))

    h, w = fmap.shape[0], fmap.shape[1]
    grid = np.zeros((grid_size * h, grid_size * w))

    for i in range(num_channels):
        row = i // grid_size
        col = i % grid_size

        feature = fmap[:, :, i]
        feature -= feature.min()
        feature /= feature.max() + 1e-5

        grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = feature

    return grid


def save_feature_map_image(model, preprocess_fn, layer_names, model_name):

    x = preprocess_fn(img_array.copy())
    outputs = [model.get_layer(name).output for name in layer_names]
    feature_model = tf.keras.models.Model(inputs=model.input, outputs=outputs)

    feature_maps = feature_model.predict(x)

    layer_grids = []

    for fmap in feature_maps:
        grid = build_grid(fmap, max_channels=16)

        target_width = 1024
        aspect_ratio = grid.shape[0] / grid.shape[1]
        target_height = int(target_width * aspect_ratio)

        resized_grid = cv2.resize(grid, (target_width, target_height))
        layer_grids.append(resized_grid)

    final_image = np.vstack(layer_grids)

    os.makedirs("feature_maps", exist_ok=True)
    save_path = f"feature_maps/{model_name}_feature_maps.png"

    plt.imsave(save_path, final_image, cmap="viridis")
    print(f"✅ Saved ONE feature map image for {model_name}")


vgg = VGG16(weights="imagenet")
resnet = ResNet50(weights="imagenet")
mobilenet = MobileNetV2(weights="imagenet")


vgg_layers = ["block1_conv1", "block3_conv1", "block5_conv1"]
resnet_layers = ["conv1_conv", "conv3_block1_out", "conv5_block1_out"]
mobilenet_layers = ["Conv1", "block_3_expand", "block_13_expand"]


save_feature_map_image(vgg, vgg_preprocess, vgg_layers, "VGG16")
save_feature_map_image(resnet, resnet_preprocess, resnet_layers, "ResNet50")
save_feature_map_image(mobilenet, mobile_preprocess, mobilenet_layers, "MobileNetV2")

print("🎯 Done! No dimension errors now.")
