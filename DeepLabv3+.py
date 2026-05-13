"""
DeepLabV3+ Crack Segmentation with Xception Backbone
=====================================================

This script trains a binary crack-segmentation model using a DeepLabV3+-style
architecture with an Xception encoder, ASPP module, and squeeze-and-excitation
blocks.

Expected dataset structure
--------------------------
crack_segmentation_dataset/
├── images/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── masks/
    ├── image_001.png
    ├── image_002.png
    └── ...

Notes
-----
- Images are resized to IMAGE_SIZE x IMAGE_SIZE.
- Masks are converted to binary values: 0 for background and 1 for crack.
- The model uses Dice loss and reports IoU, Dice coefficient, recall, and precision.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    Reshape,
    UpSampling2D,
    ZeroPadding2D,
)
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Model


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATASET_DIR = Path("Your path")
IMAGE_DIR = DATASET_DIR / "images"
MASK_DIR = DATASET_DIR / "masks"

IMAGE_SIZE = 128
NUM_SAMPLES = 8000
TEST_SIZE = 0.10
RANDOM_STATE = 42

BATCH_SIZE = 2
EPOCHS = 50
LEARNING_RATE = 1e-4

MODEL_SAVE_PATH = Path("models/deeplabv3plus_xception_crack_segmentation.h5")
WEIGHTS_SAVE_PATH = Path("models/deeplabv3plus_xception_crack_segmentation_weights.h5")

SMOOTH = 1e-15


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -----------------------------------------------------------------------------
# Metrics and losses
# -----------------------------------------------------------------------------

def iou_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Intersection over Union metric for binary segmentation."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    return (intersection + SMOOTH) / (union + SMOOTH)


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Dice coefficient for binary segmentation."""
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    return (2.0 * intersection + SMOOTH) / (denominator + SMOOTH)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Dice loss used for training."""
    return 1.0 - dice_coefficient(y_true, y_pred)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_image(image_path: Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Load, resize, and normalize an RGB image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0

    return image


def load_mask(mask_path: Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Load, resize, and binarize a segmentation mask."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)

    return mask


def load_dataset(
    image_dir: Path,
    mask_dir: Path,
    image_size: int = IMAGE_SIZE,
    max_samples: int | None = NUM_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load images and masks from disk."""
    image_paths = sorted(image_dir.glob("*"))
    mask_paths = sorted(mask_dir.glob("*"))

    if len(image_paths) == 0:
        raise ValueError(f"No images found in: {image_dir}")

    if len(mask_paths) == 0:
        raise ValueError(f"No masks found in: {mask_dir}")

    if len(image_paths) != len(mask_paths):
        raise ValueError(
            f"Number of images and masks must match. "
            f"Found {len(image_paths)} images and {len(mask_paths)} masks."
        )

    if max_samples is not None:
        image_paths = image_paths[:max_samples]
        mask_paths = mask_paths[:max_samples]

    images = [load_image(path, image_size) for path in image_paths]
    masks = [load_mask(path, image_size) for path in mask_paths]

    images = np.asarray(images, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)

    return images, masks


def show_sample(image: np.ndarray, mask: np.ndarray) -> None:
    """Display one image-mask pair for a quick sanity check."""
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask.squeeze(), cmap="gray")
    plt.title("Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Model blocks
# -----------------------------------------------------------------------------

def squeeze_and_excite(inputs: tf.Tensor, ratio: int = 8) -> tf.Tensor:
    """Squeeze-and-excitation attention block."""
    filters = inputs.shape[-1]

    if filters is None:
        raise ValueError("The input channel dimension must be defined.")

    se = GlobalAveragePooling2D()(inputs)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation="relu", kernel_initializer="he_normal", use_bias=False)(se)
    se = Dense(filters, activation="sigmoid", kernel_initializer="he_normal", use_bias=False)(se)

    return inputs * se


def convolution_block(
    inputs: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    dilation_rate: int = 1,
    use_bias: bool = False,
) -> tf.Tensor:
    """Conv2D + BatchNorm + ReLU block."""
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        use_bias=use_bias,
        kernel_initializer="he_normal",
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def aspp_block(inputs: tf.Tensor, filters: int = 256) -> tf.Tensor:
    """Atrous Spatial Pyramid Pooling block."""
    input_shape = inputs.shape
    height, width = input_shape[1], input_shape[2]

    if height is None or width is None:
        raise ValueError("ASPP input height and width must be defined.")

    # Image-level features
    y_pool = AveragePooling2D(pool_size=(height, width))(inputs)
    y_pool = convolution_block(y_pool, filters=filters, kernel_size=1)
    y_pool = UpSampling2D(size=(height, width), interpolation="bilinear")(y_pool)

    # Atrous convolution branches
    y_1 = convolution_block(inputs, filters=filters, kernel_size=1, dilation_rate=1)
    y_6 = convolution_block(inputs, filters=filters, kernel_size=3, dilation_rate=6)
    y_12 = convolution_block(inputs, filters=filters, kernel_size=3, dilation_rate=12)
    y_18 = convolution_block(inputs, filters=filters, kernel_size=3, dilation_rate=18)

    y = Concatenate()([y_pool, y_1, y_6, y_12, y_18])
    y = convolution_block(y, filters=filters, kernel_size=1)

    return y


def build_deeplabv3plus(input_shape: Tuple[int, int, int]) -> Model:
    """Build DeepLabV3+ with an ImageNet-pretrained Xception backbone."""
    inputs = Input(shape=input_shape)

    backbone = Xception(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs,
    )

    # High-level encoder features
    high_level_features = backbone.get_layer("block13_sepconv2_act").output
    x_a = aspp_block(high_level_features)
    x_a = UpSampling2D(size=(4, 4), interpolation="bilinear")(x_a)

    # Low-level encoder features
    x_b = backbone.get_layer("block3_sepconv2_act").output
    x_b = ZeroPadding2D(padding=((1, 0), (1, 0)))(x_b)
    x_b = convolution_block(x_b, filters=48, kernel_size=1)

    # Decoder
    x = Concatenate()([x_a, x_b])
    x = squeeze_and_excite(x)

    x = convolution_block(x, filters=256, kernel_size=3)
    x = convolution_block(x, filters=256, kernel_size=3)
    x = squeeze_and_excite(x)

    x = UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

    outputs = Conv2D(1, kernel_size=1, padding="same", name="output_layer")(x)
    outputs = Activation("sigmoid", name="sigmoid_output")(outputs)

    return Model(inputs=inputs, outputs=outputs, name="DeepLabV3Plus_Xception_SE")


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def compile_model(model: Model) -> Model:
    """Compile the model with Dice loss and segmentation metrics."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss=dice_loss,
        metrics=[iou_score, dice_coefficient, Recall(name="recall"), Precision(name="precision")],
    )

    return model


def train() -> None:
    """Load data, build model, train, and save results."""
    set_seed(RANDOM_STATE)

    print("Loading dataset...")
    images, masks = load_dataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_size=IMAGE_SIZE,
        max_samples=NUM_SAMPLES,
    )

    print(f"Images shape: {images.shape}")
    print(f"Masks shape : {masks.shape}")
    print(f"Mask values : {np.unique(masks)}")

    x_train, x_test, y_train, y_test = train_test_split(
        images,
        masks,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    sample_index = random.randint(0, len(x_train) - 1)
    show_sample(x_train[sample_index], y_train[sample_index])

    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    model = build_deeplabv3plus(input_shape)
    model = compile_model(model)

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(WEIGHTS_SAVE_PATH),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_SAVE_PATH))

    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Best weights saved to: {WEIGHTS_SAVE_PATH}")

    evaluate_model(model, x_test, y_test)
    plot_training_curves(history)


# -----------------------------------------------------------------------------
# Evaluation and visualization
# -----------------------------------------------------------------------------

def evaluate_model(model: Model, x_test: np.ndarray, y_test: np.ndarray) -> None:
    """Evaluate the trained model on the test set."""
    results = model.evaluate(x_test, y_test, verbose=1)

    print("\nEvaluation results")
    print("------------------")
    for metric_name, metric_value in zip(model.metrics_names, results):
        print(f"{metric_name}: {metric_value:.5f}")


def plot_training_curves(history: tf.keras.callbacks.History) -> None:
    """Plot training and validation loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def predict_and_show(model: Model, image: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> None:
    """Show image, ground-truth mask, and predicted mask."""
    prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
    prediction = (prediction > threshold).astype(np.float32)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze(), cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction.squeeze(), cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train()
