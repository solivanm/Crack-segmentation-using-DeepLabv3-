"""
Crack Segmentation Evaluation and Visualization
===============================================

This script loads a trained crack-segmentation model, evaluates it on a test
image/mask dataset, and generates publication-ready comparison figures.

The comparison figure contains four columns:
1. Input image
2. Ground-truth mask
3. Predicted mask
4. Overlay of ground truth and prediction

Expected dataset structure
--------------------------
test/
├── images/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── masks/
    ├── image_001.png
    ├── image_002.png
    └── ...

Overlay color convention
------------------------
- Green: ground-truth crack pixels
- Blue : predicted crack pixels
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.utils import CustomObjectScope


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

IMAGE_SIZE = 128
BATCH_SIZE = 2
THRESHOLD = 0.5
SMOOTH = 1e-15

IMAGE_DIR = Path("F:/PycharmProjects/pythonProject5/Segmentation/test/images")
MASK_DIR = Path("F:/PycharmProjects/pythonProject5/Segmentation/test/masks")
MODEL_PATH = Path("weights/deeplab_xception_new_imagenet_parameter11000.h5")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Metrics and losses
# -----------------------------------------------------------------------------

def iou_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Intersection over Union metric for binary segmentation."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > THRESHOLD, tf.float32)

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
    """Dice loss used when loading/compiling the trained model."""
    return 1.0 - dice_coefficient(y_true, y_pred)


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------

def load_image(image_path: Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Load, resize, and normalize one RGB image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0

    return image


def load_mask(mask_path: Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Load, resize, and binarize one mask."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)

    return mask


def load_dataset(image_dir: Path, mask_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load all test images and masks from disk."""
    image_paths = sorted(image_dir.glob("*"))
    mask_paths = sorted(mask_dir.glob("*"))

    if not image_paths:
        raise ValueError(f"No images found in: {image_dir}")

    if not mask_paths:
        raise ValueError(f"No masks found in: {mask_dir}")

    if len(image_paths) != len(mask_paths):
        raise ValueError(
            f"Image/mask count mismatch: {len(image_paths)} images, "
            f"{len(mask_paths)} masks."
        )

    images = [load_image(path) for path in image_paths]
    masks = [load_mask(path) for path in mask_paths]
    names = [path.name for path in image_paths]

    return np.asarray(images, dtype=np.float32), np.asarray(masks, dtype=np.float32), names


# -----------------------------------------------------------------------------
# Model loading and prediction
# -----------------------------------------------------------------------------

def load_trained_model(model_path: Path) -> tf.keras.Model:
    """Load a trained Keras model with custom metrics and loss functions."""
    custom_objects = {
        "iou": iou_score,
        "iou_score": iou_score,
        "dice_coef": dice_coefficient,
        "dice_coefficient": dice_coefficient,
        "dice_loss": dice_loss,
    }

    with CustomObjectScope(custom_objects):
        model = tf.keras.models.load_model(str(model_path))

    model.compile(
        optimizer="adam",
        loss=dice_loss,
        metrics=[iou_score, dice_coefficient, Recall(name="recall"), Precision(name="precision")],
    )

    return model


def predict_masks(
    model: tf.keras.Model,
    images: np.ndarray,
    batch_size: int = BATCH_SIZE,
    threshold: float = THRESHOLD,
) -> np.ndarray:
    """Predict binary masks from input images."""
    probabilities = model.predict(images, batch_size=batch_size, verbose=1)
    binary_masks = (probabilities > threshold).astype(np.float32)

    return binary_masks


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def create_overlay(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """
    Create an overlay image.

    Green pixels represent ground truth.
    Blue pixels represent predictions.
    """
    overlay = image.copy()

    gt_mask = ground_truth.squeeze() > 0
    pred_mask = prediction.squeeze() > 0

    overlay[gt_mask] = [0.0, 1.0, 0.0]
    overlay[pred_mask] = [0.0, 0.0, 1.0]

    return overlay


def plot_prediction_grid(
    images: np.ndarray,
    masks: np.ndarray,
    predictions: np.ndarray,
    sample_indices: Iterable[int],
    output_path: Path,
    resize_to: int | None = None,
) -> None:
    """Create and save a grid of image/mask/prediction/overlay comparisons."""
    sample_indices = list(sample_indices)
    num_rows = len(sample_indices)
    column_titles = ["Image", "Ground Truth", "Predicted Mask", "Segmented Image"]

    fig, axes = plt.subplots(num_rows, 4, figsize=(12, 3 * num_rows))

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, index in enumerate(sample_indices):
        image = images[index]
        mask = masks[index]
        prediction = predictions[index]
        overlay = create_overlay(image, mask, prediction)

        panels = [image, mask.squeeze(), prediction.squeeze(), overlay]
        cmaps = [None, "gray", "gray", None]

        for col, (panel, cmap) in enumerate(zip(panels, cmaps)):
            if resize_to is not None:
                panel = cv2.resize(panel, (resize_to, resize_to), interpolation=cv2.INTER_NEAREST)

            axes[row, col].imshow(panel, cmap=cmap)
            axes[row, col].axis("off")

            if row == 0:
                axes[row, col].set_title(column_titles[col], fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {output_path}")


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate_model(model: tf.keras.Model, images: np.ndarray, masks: np.ndarray) -> None:
    """Evaluate the model and print all metric values."""
    results = model.evaluate(images, masks, batch_size=BATCH_SIZE, verbose=1)

    print("\nEvaluation results")
    print("------------------")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.5f}")


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

def main() -> None:
    """Run model evaluation and visualization."""
    print("Loading test dataset...")
    images, masks, image_names = load_dataset(IMAGE_DIR, MASK_DIR)

    print(f"Number of test images: {len(images)}")
    print(f"Image shape          : {images.shape}")
    print(f"Mask shape           : {masks.shape}")
    print(f"Mask values          : {np.unique(masks)}")

    print("\nLoading trained model...")
    model = load_trained_model(MODEL_PATH)

    print("\nEvaluating model...")
    evaluate_model(model, images, masks)

    print("\nPredicting masks...")
    predictions = predict_masks(model, images)

    # Example 1: crack500-style samples from your original script.
    crack500_indices = [107, 131, 93, 228]
    crack500_indices = [i for i in crack500_indices if i < len(images)]

    if crack500_indices:
        plot_prediction_grid(
            images=images,
            masks=masks,
            predictions=predictions,
            sample_indices=crack500_indices,
            output_path=OUTPUT_DIR / "crack500_predictions.png",
            resize_to=416,
        )

    # Example 2: crack-forest-style samples from your original script.
    crack_forest_indices = [1948, 1955, 1958]
    crack_forest_indices = [i for i in crack_forest_indices if i < len(images)]

    if crack_forest_indices:
        plot_prediction_grid(
            images=images,
            masks=masks,
            predictions=predictions,
            sample_indices=crack_forest_indices,
            output_path=OUTPUT_DIR / "crack_forest_predictions.png",
            resize_to=416,
        )


if __name__ == "__main__":
    main()
