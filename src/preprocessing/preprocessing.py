"""High-performance tf.data preprocessing and augmentation pipeline."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple
import yaml

logger = logging.getLogger("Preprocessing")


def get_augmentation_layer(config: dict):
    """Build Keras Sequential augmentation layer for training split only."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    aug_cfg = config.get("augmentation", {})
    aug_layers = []

    if aug_cfg.get("horizontal_flip", True):
        aug_layers.append(layers.RandomFlip("horizontal"))

    rotation = aug_cfg.get("rotation_factor", 0.08)
    if rotation > 0:
        aug_layers.append(layers.RandomRotation(rotation, fill_mode="nearest"))

    # Zoom in up to 25% to expose the network to partial vehicle crops
    zoom_factor = aug_cfg.get("zoom_factor", 0.20)
    aug_layers.append(layers.RandomZoom(height_factor=(-zoom_factor, zoom_factor), width_factor=(-zoom_factor, zoom_factor), fill_mode="nearest"))

    # Random translation up to 15% to simulate off-center / partial vehicles
    trans_factor = aug_cfg.get("translation_factor", 0.12)
    aug_layers.append(layers.RandomTranslation(height_factor=trans_factor, width_factor=trans_factor, fill_mode="nearest"))

    contrast = aug_cfg.get("contrast_factor", 0.15)
    if contrast > 0:
        aug_layers.append(layers.RandomContrast(contrast))

    return keras.Sequential(aug_layers, name="data_augmentation")


def build_data_pipelines(
    config_path: str = "config/config.yaml",
    classes_config_path: str = "config/classes.yaml",
):
    """Create optimized tf.data.Dataset generators for train, validation, and test splits."""
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from src.dataset.normalization import get_normalizer

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_dir = Path(cfg.get("dataset", {}).get("train_dir", "data/train"))
    val_dir = Path(cfg.get("dataset", {}).get("val_dir", "data/validation"))
    test_dir = Path(cfg.get("dataset", {}).get("test_dir", "data/test"))

    img_size = int(cfg.get("model", {}).get("image_size", 224))
    batch_size = int(cfg.get("training", {}).get("batch_size", 32))
    seed = int(cfg.get("training", {}).get("seed", 42))

    normalizer = get_normalizer(classes_config_path)
    canonical_classes = normalizer.get_classes()

    logger.info(f"Target classes ({len(canonical_classes)}): {canonical_classes}")
    logger.info(f"Image target resolution: {img_size}x{img_size}x3, batch size: {batch_size}")

    # Build raw datasets using image_dataset_from_directory
    train_raw = tf.keras.utils.image_dataset_from_directory(
        directory=train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=canonical_classes,
        color_mode="rgb",
        batch_size=None,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=seed,
    )

    val_raw = tf.keras.utils.image_dataset_from_directory(
        directory=val_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=canonical_classes,
        color_mode="rgb",
        batch_size=None,
        image_size=(img_size, img_size),
        shuffle=False,
    )

    test_raw = tf.keras.utils.image_dataset_from_directory(
        directory=test_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=canonical_classes,
        color_mode="rgb",
        batch_size=None,
        image_size=(img_size, img_size),
        shuffle=False,
    )

    aug_model = get_augmentation_layer(cfg)

    def train_prep(image, label):
        # Augment
        augmented = aug_model(image, training=True)
        # EfficientNet preprocess
        processed = preprocess_input(augmented)
        return processed, label

    def eval_prep(image, label):
        # Strictly unaugmented - only preprocessing
        processed = preprocess_input(image)
        return processed, label

    autotune = tf.data.AUTOTUNE

    # Build high-performance train pipeline
    train_ds = (
        train_raw
        .shuffle(buffer_size=1000, seed=seed)
        .batch(batch_size)
        .map(train_prep, num_parallel_calls=autotune)
        .prefetch(buffer_size=autotune)
    )

    # Build validation pipeline
    val_ds = (
        val_raw
        .batch(batch_size)
        .map(eval_prep, num_parallel_calls=autotune)
        .prefetch(buffer_size=autotune)
    )

    # Build test pipeline
    test_ds = (
        test_raw
        .batch(batch_size)
        .map(eval_prep, num_parallel_calls=autotune)
        .prefetch(buffer_size=autotune)
    )

    return train_ds, val_ds, test_ds, canonical_classes


def preprocess_single_image(image_path: Path, target_size: int = 224):
    """Preprocess a single image for standalone inference."""
    import numpy as np
    from PIL import Image
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32)
    processed = preprocess_input(arr)
    batch_tensor = np.expand_dims(processed, axis=0)
    return batch_tensor
