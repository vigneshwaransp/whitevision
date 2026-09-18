"""Stage 1: Feature extraction training with frozen EfficientNetB0 backbone."""

import argparse
import csv
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import yaml

from src.models.callbacks import build_callbacks
from src.models.efficientnet import build_efficientnetb0, count_parameters, freeze_backbone
from src.preprocessing.preprocessing import build_data_pipelines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainStage1")


def setup_reproducibility(seed: int = 42):
    """Enforce deterministic random seeds across Python, NumPy, and TensorFlow."""
    import tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Deterministic random seed configured: {seed}")


def detect_and_configure_hardware():
    """Detect and log TensorFlow, GPU, CUDA, and compute environment."""
    import tensorflow as tf
    logger.info("=" * 60)
    logger.info("HARDWARE & ACCELERATION DIAGNOSTIC")
    logger.info(f"TensorFlow Version : {tf.__version__}")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info(f"GPU Acceleration   : ENABLED ({len(gpus)} physical GPU(s) detected)")
        for idx, gpu in enumerate(gpus):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                details = tf.config.experimental.get_device_details(gpu)
                gpu_name = details.get("device_name", gpu.name)
                logger.info(f"  GPU #{idx}: {gpu_name} (Memory growth enabled)")
            except Exception as e:
                logger.info(f"  GPU #{idx}: {gpu.name} (Memory growth notice: {e})")
    else:
        logger.info("GPU Acceleration   : NOT AVAILABLE (Executing on CPU)")
    logger.info("=" * 60)


def compute_training_class_weights(train_dir: Path, canonical_classes: list) -> Optional[Dict[int, float]]:
    """Compute balanced class weights exclusively from training split."""
    from sklearn.utils.class_weight import compute_class_weight

    class_counts = {}
    total_samples = 0
    all_labels = []

    for idx, c in enumerate(canonical_classes):
        class_folder = train_dir / c
        if class_folder.exists():
            files = [f for f in class_folder.iterdir() if f.is_file()]
            count = len(files)
        else:
            count = 0
        class_counts[c] = count
        total_samples += count
        all_labels.extend([idx] * count)

    if total_samples == 0:
        logger.warning("No training samples found to compute class weights.")
        return None

    logger.info(f"Training split sample count: {total_samples}")
    logger.info(f"Training class distribution: {class_counts}")

    unique_classes = np.unique(all_labels)
    if len(unique_classes) < len(canonical_classes):
        logger.warning("Some classes have 0 samples in the training set!")

    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=np.array(all_labels),
    )

    weight_dict = {int(cls): float(w) for cls, w in zip(unique_classes, weights)}
    logger.info(f"Computed training class weights: {weight_dict}")
    return weight_dict


def run_stage1_training(
    config_path: str = "config/config.yaml",
    classes_config_path: str = "config/classes.yaml",
    epochs_override: Optional[int] = None,
):
    """Execute Stage 1 feature extraction training."""
    import tensorflow as tf
    from tensorflow import keras

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("training", {}).get("seed", 42))
    setup_reproducibility(seed)
    detect_and_configure_hardware()

    train_dir = Path(cfg.get("dataset", {}).get("train_dir", "data/train"))
    lr = float(cfg.get("training", {}).get("learning_rate", 1e-3))
    epochs = epochs_override or int(cfg.get("training", {}).get("epochs", 15))
    use_weights = bool(cfg.get("training", {}).get("use_class_weights", True))

    models_dir = Path("models")
    reports_dir = Path(cfg.get("reports", {}).get("dir", "reports"))
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    stage1_checkpoint = models_dir / "stage1_feature_extractor.keras"
    stage1_log = reports_dir / "training_stage1_log.csv"

    logger.info("Building data pipelines...")
    train_ds, val_ds, test_ds, canonical_classes = build_data_pipelines(
        config_path=config_path,
        classes_config_path=classes_config_path,
    )

    # Class weights computed strictly from train set
    class_weights = None
    if use_weights:
        class_weights = compute_training_class_weights(train_dir, canonical_classes)

    logger.info("Initializing EfficientNetB0 model...")
    model = build_efficientnetb0(
        config_path=config_path,
        num_classes=len(canonical_classes),
        image_size=int(cfg.get("model", {}).get("image_size", 224)),
        dropout_rate=float(cfg.get("model", {}).get("dropout", 0.3)),
    )
    freeze_backbone(model)

    if stage1_checkpoint.exists():
        try:
            logger.info(f"Loading previous checkpoint weights from {stage1_checkpoint}...")
            model.load_weights(stage1_checkpoint)
        except Exception as e:
            logger.warning(f"Could not load previous weights ({e}), training from scratch.")

    total_params, train_params, non_train_params = count_parameters(model)
    logger.info(f"Model parameters - Total: {total_params:,} | Trainable (Head only): {train_params:,} | Non-trainable: {non_train_params:,}")

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = build_callbacks(
        checkpoint_path=stage1_checkpoint,
        csv_log_path=stage1_log,
        patience_early_stopping=int(cfg.get("training", {}).get("callbacks", {}).get("early_stopping", {}).get("patience", 5)),
        patience_reduce_lr=int(cfg.get("training", {}).get("callbacks", {}).get("reduce_lr", {}).get("patience", 2)),
        reduce_lr_factor=float(cfg.get("training", {}).get("callbacks", {}).get("reduce_lr", {}).get("factor", 0.2)),
        min_lr=float(cfg.get("training", {}).get("callbacks", {}).get("reduce_lr", {}).get("min_lr", 1e-7)),
    )

    logger.info(f"Starting Stage 1 training for {epochs} epochs (lr={lr})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    logger.info(f"Stage 1 training completed. Best model saved to: {stage1_checkpoint.resolve()}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 training: EfficientNetB0 feature extraction")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--classes-config", default="config/classes.yaml", help="Path to classes.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    args = parser.parse_args()

    run_stage1_training(
        config_path=args.config,
        classes_config_path=args.classes_config,
        epochs_override=args.epochs,
    )
