"""Generate standard and normalized confusion matrices and training loss/accuracy curves."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from src.dataset.normalization import get_normalizer
from src.preprocessing.preprocessing import build_data_pipelines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MatrixAndCurves")


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_raw_path: Path,
    output_norm_path: Path,
):
    """Plot and save both raw count and normalized confusion matrices."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-9)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Raw Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
    )
    plt.title("Construction Vehicle Classification - Confusion Matrix", fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("Predicted Vehicle Class", fontsize=12, labelpad=10)
    plt.ylabel("True Vehicle Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_raw_path, dpi=300)
    plt.close()
    logger.info(f"Saved raw confusion matrix to: {output_raw_path.resolve()}")

    # 2. Normalized Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Oranges",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
    )
    plt.title("Construction Vehicle Classification - Normalized Confusion Matrix", fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("Predicted Vehicle Class", fontsize=12, labelpad=10)
    plt.ylabel("True Vehicle Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_norm_path, dpi=300)
    plt.close()
    logger.info(f"Saved normalized confusion matrix to: {output_norm_path.resolve()}")


def plot_training_curves(
    stage1_csv: Optional[Path],
    stage2_csv: Optional[Path],
    output_acc_path: Path,
    output_loss_path: Path,
):
    """Plot training and validation accuracy and loss over both training stages."""
    frames = []
    if stage1_csv and stage1_csv.exists():
        df1 = pd.read_csv(stage1_csv)
        df1["stage"] = "Stage 1 (Feature Extraction)"
        frames.append(df1)

    if stage2_csv and stage2_csv.exists():
        df2 = pd.read_csv(stage2_csv)
        df2["stage"] = "Stage 2 (Fine-Tuning)"
        frames.append(df2)

    if not frames:
        logger.warning("No training log CSV files found. Skipping training curves.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined["global_epoch"] = range(1, len(combined) + 1)
    stage1_len = len(frames[0]) if len(frames) > 1 else 0

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(combined["global_epoch"], combined["accuracy"], label="Training Accuracy", color="#2980b9", lw=2.5)
    plt.plot(combined["global_epoch"], combined["val_accuracy"], label="Validation Accuracy", color="#e67e22", lw=2.5)
    if stage1_len > 0:
        plt.axvline(x=stage1_len + 0.5, color="gray", linestyle="--", alpha=0.7, label="Fine-Tuning Begins")
    plt.title("Training & Validation Accuracy", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(frameon=True, facecolor="white", loc="lower right")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_acc_path, dpi=300)
    plt.close()
    logger.info(f"Saved training accuracy curve to: {output_acc_path.resolve()}")

    # Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(combined["global_epoch"], combined["loss"], label="Training Loss", color="#27ae60", lw=2.5)
    plt.plot(combined["global_epoch"], combined["val_loss"], label="Validation Loss", color="#c0392b", lw=2.5)
    if stage1_len > 0:
        plt.axvline(x=stage1_len + 0.5, color="gray", linestyle="--", alpha=0.7, label="Fine-Tuning Begins")
    plt.title("Training & Validation Loss", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Categorical Crossentropy Loss", fontsize=12)
    plt.legend(frameon=True, facecolor="white", loc="upper right")
    plt.tight_layout()
    plt.savefig(output_loss_path, dpi=300)
    plt.close()
    logger.info(f"Saved training loss curve to: {output_loss_path.resolve()}")


def generate_all_matrices_and_curves(
    config_path: str = "config/config.yaml",
    classes_config_path: str = "config/classes.yaml",
    model_path: Optional[str] = None,
):
    """Orchestrate generation of confusion matrices and training progression curves."""
    from tensorflow import keras

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    reports_dir = Path(cfg.get("reports", {}).get("dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    cm_raw_path = Path(cfg.get("reports", {}).get("confusion_matrix_png", "reports/confusion_matrix.png"))
    cm_norm_path = Path(cfg.get("reports", {}).get("confusion_matrix_normalized_png", "reports/confusion_matrix_normalized.png"))
    acc_curve_path = Path(cfg.get("reports", {}).get("training_accuracy_png", "reports/training_accuracy.png"))
    loss_curve_path = Path(cfg.get("reports", {}).get("training_loss_png", "reports/training_loss.png"))

    stage1_log = reports_dir / "training_stage1_log.csv"
    stage2_log = reports_dir / "training_stage2_log.csv"

    # Plot training curves from log CSVs
    plot_training_curves(stage1_log, stage2_log, acc_curve_path, loss_curve_path)

    # Plot confusion matrices
    keras_model_path = Path(model_path or cfg.get("export", {}).get("keras_model_path", "models/construction_vehicle_efficientnetb0.keras"))
    if not keras_model_path.exists():
        fallback_stage1 = Path("models/stage1_feature_extractor.keras")
        if fallback_stage1.exists():
            keras_model_path = fallback_stage1
        else:
            logger.warning("No trained model found to generate confusion matrix.")
            return

    model = keras.models.load_model(keras_model_path)
    _, _, test_ds, canonical_classes = build_data_pipelines(config_path, classes_config_path)

    y_true_list = []
    y_pred_list = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_list.append(np.argmax(preds, axis=1))
        y_true_list.append(np.argmax(labels.numpy(), axis=1))

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    normalizer = get_normalizer(classes_config_path)
    display_names = [normalizer.get_display_name(c) for c in canonical_classes]

    plot_confusion_matrices(y_true, y_pred, display_names, cm_raw_path, cm_norm_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate confusion matrices and training curves")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--classes-config", default="config/classes.yaml", help="Path to classes.yaml")
    parser.add_argument("--model", default=None, help="Path to model")
    args = parser.parse_args()

    generate_all_matrices_and_curves(
        config_path=args.config,
        classes_config_path=args.classes_config,
        model_path=args.model,
    )
