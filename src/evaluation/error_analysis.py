"""Error analysis: Find misclassifications, rank top confident errors, and render grid."""

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import yaml

from src.dataset.normalization import get_normalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ErrorAnalysis")


def perform_error_analysis(
    config_path: str = "config/config.yaml",
    classes_config_path: str = "config/classes.yaml",
    model_path: Optional[str] = None,
    max_grid_samples: int = 16,
):
    """Scan test set, identify errors, save crops to reports/misclassified, and rank confident errors."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications.efficientnet import preprocess_input

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    test_dir = Path(cfg.get("dataset", {}).get("test_dir", "data/test"))
    reports_dir = Path(cfg.get("reports", {}).get("dir", "reports"))
    misclassified_dir = Path(cfg.get("reports", {}).get("misclassified_dir", "reports/misclassified"))
    top_errors_csv = Path(cfg.get("reports", {}).get("top_confident_errors_csv", "reports/top_confident_errors.csv"))
    grid_png_path = reports_dir / "misclassified_grid.png"

    reports_dir.mkdir(parents=True, exist_ok=True)
    misclassified_dir.mkdir(parents=True, exist_ok=True)

    normalizer = get_normalizer(classes_config_path)
    canonical_classes = normalizer.get_classes()
    class_to_idx = {c: i for i, c in enumerate(canonical_classes)}

    keras_model_path = Path(model_path or cfg.get("export", {}).get("keras_model_path", "models/construction_vehicle_efficientnetb0.keras"))
    if not keras_model_path.exists():
        fallback_stage1 = Path("models/stage1_feature_extractor.keras")
        if fallback_stage1.exists():
            keras_model_path = fallback_stage1
        else:
            raise FileNotFoundError(f"Model file not found at {keras_model_path}")

    logger.info(f"Loading model for error analysis: {keras_model_path}")
    model = keras.models.load_model(keras_model_path)
    img_size = int(cfg.get("model", {}).get("image_size", 224))

    # Collect all test images
    test_image_paths = []
    for c in canonical_classes:
        c_folder = test_dir / c
        if c_folder.exists():
            for f in c_folder.iterdir():
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    test_image_paths.append((f, c))

    logger.info(f"Found {len(test_image_paths)} test crops across all classes.")
    if not test_image_paths:
        logger.warning("No test images found for error analysis.")
        return

    misclassified = []

    for img_path, actual_class in test_image_paths:
        try:
            pil_img = Image.open(img_path).convert("RGB")
            resized = pil_img.resize((img_size, img_size), Image.Resampling.BILINEAR)
            arr = np.array(resized, dtype=np.float32)
            prep = preprocess_input(arr)
            batch = np.expand_dims(prep, axis=0)

            probs = model.predict(batch, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            predicted_class = canonical_classes[pred_idx]
            confidence = float(probs[pred_idx])

            if predicted_class != actual_class:
                misclassified.append({
                    "image_path": str(img_path.resolve()),
                    "image_name": img_path.name,
                    "actual_class": actual_class,
                    "actual_display": normalizer.get_display_name(actual_class),
                    "predicted_class": predicted_class,
                    "predicted_display": normalizer.get_display_name(predicted_class),
                    "confidence": confidence,
                    "all_probabilities": {c: float(p) for c, p in zip(canonical_classes, probs)},
                    "pil_img": pil_img,
                })
        except Exception as e:
            logger.warning(f"Error analyzing image {img_path}: {e}")

    logger.info(f"Total misclassified test crops: {len(misclassified)} / {len(test_image_paths)}")

    # Sort descending by confidence (Top Confident Errors)
    misclassified_sorted = sorted(misclassified, key=lambda x: x["confidence"], reverse=True)

    # Save to reports/top_confident_errors.csv
    with open(top_errors_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_name", "actual_class", "predicted_class", "confidence", "image_path"]
        )
        writer.writeheader()
        for item in misclassified_sorted:
            writer.writerow({
                "image_name": item["image_name"],
                "actual_class": item["actual_class"],
                "predicted_class": item["predicted_class"],
                "confidence": round(item["confidence"], 4),
                "image_path": item["image_path"],
            })
    logger.info(f"Saved top confident errors to: {top_errors_csv.resolve()}")

    # Save misclassified crops to reports/misclassified/
    for item in misclassified_sorted[:50]:  # save up to 50
        dest_filename = f"ACTUAL_{item['actual_class']}_PRED_{item['predicted_class']}_{item['image_name']}"
        dest_path = misclassified_dir / dest_filename
        try:
            item["pil_img"].save(dest_path, "JPEG")
        except Exception:
            pass
    logger.info(f"Saved misclassified sample crops to: {misclassified_dir.resolve()}")

    # Render visual inspection grid of top errors
    if misclassified_sorted:
        samples_to_plot = misclassified_sorted[:max_grid_samples]
        n_items = len(samples_to_plot)
        cols = 4
        rows = (n_items + cols - 1) // cols

        plt.figure(figsize=(16, 4 * rows))
        for idx, item in enumerate(samples_to_plot):
            plt.subplot(rows, cols, idx + 1)
            plt.imshow(item["pil_img"])
            plt.axis("off")
            title_text = (
                f"True: {item['actual_display']}\n"
                f"Pred: {item['predicted_display']} ({item['confidence']*100:.1f}%)"
            )
            plt.title(title_text, color="darkred", fontsize=10, fontweight="bold")
        plt.suptitle("Top Confident Misclassified Test Examples", fontsize=15, fontweight="bold", y=0.99)
        plt.tight_layout()
        plt.savefig(grid_png_path, dpi=300)
        plt.close()
        logger.info(f"Saved misclassified visual grid to: {grid_png_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform error analysis on test dataset")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--classes-config", default="config/classes.yaml", help="Path to classes.yaml")
    parser.add_argument("--model", default=None, help="Path to model file")
    args = parser.parse_args()

    perform_error_analysis(
        config_path=args.config,
        classes_config_path=args.classes_config,
        model_path=args.model,
    )
