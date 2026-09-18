"""Stage 2: Fine-tuning top layers of EfficientNetB0 with frozen BatchNorm."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml

from src.dataset.normalization import get_normalizer
from src.models.callbacks import build_callbacks
from src.models.efficientnet import count_parameters, unfreeze_for_fine_tuning
from src.preprocessing.preprocessing import build_data_pipelines
from src.training.train import compute_training_class_weights, detect_and_configure_hardware, setup_reproducibility

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FineTuneStage2")


def export_classes_json(canonical_classes: list, output_path: Path, normalizer):
    """Save ordered canonical classes and display metadata to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classes_payload = {
        "classes": canonical_classes,
        "num_classes": len(canonical_classes),
        "details": {
            c: {
                "id": idx,
                "display_name": normalizer.get_display_name(c),
                "emoji": normalizer.get_emoji(c),
                "color": normalizer.get_color(c),
            }
            for idx, c in enumerate(canonical_classes)
        }
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(classes_payload, f, indent=2)
    logger.info(f"Exported classes metadata to: {output_path.resolve()}")


def export_tflite_model(keras_model, tflite_path: Path):
    """Export trained Keras model to lightweight TFLite format."""
    try:
        import tensorflow as tf
        tflite_path.parent.mkdir(parents=True, exist_ok=True)
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        logger.info(f"TFLite model successfully exported to: {tflite_path.resolve()} ({tflite_path.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        logger.warning(f"TFLite export failed or skipped: {e}")


def run_stage2_fine_tuning(
    config_path: str = "config/config.yaml",
    classes_config_path: str = "config/classes.yaml",
    stage1_model_path: Optional[str] = None,
    epochs_override: Optional[int] = None,
):
    """Execute Stage 2 fine-tuning and export final models."""
    import tensorflow as tf
    from tensorflow import keras

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("training", {}).get("seed", 42))
    setup_reproducibility(seed)
    detect_and_configure_hardware()

    models_dir = Path("models")
    reports_dir = Path(cfg.get("reports", {}).get("dir", "reports"))
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    input_model_path = Path(stage1_model_path or "models/stage1_feature_extractor.keras")
    final_model_path = Path(cfg.get("export", {}).get("keras_model_path", "models/construction_vehicle_efficientnetb0.keras"))
    classes_json_path = Path(cfg.get("export", {}).get("classes_json_path", "models/classes.json"))
    tflite_path = Path(cfg.get("export", {}).get("tflite_model_path", "models/construction_vehicle_efficientnetb0.tflite"))
    stage2_log = reports_dir / "training_stage2_log.csv"

    if not input_model_path.exists():
        raise FileNotFoundError(f"Stage 1 model not found at {input_model_path}. Run train.py first.")

    logger.info(f"Loading Stage 1 model from: {input_model_path}")
    model = keras.models.load_model(input_model_path)

    unfreeze_layers = int(cfg.get("fine_tuning", {}).get("unfreeze_top_layers", 30))
    freeze_bn = bool(cfg.get("fine_tuning", {}).get("freeze_batch_norm", True))
    unfreeze_for_fine_tuning(model, unfreeze_top_layers=unfreeze_layers, freeze_batch_norm=freeze_bn)

    total_params, train_params, non_train_params = count_parameters(model)
    logger.info(f"Fine-tuning parameters - Total: {total_params:,} | Trainable: {train_params:,} | Non-trainable: {non_train_params:,}")

    lr = float(cfg.get("fine_tuning", {}).get("learning_rate", 1e-5))
    epochs = epochs_override or int(cfg.get("fine_tuning", {}).get("epochs", 12))
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("Building data pipelines for fine-tuning...")
    train_ds, val_ds, test_ds, canonical_classes = build_data_pipelines(
        config_path=config_path,
        classes_config_path=classes_config_path,
    )

    normalizer = get_normalizer(classes_config_path)
    train_dir = Path(cfg.get("dataset", {}).get("train_dir", "data/train"))
    class_weights = None
    if bool(cfg.get("training", {}).get("use_class_weights", True)):
        class_weights = compute_training_class_weights(train_dir, canonical_classes)

    callbacks = build_callbacks(
        checkpoint_path=final_model_path,
        csv_log_path=stage2_log,
        patience_early_stopping=int(cfg.get("fine_tuning", {}).get("callbacks", {}).get("early_stopping", {}).get("patience", 4)),
        patience_reduce_lr=int(cfg.get("fine_tuning", {}).get("callbacks", {}).get("reduce_lr", {}).get("patience", 2)),
        reduce_lr_factor=float(cfg.get("fine_tuning", {}).get("callbacks", {}).get("reduce_lr", {}).get("factor", 0.2)),
        min_lr=float(cfg.get("fine_tuning", {}).get("callbacks", {}).get("reduce_lr", {}).get("min_lr", 1e-8)),
    )

    logger.info(f"Starting Stage 2 fine-tuning for {epochs} epochs (lr={lr})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # Load best checkpoint saved by ModelCheckpoint
    if final_model_path.exists():
        logger.info(f"Reloading best validation checkpoint from: {final_model_path}")
        best_model = keras.models.load_model(final_model_path)
    else:
        best_model = model
        best_model.save(final_model_path)

    # Export metadata and tflite
    export_classes_json(canonical_classes, classes_json_path, normalizer)
    export_tflite_model(best_model, tflite_path)

    logger.info("Stage 2 fine-tuning completed successfully.")
    return best_model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 fine-tuning: EfficientNetB0 unfreezing")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--classes-config", default="config/classes.yaml", help="Path to classes.yaml")
    parser.add_argument("--stage1-model", default=None, help="Path to Stage 1 checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()

    run_stage2_fine_tuning(
        config_path=args.config,
        classes_config_path=args.classes_config,
        stage1_model_path=args.stage1_model,
        epochs_override=args.epochs,
    )
