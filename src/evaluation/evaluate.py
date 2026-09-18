"""Evaluation on untouched test set with comprehensive classification metrics."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import yaml

from src.dataset.normalization import get_normalizer
from src.preprocessing.preprocessing import build_data_pipelines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ModelEvaluator")


def evaluate_model_on_test(
    config_path: str = "config/config.yaml",
    classes_config_path: str = "config/classes.yaml",
    model_path: Optional[str] = None,
) -> Dict:
    """Evaluate trained model on test set and save classification reports."""
    from tensorflow import keras
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    reports_dir = Path(cfg.get("reports", {}).get("dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    txt_report_path = Path(cfg.get("reports", {}).get("classification_report_txt", "reports/classification_report.txt"))
    json_report_path = Path(cfg.get("reports", {}).get("classification_report_json", "reports/classification_report.json"))

    keras_model_path = Path(model_path or cfg.get("export", {}).get("keras_model_path", "models/construction_vehicle_efficientnetb0.keras"))
    if not keras_model_path.exists():
        fallback_stage1 = Path("models/stage1_feature_extractor.keras")
        if fallback_stage1.exists():
            keras_model_path = fallback_stage1
        else:
            raise FileNotFoundError(f"No trained model found at {keras_model_path} or {fallback_stage1}.")

    logger.info(f"Loading model for evaluation: {keras_model_path}")
    model = keras.models.load_model(keras_model_path)

    _, _, test_ds, canonical_classes = build_data_pipelines(
        config_path=config_path,
        classes_config_path=classes_config_path,
    )

    logger.info("Evaluating model on test dataset...")
    # Gather all ground truth and predictions
    y_true_list = []
    y_pred_probs_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs_list.append(preds)
        y_true_list.append(labels.numpy())

    y_true_onehot = np.concatenate(y_true_list, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)

    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    eval_loss, eval_acc = model.evaluate(test_ds, verbose=0)

    # Scikit-learn metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    normalizer = get_normalizer(classes_config_path)
    display_names = [normalizer.get_display_name(c) for c in canonical_classes]

    text_report = classification_report(
        y_true,
        y_pred,
        target_names=display_names,
        digits=4,
        zero_division=0,
    )

    dict_report = classification_report(
        y_true,
        y_pred,
        target_names=display_names,
        output_dict=True,
        zero_division=0,
    )

    # Identify high confusion pairs
    cm = confusion_matrix(y_true, y_pred)
    confusion_pairs = []
    for i in range(len(canonical_classes)):
        for j in range(len(canonical_classes)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    "actual": display_names[i],
                    "predicted": display_names[j],
                    "count": int(cm[i, j]),
                    "rate_relative_to_actual": round(float(cm[i, j]) / max(1, int(np.sum(cm[i, :]))), 4),
                })
    confusion_pairs = sorted(confusion_pairs, key=lambda x: x["count"], reverse=True)

    results = {
        "model_path": str(keras_model_path.resolve()),
        "test_loss": float(eval_loss),
        "test_accuracy": float(acc),
        "macro_metrics": {
            "precision": float(prec_macro),
            "recall": float(rec_macro),
            "f1_score": float(f1_macro),
        },
        "weighted_metrics": {
            "precision": float(prec_weighted),
            "recall": float(rec_weighted),
            "f1_score": float(f1_weighted),
        },
        "per_class_report": dict_report,
        "high_confusion_pairs": confusion_pairs[:10],
    }

    # Save reports
    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("CONSTRUCTION VEHICLE CLASSIFIER - TEST EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {keras_model_path.name}\n")
        f.write(f"Test Loss: {eval_loss:.4f}\n")
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n\n")
        f.write(f"Macro Precision: {prec_macro:.4f} | Recall: {rec_macro:.4f} | F1: {f1_macro:.4f}\n")
        f.write(f"Weighted Precision: {prec_weighted:.4f} | Recall: {rec_weighted:.4f} | F1: {f1_weighted:.4f}\n\n")
        f.write("Classification Breakdown:\n")
        f.write(text_report)
        f.write("\n\nTop Confused Class Pairs:\n")
        for p in confusion_pairs[:5]:
            f.write(f"  {p['actual']} -> predicted as {p['predicted']}: {p['count']} occurrences ({p['rate_relative_to_actual']*100:.1f}%)\n")

    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved classification text report to: {txt_report_path.resolve()}")
    logger.info(f"Saved classification JSON report to: {json_report_path.resolve()}")
    logger.info(f"Test Accuracy: {acc * 100:.2f}% | Macro F1: {f1_macro:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on test dataset")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--classes-config", default="config/classes.yaml", help="Path to classes.yaml")
    parser.add_argument("--model", default=None, help="Optional path to model .keras file")
    args = parser.parse_args()

    evaluate_model_on_test(
        config_path=args.config,
        classes_config_path=args.classes_config,
        model_path=args.model,
    )
