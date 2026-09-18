"""Dataset inspection and health report generator."""

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict
from PIL import Image
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.normalization import get_normalizer
from src.dataset.parse_annotations import detect_annotation_format, discover_and_parse_all_annotations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DatasetInspector")


def inspect_dataset(
    config_path: str = "config/config.yaml",
    classes_config_path: str = "config/classes.yaml",
) -> Dict[str, Any]:
    """Inspect dataset files, detect format, count classes, check anomalies, and write reports."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg.get("dataset", {}).get("raw_dir", "data/raw"))
    reports_dir = Path(cfg.get("reports", {}).get("dir", "reports"))
    report_json_path = Path(cfg.get("reports", {}).get("dataset_report_json", "reports/dataset_report.json"))
    report_csv_path = Path(cfg.get("reports", {}).get("dataset_report_csv", "reports/dataset_report.csv"))
    min_crop_size = int(cfg.get("dataset", {}).get("min_crop_size", 16))

    reports_dir.mkdir(parents=True, exist_ok=True)
    normalizer = get_normalizer(classes_config_path)
    canonical_classes = normalizer.get_classes()

    logger.info(f"Inspecting dataset at {raw_dir}...")
    format_detected = detect_annotation_format(raw_dir)
    logger.info(f"Detected annotation format: {format_detected}")

    image_extensions = set(cfg.get("dataset", {}).get("supported_extensions", [".jpg", ".jpeg", ".png"]))
    all_image_paths = [p for p in raw_dir.rglob("*") if p.suffix.lower() in image_extensions]
    total_images = len(all_image_paths)

    # Check corruption
    corrupt_images = []
    valid_images = []
    for ip in all_image_paths:
        try:
            with Image.open(ip) as img:
                img.verify()
            valid_images.append(ip)
        except Exception as e:
            corrupt_images.append({"path": str(ip), "error": str(e)})

    # Parse annotations
    annotations = discover_and_parse_all_annotations(raw_dir, class_names=canonical_classes)
    total_annotations = len(annotations)

    # Analyze annotations
    class_counter = Counter()
    unmapped_counter = Counter()
    missing_image_count = 0
    zero_or_negative_boxes = 0
    very_small_boxes = 0

    for annot in annotations:
        if not annot.source_image_path.exists():
            missing_image_count += 1

        if annot.width <= 0 or annot.height <= 0:
            zero_or_negative_boxes += 1
        elif annot.width < min_crop_size or annot.height < min_crop_size:
            very_small_boxes += 1

        canonical = normalizer.normalize(annot.category_name)
        if canonical and canonical in canonical_classes:
            class_counter[canonical] += 1
        else:
            unmapped_counter[annot.category_name] += 1

    # Class imbalance metrics
    counts = list(class_counter.values())
    max_count = max(counts) if counts else 0
    min_count = min(counts) if counts else 0
    imbalance_ratio = round(max_count / max(1, min_count), 2)

    report_data = {
        "dataset_name": cfg.get("dataset", {}).get("name", "ConstructionXC7C"),
        "raw_directory": str(raw_dir.resolve()),
        "annotation_format_detected": format_detected,
        "total_images_found": total_images,
        "valid_images": len(valid_images),
        "corrupted_images_count": len(corrupt_images),
        "corrupted_images": corrupt_images[:10],
        "total_annotations_found": total_annotations,
        "target_canonical_classes": canonical_classes,
        "total_target_classes": len(canonical_classes),
        "annotations_per_class": dict(class_counter),
        "unmapped_categories": dict(unmapped_counter),
        "missing_source_images_count": missing_image_count,
        "zero_or_negative_boxes_count": zero_or_negative_boxes,
        "very_small_boxes_count": very_small_boxes,
        "class_imbalance": {
            "max_class_samples": max_count,
            "min_class_samples": min_count,
            "imbalance_ratio_max_to_min": imbalance_ratio,
            "is_significantly_imbalanced": imbalance_ratio > 3.0,
        },
    }

    # Save reports/dataset_report.json
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Saved dataset report JSON to {report_json_path}")

    # Save reports/dataset_report.csv
    csv_rows = []
    for cls_name in canonical_classes:
        cnt = class_counter.get(cls_name, 0)
        percentage = round((cnt / total_annotations * 100) if total_annotations else 0, 2)
        csv_rows.append({
            "class_name": cls_name,
            "annotation_count": cnt,
            "percentage_of_total": percentage,
        })
    with open(report_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_name", "annotation_count", "percentage_of_total"])
        writer.writeheader()
        writer.writerows(csv_rows)
    logger.info(f"Saved dataset report CSV to {report_csv_path}")

    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect dataset and generate health reports")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--classes-config", default="config/classes.yaml", help="Path to classes.yaml")
    args = parser.parse_args()

    inspect_dataset(config_path=args.config, classes_config_path=args.classes_config)
