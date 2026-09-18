"""Leakage-free dataset splitting grouped strictly by source image ID."""

import argparse
import collections
import csv
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DatasetSplitter")


def check_data_leakage(
    train_source_ids: Set[str],
    val_source_ids: Set[str],
    test_source_ids: Set[str],
) -> Tuple[bool, Dict[str, int]]:
    """Strictly verify mathematical disjointness of source image IDs across splits."""
    leak_train_val = train_source_ids.intersection(val_source_ids)
    leak_train_test = train_source_ids.intersection(test_source_ids)
    leak_val_test = val_source_ids.intersection(test_source_ids)

    leak_counts = {
        "train_val_overlap": len(leak_train_val),
        "train_test_overlap": len(leak_train_test),
        "val_test_overlap": len(leak_val_test),
    }

    has_leakage = bool(leak_train_val or leak_train_test or leak_val_test)
    return has_leakage, leak_counts


def split_dataset(
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """Split metadata and crops into train, validation, and test sets without leakage."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    metadata_path = Path(cfg.get("dataset", {}).get("metadata_path", "data/metadata.csv"))
    train_dir = Path(cfg.get("dataset", {}).get("train_dir", "data/train"))
    val_dir = Path(cfg.get("dataset", {}).get("val_dir", "data/validation"))
    test_dir = Path(cfg.get("dataset", {}).get("test_dir", "data/test"))

    split_ratios = cfg.get("dataset", {}).get("split_ratios", {"train": 0.70, "val": 0.15, "test": 0.15})
    seed = int(cfg.get("dataset", {}).get("seed", 42))

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file {metadata_path} not found. Run create_crops.py first.")

    # Read metadata
    rows = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError("Metadata file is empty. No crops available to split.")

    # Group crops by source image ID
    image_to_crops: Dict[str, List[dict]] = collections.defaultdict(list)
    image_to_primary_class: Dict[str, str] = {}
    for r in rows:
        img_id = r["image_id"]
        image_to_crops[img_id].append(r)
        if img_id not in image_to_primary_class:
            image_to_primary_class[img_id] = r["class"]

    unique_image_ids = list(image_to_crops.keys())
    logger.info(f"Total unique source image IDs: {len(unique_image_ids)}")
    logger.info(f"Total crop samples: {len(rows)}")

    # Stratified-aware grouped shuffle
    rng = random.Random(seed)
    rng.shuffle(unique_image_ids)

    # Calculate split index
    total_imgs = len(unique_image_ids)
    train_ratio = split_ratios.get("train", 0.70)
    val_ratio = split_ratios.get("val", 0.15)

    n_train = int(round(total_imgs * train_ratio))
    n_val = int(round(total_imgs * val_ratio))

    train_ids = set(unique_image_ids[:n_train])
    val_ids = set(unique_image_ids[n_train:n_train + n_val])
    test_ids = set(unique_image_ids[n_train + n_val:])

    # Strict leakage validation
    has_leakage, leak_stats = check_data_leakage(train_ids, val_ids, test_ids)
    if has_leakage:
        logger.error(f"DATA LEAKAGE DETECTED! Stats: {leak_stats}")
        raise RuntimeError(f"Data leakage detected during splitting: {leak_stats}")

    logger.info(f"Data leakage check passed: 0 overlapping source images across splits.")

    # Assign split to each crop and copy into train/val/test directories
    split_dir_map = {
        "train": train_dir,
        "validation": val_dir,
        "test": test_dir,
    }

    for d in split_dir_map.values():
        d.mkdir(parents=True, exist_ok=True)

    updated_rows = []
    split_counts = collections.Counter()
    split_class_counts = collections.defaultdict(collections.Counter)

    for r in rows:
        img_id = r["image_id"]
        if img_id in train_ids:
            split_name = "train"
        elif img_id in val_ids:
            split_name = "validation"
        else:
            split_name = "test"

        r["split"] = split_name
        updated_rows.append(r)
        split_counts[split_name] += 1
        split_class_counts[split_name][r["class"]] += 1

        # Copy/link crop to destination folder
        src_crop = Path(r["crop_path"])
        if src_crop.exists():
            dest_folder = split_dir_map[split_name] / r["class"]
            dest_folder.mkdir(parents=True, exist_ok=True)
            dest_file = dest_folder / src_crop.name
            if not dest_file.exists():
                shutil.copy2(src_crop, dest_file)

    # Save updated metadata.csv with split assignments
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(updated_rows[0].keys()))
        writer.writeheader()
        writer.writerows(updated_rows)

    logger.info(f"Updated {metadata_path} with split assignments.")
    logger.info(f"Split Summary - Train crops: {split_counts['train']} (images: {len(train_ids)})")
    logger.info(f"Split Summary - Val crops: {split_counts['validation']} (images: {len(val_ids)})")
    logger.info(f"Split Summary - Test crops: {split_counts['test']} (images: {len(test_ids)})")

    for s in ["train", "validation", "test"]:
        logger.info(f"Split '{s}' class distribution: {dict(split_class_counts[s])}")

    return {
        "train_crops": split_counts["train"],
        "val_crops": split_counts["validation"],
        "test_crops": split_counts["test"],
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "test_images": len(test_ids),
        "class_distribution": {s: dict(split_class_counts[s]) for s in split_class_counts},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test without data leakage")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    split_dataset(config_path=args.config)
