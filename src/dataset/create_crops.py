"""Vehicle bounding-box extraction, crop generation, and data quality cleaning."""

import argparse
import csv
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from PIL import Image
import sys
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.normalization import get_normalizer
from src.dataset.parse_annotations import AnnotationRecord, discover_and_parse_all_annotations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CropGenerator")


def compute_image_hash(image_path: Path) -> str:
    """Compute MD5 hash of an image file to identify duplicate files."""
    hasher = hashlib.md5()
    with open(image_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(img: Image.Image, hash_size: int = 8) -> str:
    """Compute difference hash (dHash) of PIL image for visual duplicate detection."""
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = resized.getpixel((col, row))
            pixel_right = resized.getpixel((col + 1, row))
            diff.append(pixel_left > pixel_right)
    decimal_val = 0
    hex_str = []
    for index, val in enumerate(diff):
        if val:
            decimal_val += 2 ** (index % 4)
        if index % 4 == 3:
            hex_str.append(hex(decimal_val)[2:])
            decimal_val = 0
    return "".join(hex_str)


def validate_and_clip_box(
    x: float,
    y: float,
    w: float,
    h: float,
    img_width: int,
    img_height: int,
    padding_ratio: float = 0.05,
    min_crop_size: int = 16,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[str]]:
    """Validate box coordinates, apply padding, and clip to image bounds."""
    if w <= 0 or h <= 0:
        return None, f"Zero or negative dimensions: width={w}, height={h}"

    if x >= img_width or y >= img_height:
        return None, f"Bounding box outside image: x={x} >= W={img_width} or y={y} >= H={img_height}"

    # Calculate padding
    pad_w = w * padding_ratio
    pad_h = h * padding_ratio

    # Coordinates with padding
    x1 = max(0, int(round(x - pad_w)))
    y1 = max(0, int(round(y - pad_h)))
    x2 = min(img_width, int(round(x + w + pad_w)))
    y2 = min(img_height, int(round(y + h + pad_h)))

    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w <= 0 or crop_h <= 0:
        return None, f"Clipped box has non-positive area: {crop_w}x{crop_h}"

    if crop_w < min_crop_size or crop_h < min_crop_size:
        return None, f"Crop too small: {crop_w}x{crop_h} < min {min_crop_size}"

    return (x1, y1, x2, y2), None


def generate_vehicle_crops(
    config_path: str = "config/config.yaml",
    classes_config_path: str = "config/classes.yaml",
) -> Tuple[Path, Path]:
    """Process all annotations into cropped images, metadata, and quality reports."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg.get("dataset", {}).get("raw_dir", "data/raw"))
    crops_dir = Path(cfg.get("dataset", {}).get("crops_dir", "data/crops"))
    metadata_path = Path(cfg.get("dataset", {}).get("metadata_path", "data/metadata.csv"))
    quality_report_path = Path(cfg.get("reports", {}).get("data_quality_report_csv", "reports/data_quality_report.csv"))

    min_crop_size = int(cfg.get("dataset", {}).get("min_crop_size", 16))
    padding_ratio = float(cfg.get("dataset", {}).get("padding_ratio", 0.05))

    crops_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    quality_report_path.parent.mkdir(parents=True, exist_ok=True)

    normalizer = get_normalizer(classes_config_path)
    canonical_classes = normalizer.get_classes()
    for c in canonical_classes:
        (crops_dir / c).mkdir(parents=True, exist_ok=True)

    logger.info(f"Discovering annotations in {raw_dir}...")
    annotations = discover_and_parse_all_annotations(raw_dir, class_names=canonical_classes)
    logger.info(f"Total annotation records found: {len(annotations)}")

    metadata_rows = []
    discarded_rows = []

    seen_file_hashes: Set[str] = set()
    seen_visual_hashes: Set[str] = set()
    opened_images: Dict[Path, Image.Image] = {}

    for idx, annot in enumerate(annotations):
        img_path = annot.source_image_path
        if not img_path.exists():
            discarded_rows.append({
                "source_image": str(img_path),
                "annotation_id": annot.annotation_id,
                "raw_class": annot.category_name,
                "rejection_reason": "Missing source image file",
                "bbox": f"[{annot.x}, {annot.y}, {annot.width}, {annot.height}]",
            })
            continue

        # Check corrupted image
        if img_path not in opened_images:
            try:
                img = Image.open(img_path)
                img.verify()
                # Reopen after verify
                img = Image.open(img_path)
                opened_images[img_path] = img

                # File hash duplicate detection
                f_hash = compute_image_hash(img_path)
                if f_hash in seen_file_hashes:
                    logger.debug(f"Identical duplicate file detected: {img_path}")
                else:
                    seen_file_hashes.add(f_hash)

            except Exception as e:
                discarded_rows.append({
                    "source_image": str(img_path),
                    "annotation_id": annot.annotation_id,
                    "raw_class": annot.category_name,
                    "rejection_reason": f"Corrupted image: {str(e)}",
                    "bbox": f"[{annot.x}, {annot.y}, {annot.width}, {annot.height}]",
                })
                continue
        else:
            img = opened_images[img_path]

        img_w, img_h = img.size

        # Normalize class
        canonical_class = normalizer.normalize(annot.category_name)
        if not canonical_class or canonical_class not in canonical_classes:
            discarded_rows.append({
                "source_image": str(img_path),
                "annotation_id": annot.annotation_id,
                "raw_class": annot.category_name,
                "rejection_reason": f"Unmapped/non-target class: '{annot.category_name}'",
                "bbox": f"[{annot.x}, {annot.y}, {annot.width}, {annot.height}]",
            })
            continue

        # Validate bounding box
        coords, error_msg = validate_and_clip_box(
            x=annot.x,
            y=annot.y,
            w=annot.width,
            h=annot.height,
            img_width=img_w,
            img_height=img_h,
            padding_ratio=padding_ratio,
            min_crop_size=min_crop_size,
        )

        if coords is None:
            discarded_rows.append({
                "source_image": str(img_path),
                "annotation_id": annot.annotation_id,
                "raw_class": annot.category_name,
                "rejection_reason": error_msg,
                "bbox": f"[{annot.x}, {annot.y}, {annot.width}, {annot.height}]",
            })
            continue

        x1, y1, x2, y2 = coords
        try:
            crop_img = img.crop((x1, y1, x2, y2))
            crop_filename = f"{annot.source_image_id}_annot{annot.annotation_id}_{canonical_class}.jpg"
            crop_path = crops_dir / canonical_class / crop_filename

            # Visual hash duplicate check
            v_hash = compute_dhash(crop_img)
            if v_hash in seen_visual_hashes:
                # Still save but flag duplicate
                pass
            else:
                seen_visual_hashes.add(v_hash)

            # Save crop
            crop_img.convert("RGB").save(crop_path, "JPEG", quality=95)

            metadata_rows.append({
                "image_id": annot.source_image_id,
                "annotation_id": annot.annotation_id,
                "source_image": str(img_path.resolve()),
                "class": canonical_class,
                "crop_path": str(crop_path.resolve()),
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "original_width": img_w,
                "original_height": img_h,
                "split": annot.split_origin,
            })
        except Exception as e:
            discarded_rows.append({
                "source_image": str(img_path),
                "annotation_id": annot.annotation_id,
                "raw_class": annot.category_name,
                "rejection_reason": f"Cropping/saving error: {str(e)}",
                "bbox": f"[{annot.x}, {annot.y}, {annot.width}, {annot.height}]",
            })

    # Close PIL images
    for im in opened_images.values():
        im.close()

    # Save metadata.csv
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id", "annotation_id", "source_image", "class",
                "crop_path", "x", "y", "width", "height",
                "original_width", "original_height", "split"
            ]
        )
        writer.writeheader()
        writer.writerows(metadata_rows)
    logger.info(f"Saved {len(metadata_rows)} crop records to {metadata_path}")

    # Save data_quality_report.csv
    with open(quality_report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_image", "annotation_id", "raw_class", "rejection_reason", "bbox"]
        )
        writer.writeheader()
        writer.writerows(discarded_rows)
    logger.info(f"Recorded {len(discarded_rows)} rejected items to {quality_report_path}")

    return metadata_path, quality_report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract vehicle crops from bounding box annotations")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--classes-config", default="config/classes.yaml", help="Path to classes.yaml")
    args = parser.parse_args()

    generate_vehicle_crops(config_path=args.config, classes_config_path=args.classes_config)
