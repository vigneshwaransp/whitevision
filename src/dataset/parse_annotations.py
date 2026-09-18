"""Universal annotation parser for ConstructionXC7C.

Supports:
- COCO JSON (`_annotations.coco.json`, `annotations.json`)
- YOLO TXT (`<class_id> <x_center> <y_center> <width> <height>`)
- Pascal VOC XML (`<annotation><object><bndbox>...`)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

logger = logging.getLogger("AnnotationParser")


@dataclass
class AnnotationRecord:
    """Standardized representation of a single vehicle bounding box."""
    source_image_id: str
    source_image_path: Path
    annotation_id: str
    category_name: str
    x: float
    y: float
    width: float
    height: float
    image_width: int
    image_height: int
    split_origin: str


def detect_annotation_format(folder_path: Path) -> str:
    """Automatically detect the annotation format present in the given folder."""
    if not folder_path.exists():
        return "unknown"

    # Check for COCO JSON
    json_files = list(folder_path.rglob("*.json"))
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "annotations" in data and "images" in data and "categories" in data:
                    return "coco"
        except Exception:
            continue

    # Check for Pascal VOC XML
    xml_files = list(folder_path.rglob("*.xml"))
    if len(xml_files) > 0:
        return "voc"

    # Check for YOLO TXT
    txt_files = list(folder_path.rglob("*.txt"))
    for tf in txt_files:
        if tf.name in ["README.txt", "classes.txt", "notes.txt", "README.dataset.txt", "README.roboflow.txt"]:
            continue
        try:
            with open(tf, "r", encoding="utf-8") as f:
                line = f.readline().strip()
                parts = line.split()
                if len(parts) == 5 and parts[0].isdigit():
                    return "yolo"
        except Exception:
            continue

    return "unknown"


def parse_coco_json(json_path: Path, images_root: Path, split_name: str = "raw") -> List[AnnotationRecord]:
    """Parse COCO format JSON annotation file."""
    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    categories: Dict[int, str] = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
    images_info: Dict[int, dict] = {img["id"]: img for img in coco_data.get("images", [])}

    records: List[AnnotationRecord] = []
    annotations = coco_data.get("annotations", [])

    for annot in annotations:
        image_id = annot.get("image_id")
        img_meta = images_info.get(image_id)
        if not img_meta:
            continue

        file_name = img_meta.get("file_name", "")
        # Search for file relative to json_path parent, images_root, or direct
        candidate_paths = [
            json_path.parent / file_name,
            images_root / file_name,
            images_root / split_name / file_name,
        ]
        source_image_path = None
        for cp in candidate_paths:
            if cp.exists():
                source_image_path = cp
                break

        if source_image_path is None:
            # Still keep track of candidate path
            source_image_path = candidate_paths[0]

        category_id = annot.get("category_id")
        category_name = categories.get(category_id, "unknown")

        bbox = annot.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            continue

        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

        rec = AnnotationRecord(
            source_image_id=str(image_id),
            source_image_path=source_image_path,
            annotation_id=str(annot.get("id", len(records))),
            category_name=str(category_name),
            x=x,
            y=y,
            width=w,
            height=h,
            image_width=int(img_meta.get("width", 0)),
            image_height=int(img_meta.get("height", 0)),
            split_origin=split_name,
        )
        records.append(rec)

    logger.info(f"Parsed {len(records)} COCO annotations from {json_path}")
    return records


def parse_yolo_txt(
    txt_path: Path,
    image_path: Path,
    class_names: List[str],
    split_name: str = "raw",
    image_width: int = 640,
    image_height: int = 640,
) -> List[AnnotationRecord]:
    """Parse YOLO TXT bounding box format."""
    records = []
    if not txt_path.exists():
        return records

    stem = txt_path.stem
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue

        class_idx = int(parts[0])
        category_name = class_names[class_idx] if 0 <= class_idx < len(class_names) else f"class_{class_idx}"

        # YOLO is normalized: x_center, y_center, width, height
        x_center = float(parts[1]) * image_width
        y_center = float(parts[2]) * image_height
        box_w = float(parts[3]) * image_width
        box_h = float(parts[4]) * image_height

        x = x_center - (box_w / 2.0)
        y = y_center - (box_h / 2.0)

        records.append(
            AnnotationRecord(
                source_image_id=stem,
                source_image_path=image_path,
                annotation_id=f"{stem}_{idx}",
                category_name=category_name,
                x=x,
                y=y,
                width=box_w,
                height=box_h,
                image_width=image_width,
                image_height=image_height,
                split_origin=split_name,
            )
        )
    return records


def parse_voc_xml(xml_path: Path, image_path: Path, split_name: str = "raw") -> List[AnnotationRecord]:
    """Parse Pascal VOC XML annotation file."""
    records = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"Failed to parse XML {xml_path}: {e}")
        return records

    stem = xml_path.stem
    size_elem = root.find("size")
    image_width = int(size_elem.find("width").text) if size_elem is not None and size_elem.find("width") is not None else 0
    image_height = int(size_elem.find("height").text) if size_elem is not None and size_elem.find("height") is not None else 0

    for idx, obj in enumerate(root.findall("object")):
        name_elem = obj.find("name")
        category_name = name_elem.text.strip() if name_elem is not None and name_elem.text else "unknown"

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        records.append(
            AnnotationRecord(
                source_image_id=stem,
                source_image_path=image_path,
                annotation_id=f"{stem}_{idx}",
                category_name=category_name,
                x=xmin,
                y=ymin,
                width=xmax - xmin,
                height=ymax - ymin,
                image_width=image_width,
                image_height=image_height,
                split_origin=split_name,
            )
        )
    return records


def discover_and_parse_all_annotations(
    root_dir: Path,
    class_names: Optional[List[str]] = None,
) -> List[AnnotationRecord]:
    """Recursively discover and parse all annotations across raw directories."""
    all_records: List[AnnotationRecord] = []
    if not root_dir.exists():
        logger.warning(f"Root dir {root_dir} does not exist.")
        return all_records

    # 1. Look for COCO JSON files
    json_files = list(root_dir.rglob("*.json"))
    coco_files = []
    for jf in json_files:
        if jf.name in ["split_name_to_num_samples.json", "dataset_report.json"]:
            continue
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "annotations" in data and "categories" in data:
                    coco_files.append(jf)
        except Exception:
            continue

    if coco_files:
        logger.info(f"Found {len(coco_files)} COCO annotation files.")
        for cf in coco_files:
            split_name = cf.parent.name
            records = parse_coco_json(cf, cf.parent, split_name=split_name)
            all_records.extend(records)
        return all_records

    # 2. Check for VOC XML files
    xml_files = list(root_dir.rglob("*.xml"))
    if xml_files:
        logger.info(f"Found {len(xml_files)} VOC XML files.")
        for xf in xml_files:
            # find corresponding image
            image_candidates = [
                xf.with_suffix(".jpg"),
                xf.with_suffix(".jpeg"),
                xf.with_suffix(".png"),
            ]
            img_path = next((p for p in image_candidates if p.exists()), xf.with_suffix(".jpg"))
            records = parse_voc_xml(xf, img_path, split_name=xf.parent.name)
            all_records.extend(records)
        return all_records

    # 3. Check for YOLO TXT files
    txt_files = list(root_dir.rglob("*.txt"))
    valid_txts = [tf for tf in txt_files if not tf.name.startswith("README")]
    if valid_txts and class_names:
        logger.info(f"Found {len(valid_txts)} potential YOLO label files.")
        for tf in valid_txts:
            image_candidates = [
                tf.with_suffix(".jpg"),
                tf.with_suffix(".jpeg"),
                tf.with_suffix(".png"),
            ]
            img_path = next((p for p in image_candidates if p.exists()), tf.with_suffix(".jpg"))
            records = parse_yolo_txt(tf, img_path, class_names, split_name=tf.parent.name)
            all_records.extend(records)
        return all_records

    logger.warning(f"No recognized annotations found in {root_dir}")
    return all_records
