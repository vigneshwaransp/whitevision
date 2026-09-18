"""Unit tests for annotation parsing across COCO, YOLO, and VOC."""

import json
from pathlib import Path
import pytest
from src.dataset.parse_annotations import (
    detect_annotation_format,
    parse_coco_json,
    parse_voc_xml,
    parse_yolo_txt,
)


def test_coco_json_parsing(tmp_path: Path):
    coco_dict = {
        "images": [{"id": 1, "file_name": "excavator_01.jpg", "width": 640, "height": 480}],
        "categories": [{"id": 10, "name": "excavator"}],
        "annotations": [
            {
                "id": 101,
                "image_id": 1,
                "category_id": 10,
                "bbox": [50.0, 60.0, 200.0, 150.0],
                "area": 30000.0,
            }
        ],
    }
    json_path = tmp_path / "_annotations.coco.json"
    with open(json_path, "w") as f:
        json.dump(coco_dict, f)

    records = parse_coco_json(json_path, tmp_path, split_name="test_split")
    assert len(records) == 1
    r = records[0]
    assert r.source_image_id == "1"
    assert r.category_name == "excavator"
    assert r.x == 50.0
    assert r.y == 60.0
    assert r.width == 200.0
    assert r.height == 150.0


def test_yolo_txt_parsing(tmp_path: Path):
    txt_path = tmp_path / "truck_01.txt"
    # YOLO format: class_id x_center y_center width height (normalized)
    with open(txt_path, "w") as f:
        f.write("1 0.5 0.5 0.4 0.3\n")

    img_path = tmp_path / "truck_01.jpg"
    classes = ["bulldozer", "dump_truck"]
    records = parse_yolo_txt(
        txt_path=txt_path,
        image_path=img_path,
        class_names=classes,
        image_width=1000,
        image_height=800,
    )
    assert len(records) == 1
    r = records[0]
    assert r.category_name == "dump_truck"
    # center is (500, 400), w is 400, h is 240 -> x = 300, y = 280
    assert r.x == 300.0
    assert r.y == 280.0
    assert r.width == 400.0
    assert r.height == 240.0


def test_format_auto_detection(tmp_path: Path):
    coco_folder = tmp_path / "coco_dir"
    coco_folder.mkdir()
    with open(coco_folder / "_annotations.coco.json", "w") as f:
        json.dump({"images": [], "annotations": [], "categories": []}, f)

    assert detect_annotation_format(coco_folder) == "coco"
