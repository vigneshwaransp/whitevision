"""Unit tests for bounding-box validation, clipping, and padding."""

import pytest
from src.dataset.create_crops import validate_and_clip_box


def test_valid_box_within_image():
    # Box: x=100, y=100, w=50, h=50 in 500x500 image with 10% padding
    coords, error = validate_and_clip_box(
        x=100, y=100, w=50, h=50,
        img_width=500, img_height=500,
        padding_ratio=0.10, min_crop_size=16
    )
    assert error is None
    assert coords is not None
    x1, y1, x2, y2 = coords
    # 10% padding on 50 is 5
    assert x1 == 95
    assert y1 == 95
    assert x2 == 155
    assert y2 == 155


def test_box_clipping_at_image_boundaries():
    # Box starting at x=0, y=0 with 10% padding -> should clip to 0
    coords, error = validate_and_clip_box(
        x=5, y=5, w=100, h=100,
        img_width=200, img_height=200,
        padding_ratio=0.10, min_crop_size=16
    )
    assert error is None
    x1, y1, x2, y2 = coords
    assert x1 == 0  # clipped at 0 instead of -5
    assert y1 == 0
    assert x2 == 115
    assert y2 == 115


def test_zero_or_negative_box_rejection():
    coords, error = validate_and_clip_box(
        x=50, y=50, w=0, h=50,
        img_width=200, img_height=200
    )
    assert coords is None
    assert "Zero or negative" in error

    coords2, error2 = validate_and_clip_box(
        x=50, y=50, w=50, h=-10,
        img_width=200, img_height=200
    )
    assert coords2 is None
    assert "Zero or negative" in error2


def test_tiny_crop_rejection():
    coords, error = validate_and_clip_box(
        x=10, y=10, w=8, h=8,
        img_width=200, img_height=200,
        padding_ratio=0.0, min_crop_size=16
    )
    assert coords is None
    assert "Crop too small" in error
