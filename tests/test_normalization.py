"""Unit tests for class normalization and synonym mapping."""

import pytest
from src.dataset.normalization import ClassNormalizer, get_normalizer


def test_canonical_classes_count():
    norm = get_normalizer("config/classes.yaml")
    classes = norm.get_classes()
    assert len(classes) == 8
    expected = [
        "bulldozer", "dump_truck", "excavator", "grader",
        "loader", "mixer_truck", "mobile_crane", "roller"
    ]
    assert sorted(classes) == sorted(expected)


@pytest.mark.parametrize("raw_input,expected_canonical", [
    ("dump truck", "dump_truck"),
    ("dump_truck", "dump_truck"),
    ("dump-truck", "dump_truck"),
    ("tipper", "dump_truck"),
    ("dumper", "dump_truck"),
    ("motor grader", "grader"),
    ("road grader", "grader"),
    ("grader", "grader"),
    ("wheel loader", "loader"),
    ("front loader", "loader"),
    ("payloader", "loader"),
    ("loader", "loader"),
    ("road roller", "roller"),
    ("roller", "roller"),
    ("steam roller", "roller"),
    ("concrete mixer", "mixer_truck"),
    ("mixer truck", "mixer_truck"),
    ("cement mixer", "mixer_truck"),
    ("bulldozer", "bulldozer"),
    ("dozer", "bulldozer"),
    ("excavator", "excavator"),
    ("digger", "excavator"),
    ("mobile crane", "mobile_crane"),
    ("crane", "mobile_crane"),
])
def test_synonym_normalization(raw_input, expected_canonical):
    norm = get_normalizer("config/classes.yaml")
    result = norm.normalize(raw_input)
    assert result == expected_canonical, f"Failed to normalize '{raw_input}', got '{result}' instead of '{expected_canonical}'"


def test_invalid_class_returns_none():
    norm = get_normalizer("config/classes.yaml")
    assert norm.normalize("airplane") is None
    assert norm.normalize("bicycle") is None
    assert norm.normalize("") is None
