"""Unit tests for dataset leakage prevention and disjointness checking."""

import pytest
from src.dataset.split import check_data_leakage


def test_disjoint_splits_no_leakage():
    train_ids = {"img1", "img2", "img3", "img4"}
    val_ids = {"img5", "img6"}
    test_ids = {"img7", "img8"}

    has_leak, counts = check_data_leakage(train_ids, val_ids, test_ids)
    assert not has_leak
    assert counts["train_val_overlap"] == 0
    assert counts["train_test_overlap"] == 0
    assert counts["val_test_overlap"] == 0


def test_leakage_detection():
    # img2 is in both train and val
    train_ids = {"img1", "img2", "img3"}
    val_ids = {"img2", "img4"}
    test_ids = {"img5", "img6"}

    has_leak, counts = check_data_leakage(train_ids, val_ids, test_ids)
    assert has_leak
    assert counts["train_val_overlap"] == 1
    assert counts["train_test_overlap"] == 0


def test_cross_split_test_leakage():
    # img1 is in both train and test
    train_ids = {"img1", "img2"}
    val_ids = {"img3", "img4"}
    test_ids = {"img1", "img5"}

    has_leak, counts = check_data_leakage(train_ids, val_ids, test_ids)
    assert has_leak
    assert counts["train_test_overlap"] == 1
