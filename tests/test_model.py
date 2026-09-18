"""Unit tests for EfficientNetB0 model construction and layer freezing."""

import numpy as np
import pytest


def test_model_shapes_and_probabilities():
    import tensorflow as tf
    from src.models.efficientnet import build_efficientnetb0, freeze_backbone, count_parameters, unfreeze_for_fine_tuning

    model = build_efficientnetb0(num_classes=8, image_size=224, dropout_rate=0.3)
    freeze_backbone(model)

    # Verify output shape
    assert model.output_shape == (None, 8)
    assert model.input_shape == (None, 224, 224, 3)

    # Check Stage 1 parameter counts
    total, train_p, non_train_p = count_parameters(model)
    assert total > 4_000_000
    # In stage 1, only head is trainable
    assert train_p < 50_000

    # Test dummy inference and softmax sum
    dummy_input = np.random.uniform(0, 255, size=(2, 224, 224, 3)).astype(np.float32)
    preds = model.predict(dummy_input, verbose=0)
    assert preds.shape == (2, 8)

    # Verify probabilities sum to 1.0
    for row in preds:
        assert np.isclose(np.sum(row), 1.0, atol=1e-5)

    # Check Stage 2 unfreezing
    unfreeze_for_fine_tuning(model, unfreeze_top_layers=30, freeze_batch_norm=True)
    _, train_p_stage2, _ = count_parameters(model)
    assert train_p_stage2 > train_p
