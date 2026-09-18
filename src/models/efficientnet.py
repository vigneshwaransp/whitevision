"""EfficientNetB0 model builder and layer freeze/unfreeze orchestrator."""

import logging
from typing import Optional, Tuple
import yaml

logger = logging.getLogger("EfficientNetModel")


def build_efficientnetb0(
    config_path: str = "config/config.yaml",
    num_classes: int = 8,
    image_size: int = 224,
    dropout_rate: float = 0.3,
):
    """Construct EfficientNetB0 classification model with ImageNet pretraining."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import EfficientNetB0

    inputs = layers.Input(shape=(image_size, image_size, 3), name="image_input")

    # EfficientNetB0 Backbone (name defaults to 'efficientnetb0' matching official weights)
    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )

    # Initially freeze backbone for Stage 1 feature extraction
    backbone.trainable = False

    # Classification Head
    x = backbone(inputs)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ConstructionVehicleClassifier")
    return model


def _get_backbone(model):
    """Retrieve backbone layer by standard names or fallback."""
    for name in ["efficientnetb0", "efficientnetb0_backbone"]:
        try:
            return model.get_layer(name)
        except ValueError:
            pass
    return None


def freeze_backbone(model):
    """Freeze all layers in the EfficientNetB0 backbone for Stage 1 training."""
    from tensorflow.keras import layers
    backbone = _get_backbone(model)
    if backbone is not None:
        backbone.trainable = False
    else:
        head_names = {"global_avg_pool", "head_batch_norm", "head_dropout", "predictions"}
        for layer in model.layers:
            if layer.name not in head_names:
                layer.trainable = False
    logger.info("Backbone frozen for Stage 1 feature extraction.")


def unfreeze_for_fine_tuning(
    model,
    unfreeze_top_layers: int = 30,
    freeze_batch_norm: bool = True,
):
    """Unfreeze top layers of backbone while keeping BatchNormalization frozen for Stage 2."""
    from tensorflow.keras import layers
    backbone = _get_backbone(model)
    if backbone is not None:
        target_layers = backbone.layers
        backbone.trainable = True
    else:
        head_names = {"global_avg_pool", "head_batch_norm", "head_dropout", "predictions"}
        target_layers = [l for l in model.layers if l.name not in head_names]

    total_layers = len(target_layers)
    cutoff = max(0, total_layers - unfreeze_top_layers)

    for i, layer in enumerate(target_layers):
        if i < cutoff:
            layer.trainable = False
        else:
            layer.trainable = True

        # Keep BatchNormalization layers frozen to protect learned statistics
        if freeze_batch_norm and isinstance(layer, (layers.BatchNormalization, layers.LayerNormalization)):
            layer.trainable = False

    logger.info(
        f"Stage 2 fine-tuning: Unfroze top {unfreeze_top_layers} layers of backbone "
        f"(total {total_layers}). BatchNormalization frozen: {freeze_batch_norm}."
    )


def count_parameters(model) -> Tuple[int, int, int]:
    """Calculate total, trainable, and non-trainable parameter counts."""
    import numpy as np
    trainable_count = int(np.sum([np.prod(p.shape) for p in model.trainable_weights]))
    non_trainable_count = int(np.sum([np.prod(p.shape) for p in model.non_trainable_weights]))
    total_count = trainable_count + non_trainable_count
    return total_count, trainable_count, non_trainable_count
