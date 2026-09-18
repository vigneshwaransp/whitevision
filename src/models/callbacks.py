"""Training callbacks and formatted progress monitor."""

import logging
from pathlib import Path
from typing import List, Optional
import yaml

logger = logging.getLogger("Callbacks")


def build_callbacks(
    checkpoint_path: Path,
    csv_log_path: Path,
    patience_early_stopping: int = 5,
    patience_reduce_lr: int = 2,
    reduce_lr_factor: float = 0.2,
    min_lr: float = 1e-7,
    monitor: str = "val_loss",
):
    """Construct standard robust callbacks suite for training and fine-tuning."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
        CSVLogger,
        Callback,
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    csv_log_path.parent.mkdir(parents=True, exist_ok=True)

    class PrettyEpochLogger(Callback):
        def __init__(self):
            super().__init__()
            self.best_val_score = float("inf")

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_loss = logs.get("val_loss", 0.0)
            if val_loss < self.best_val_score:
                self.best_val_score = val_loss
                star = " ★ (Best)"
            else:
                star = ""

            lr = float(self.model.optimizer.learning_rate)
            logger.info(
                f"Epoch {epoch + 1:02d} | "
                f"Loss: {logs.get('loss', 0.0):.4f} | "
                f"Acc: {logs.get('accuracy', 0.0) * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {logs.get('val_accuracy', 0.0) * 100:.2f}% | "
                f"LR: {lr:.2e}{star}"
            )

    callbacks: List[Callback] = [
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            mode="min",
            save_best_only=True,
            verbose=0,
        ),
        EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=patience_early_stopping,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            mode="min",
            factor=reduce_lr_factor,
            patience=patience_reduce_lr,
            min_lr=min_lr,
            verbose=1,
        ),
        CSVLogger(
            filename=str(csv_log_path),
            separator=",",
            append=False,
        ),
        PrettyEpochLogger(),
    ]

    return callbacks
