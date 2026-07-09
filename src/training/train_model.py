"""Train the baseline CNN model on MNIST."""

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.data.data_loader import get_dataset_summary, load_mnist_data
from src.models.cnn_model import build_baseline_cnn
from src.preprocessing.image_preprocessor import preprocess_mnist_data
from src.utils.logging_config import setup_logger
from src.utils.paths import IMAGES_DIR, MODELS_DIR, REPORTS_DIR


logger = setup_logger(__name__)

MODEL_PATH = MODELS_DIR / "mnist_cnn_baseline.keras"
HISTORY_PATH = REPORTS_DIR / "training_history.csv"
SUMMARY_PATH = REPORTS_DIR / "training_summary.json"
ACCURACY_PLOT_PATH = IMAGES_DIR / "evaluation" / "training_accuracy.png"
LOSS_PLOT_PATH = IMAGES_DIR / "evaluation" / "training_loss.png"


def ensure_output_dirs() -> None:
    """Create output directories required by the training workflow."""
    for directory in [MODELS_DIR, REPORTS_DIR, IMAGES_DIR / "evaluation"]:
        directory.mkdir(parents=True, exist_ok=True)


def plot_training_metric(
    history: pd.DataFrame,
    train_metric: str,
    validation_metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Save a training curve for a metric and its validation counterpart.

    Args:
        history: Training history as a dataframe.
        train_metric: Training metric column name.
        validation_metric: Validation metric column name.
        title: Plot title.
        ylabel: Y-axis label.
        output_path: Destination image path.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history[train_metric], label=train_metric)
    plt.plot(history[validation_metric], label=validation_metric)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_baseline_model(epochs: int = 6, batch_size: int = 128) -> Dict[str, Any]:
    """Train the baseline MNIST CNN and save artifacts.

    Args:
        epochs: Maximum number of training epochs.
        batch_size: Number of samples per optimization batch.

    Returns:
        Training summary dictionary.
    """
    ensure_output_dirs()
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    dataset_summary = get_dataset_summary(x_train, y_train, x_test, y_test)
    x_train_processed, y_train_processed, _, _ = preprocess_mnist_data(
        x_train,
        y_train,
        x_test,
        y_test,
    )

    model = build_baseline_cnn()
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True,
            mode="max",
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
    ]

    logger.info("Starting CNN training for up to %s epochs", epochs)
    history_obj = model.fit(
        x_train_processed,
        y_train_processed,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    history = pd.DataFrame(history_obj.history)
    history.to_csv(HISTORY_PATH, index=False)
    plot_training_metric(
        history,
        train_metric="accuracy",
        validation_metric="val_accuracy",
        title="Training and Validation Accuracy",
        ylabel="Accuracy",
        output_path=ACCURACY_PLOT_PATH,
    )
    plot_training_metric(
        history,
        train_metric="loss",
        validation_metric="val_loss",
        title="Training and Validation Loss",
        ylabel="Loss",
        output_path=LOSS_PLOT_PATH,
    )

    if not MODEL_PATH.exists():
        model.save(MODEL_PATH)

    best_epoch = int(history["val_accuracy"].idxmax() + 1)
    summary: Dict[str, Any] = {
        "model_path": str(MODEL_PATH.relative_to(MODEL_PATH.parents[1])),
        "epochs_requested": epochs,
        "epochs_completed": int(len(history)),
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        "final_training_accuracy": float(history["accuracy"].iloc[-1]),
        "final_validation_accuracy": float(history["val_accuracy"].iloc[-1]),
        "best_validation_accuracy": float(history["val_accuracy"].max()),
        "final_training_loss": float(history["loss"].iloc[-1]),
        "final_validation_loss": float(history["val_loss"].iloc[-1]),
        "dataset_summary": dataset_summary,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Training complete. Model saved to %s", MODEL_PATH)
    return summary


def main() -> None:
    """Run the baseline CNN training workflow."""
    train_baseline_model()


if __name__ == "__main__":
    main()
