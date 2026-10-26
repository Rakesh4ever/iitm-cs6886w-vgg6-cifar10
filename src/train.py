"""
train.py
Training script for VGG6 on CIFAR-10 with support for multiple optimizers,
learning rate schedules, checkpointing, and Weights & Biases (W&B) logging.
"""

import argparse
import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf

try:
    import wandb
except ImportError:
    wandb = None

from src.dataset import get_cifar10_datasets, set_global_seed
from src.model import build_vgg6
from src.evaluate import evaluate_model


def get_optimizer(
    optimizer_name: str,
    learning_rate: float = 1e-3,
    momentum: float = 0.9,
) -> tf.keras.optimizers.Optimizer:
    """
    Creates and returns a TensorFlow/Keras optimizer based on configuration.

    Supported optimizers:
    - sgd: Standard Stochastic Gradient Descent
    - nesterov (or nesterov-sgd): SGD with Nesterov Accelerated Gradient
    - adam: Adaptive Moment Estimation
    - adagrad: Adaptive Gradient Algorithm
    - rmsprop: Root Mean Square Propagation
    - nadam: Nesterov-accelerated Adaptive Moment Estimation
    """
    name = optimizer_name.lower().strip()
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=False)
    elif name in ("nesterov", "nesterov-sgd", "nesterov_sgd"):
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    elif name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif name == "adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate, momentum=momentum)
    elif name == "nadam":
        return tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


class WandbMetricsLogger(tf.keras.callbacks.Callback):
    """
    Custom Keras callback that logs epoch-level training and validation metrics
    (train_loss, train_acc, val_loss, val_acc) to Weights & Biases in percentage (0-100)
    to match the assignment's W&B plot format.
    """

    def __init__(self, use_wandb: bool = False):
        super().__init__()
        self.use_wandb = use_wandb and (wandb is not None) and (wandb.run is not None)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(logs.get("loss", 0.0)),
                    "train_acc": float(logs.get("accuracy", 0.0)) * 100.0,
                    "val_loss": float(logs.get("val_loss", 0.0)),
                    "val_acc": float(logs.get("val_accuracy", 0.0)) * 100.0,
                },
                step=epoch + 1,
            )


def plot_training_curves(history: dict, save_path: str = "outputs/baseline_curves.png") -> None:
    """
    Plots and saves training/validation Loss and Accuracy curves side-by-side.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Accuracy Plot
    train_acc = [a * 100.0 for a in history["accuracy"]]
    val_acc = [a * 100.0 for a in history["val_accuracy"]]
    ax1.plot(epochs, train_acc, label="Train Accuracy", linewidth=2)
    ax1.plot(epochs, val_acc, label="Validation Accuracy", linewidth=2)
    ax1.set_title("VGG6 Training & Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Loss Plot
    ax2.plot(epochs, history["loss"], label="Train Loss", linewidth=2)
    ax2.plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)
    ax2.set_title("VGG6 Training & Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def train_and_evaluate(config: dict) -> dict:
    """
    Runs a complete training and evaluation experiment for a given configuration dictionary.
    """
    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    # Load datasets
    train_ds, val_ds, test_ds, _ = get_cifar10_datasets(
        batch_size=int(config.get("batch_size", 128)),
        val_split=float(config.get("val_split", 0.1)),
        augment=bool(config.get("augment", True)),
        seed=seed,
    )

    # Build VGG6 model
    model = build_vgg6(
        activation=config.get("activation", "gelu"),
        batch_norm=bool(config.get("batch_norm", True)),
        dropout_rate=float(config.get("dropout_rate", 0.2)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    # Configure optimizer and compile
    optimizer = get_optimizer(
        optimizer_name=config.get("optimizer", "nadam"),
        learning_rate=float(config.get("learning_rate", 1e-3)),
        momentum=float(config.get("momentum", 0.9)),
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    # Setup callbacks
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = config.get("checkpoint_path", "checkpoints/best_vgg6.weights.h5")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        WandbMetricsLogger(use_wandb=bool(config.get("use_wandb", False))),
    ]

    # Train model
    history_obj = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(config.get("epochs", 50)),
        callbacks=callbacks,
        verbose=1,
    )

    # Restore best validation weights before final evaluation
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)

    # Save full Keras model if requested
    model_save_path = config.get("model_save_path", "checkpoints/best_vgg6_cifar10.keras")
    if config.get("save_full_model", True):
        model.save(model_save_path)

    # Evaluate on validation and test sets
    val_metrics = evaluate_model(model, val_ds, dataset_name="Validation")
    test_metrics = evaluate_model(model, test_ds, dataset_name="Test")

    best_val_acc = max(history_obj.history["val_accuracy"]) * 100.0
    test_acc = test_metrics["accuracy"] * 100.0

    if bool(config.get("use_wandb", False)) and wandb is not None and wandb.run is not None:
        wandb.summary["best_val_acc"] = best_val_acc
        wandb.summary["test_top1_acc"] = test_acc

    # Save training curves plot
    plot_path = config.get("plot_path", "outputs/baseline_curves.png")
    plot_training_curves(history_obj.history, save_path=plot_path)

    results = {
        "config": config,
        "best_val_acc": best_val_acc,
        "test_top1_acc": test_acc,
        "test_loss": test_metrics["loss"],
        "history": {k: [float(v) for v in vals] for k, vals in history_obj.history.items()},
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/latest_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train VGG6 on CIFAR-10 in TensorFlow")
    parser.add_argument("--activation", type=str, default="gelu",
                        choices=["relu", "sigmoid", "tanh", "silu", "gelu", "leaky_relu", "selu"])
    parser.add_argument("--optimizer", type=str, default="nadam",
                        choices=["sgd", "nesterov", "adam", "adagrad", "rmsprop", "nadam"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--batch_norm", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--augment", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="iitm-cs6886w-vgg6-cifar10")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = vars(args)

    if cfg["use_wandb"] and wandb is not None:
        run_name = (
            f"act_{cfg['activation']}_bs_{cfg['batch_size']}_opt_{cfg['optimizer']}_"
            f"lr_{cfg['learning_rate']}_wd_{cfg['weight_decay']}_bn_{cfg['batch_norm']}"
        )
        wandb.init(project=cfg["wandb_project"], name=run_name, config=cfg)

    res = train_and_evaluate(cfg)
    print(f"\n[ Summary ] Best Val Accuracy: {res['best_val_acc']:.2f}% | Test Top-1 Accuracy: {res['test_top1_acc']:.2f}%")

    if cfg["use_wandb"] and wandb is not None and wandb.run is not None:
        wandb.finish()
