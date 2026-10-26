"""
train.py
Purpose:
    Main training script for VGG6 on CIFAR-10. Supports configurable optimizers,
    learning rate scheduling, model checkpointing, local curve plotting, and
    real-time experiment tracking with Weights & Biases (W&B).
"""

# ==============================================================================
# LIBRARY IMPORTS (AND WHY WE NEED EACH ONE)
# ==============================================================================
import argparse                 # Built-in Python module to parse command-line flags
                                # (e.g., `--activation gelu --optimizer nadam --epochs 80`).
import os                       # Built-in module for creating folders (`checkpoints/`, `outputs/`)
                                # and checking file paths.
import json                     # Built-in module to save final training metrics and history
                                # to a human-readable `.json` file.
import matplotlib.pyplot as plt # Matplotlib plotting library used to generate and save
                                # Training vs. Validation Loss and Accuracy curves as PNG images.
import tensorflow as tf         # TensorFlow deep learning library used to compile, train, and save VGG6.

# Optional import for Weights & Biases (W&B) cloud experiment tracking.
# Using try/except allows the code to still run locally even if `wandb` is not installed.
try:
    import wandb
except ImportError:
    wandb = None

# Import our custom modular functions from the `src/` package
from src.dataset import get_cifar10_datasets, set_global_seed  # Data pipeline & seeding
from src.model import build_vgg6                               # VGG6 model builder
from src.evaluate import evaluate_model                        # Validation/Test evaluator


# ==============================================================================
# 1. OPTIMIZER FACTORY FUNCTION
# ==============================================================================
def get_optimizer(
    optimizer_name: str,
    learning_rate: float = 1e-3,
    momentum: float = 0.9,
) -> tf.keras.optimizers.Optimizer:
    """
    Creates and returns a TensorFlow/Keras optimizer object based on the user's choice.
    Why do we need an optimizer? The optimizer is the mathematical algorithm that updates
    the neural network's weights using the gradients computed during backpropagation.

    Supported optimizers:
    - sgd: Standard Stochastic Gradient Descent (updates weights directly along negative gradient).
    - nesterov: SGD with Nesterov Accelerated Gradient (looks ahead before computing gradient).
    - adam: Adaptive Moment Estimation (combines momentum + per-parameter adaptive learning rates).
    - adagrad: Adaptive Gradient Algorithm (scales learning rate by sum of historical squared gradients).
    - rmsprop: Root Mean Square Propagation (scales learning rate by moving average of squared gradients).
    - nadam: Adam + Nesterov momentum (often achieves fastest convergence and highest accuracy).
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


# ==============================================================================
# 2. CUSTOM WEIGHTS & BIASES (W&B) LOGGING CALLBACK
# ==============================================================================
class WandbMetricsLogger(tf.keras.callbacks.Callback):
    """
    Custom Keras Callback that runs automatically at the end of every training epoch.
    It sends `train_loss`, `train_acc`, `val_loss`, and `val_acc` (scaled to 0-100%)
    to your Weights & Biases dashboard so your charts update live in the browser.
    """

    def __init__(self, use_wandb: bool = False):
        super().__init__()
        # Only log if `--use_wandb` was passed AND an active W&B run is initialized
        self.use_wandb = use_wandb and (wandb is not None) and (wandb.run is not None)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.use_wandb:
            wandb.log(
                {
                    "train_loss": float(logs.get("loss", 0.0)),
                    "train_acc": float(logs.get("accuracy", 0.0)) * 100.0,
                    "val_loss": float(logs.get("val_loss", 0.0)),
                    "val_acc": float(logs.get("val_accuracy", 0.0)) * 100.0,
                },
                step=epoch + 1,
            )


# ==============================================================================
# 3. PLOTTING TRAINING & VALIDATION CURVES
# ==============================================================================
def plot_training_curves(history: dict, save_path: str = "outputs/baseline_curves.png") -> None:
    """
    Creates a side-by-side plot of Training vs. Validation Accuracy (left) and
    Training vs. Validation Cross-Entropy Loss (right) and saves it as a high-res PNG.
    """
    # Ensure the `outputs/` folder exists before saving the image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)

    # Create a figure with 1 row and 2 columns of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left Subplot: Accuracy (%) over Epochs
    train_acc = [a * 100.0 for a in history["accuracy"]]
    val_acc = [a * 100.0 for a in history["val_accuracy"]]
    ax1.plot(epochs, train_acc, label="Train Accuracy", linewidth=2)
    ax1.plot(epochs, val_acc, label="Validation Accuracy", linewidth=2)
    ax1.set_title("VGG6 Training & Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Right Subplot: Cross-Entropy Loss over Epochs
    ax2.plot(epochs, history["loss"], label="Train Loss", linewidth=2)
    ax2.plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)
    ax2.set_title("VGG6 Training & Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    # Adjust spacing, save to disk at 200 DPI, and close the figure to free RAM
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# ==============================================================================
# 4. CORE TRAINING & EVALUATION WORKFLOW
# ==============================================================================
def train_and_evaluate(config: dict) -> dict:
    """
    Executes a complete end-to-end experiment given a hyperparameter dictionary `config`:
      1. Seeds all random generators
      2. Builds the CIFAR-10 `tf.data` pipelines
      3. Builds and compiles the VGG6 model
      4. Trains the model with checkpointing and learning-rate decay
      5. Restores the best validation weights and evaluates on the Test set
      6. Saves plots and JSON metrics
    """
    # Step 1: Set global random seed for reproducibility
    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    # Step 2: Load Training (45k), Validation (5k), and Test (10k) datasets
    train_ds, val_ds, test_ds, _ = get_cifar10_datasets(
        batch_size=int(config.get("batch_size", 128)),
        val_split=float(config.get("val_split", 0.1)),
        augment=bool(config.get("augment", True)),
        seed=seed,
    )

    # Step 3: Build the VGG6 neural network with the chosen activation and regularization
    model = build_vgg6(
        activation=config.get("activation", "gelu"),
        batch_norm=bool(config.get("batch_norm", True)),
        dropout_rate=float(config.get("dropout_rate", 0.2)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    # Step 4: Instantiate the chosen optimizer
    optimizer = get_optimizer(
        optimizer_name=config.get("optimizer", "nadam"),
        learning_rate=float(config.get("learning_rate", 1e-3)),
        momentum=float(config.get("momentum", 0.9)),
    )

    # Step 5: Compile the model with SparseCategoricalCrossentropy loss
    # (used when labels are integers 0..9 rather than one-hot vectors)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    # Step 6: Configure training callbacks
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = config.get("checkpoint_path", "checkpoints/best_vgg6.weights.h5")

    callbacks = [
        # Save model weights whenever validation accuracy reaches a new high
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        # Halve the learning rate (`factor=0.5`) if validation loss stops improving for 5 epochs
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        # Log epoch metrics to Weights & Biases if enabled
        WandbMetricsLogger(use_wandb=bool(config.get("use_wandb", False))),
    ]

    # Step 7: Run the training loop across the specified number of epochs
    history_obj = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(config.get("epochs", 50)),
        callbacks=callbacks,
        verbose=1,
    )

    # Step 8: Reload the best validation weights (so we evaluate the best epoch, not the last epoch)
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)

    # Step 9: Save the complete trained Keras model (`.keras` file) for submission/deployment
    model_save_path = config.get("model_save_path", "checkpoints/best_vgg6_cifar10.keras")
    if config.get("save_full_model", True):
        model.save(model_save_path)

    # Step 10: Evaluate final model on Validation and Test sets
    val_metrics = evaluate_model(model, val_ds, dataset_name="Validation")
    test_metrics = evaluate_model(model, test_ds, dataset_name="Test")

    best_val_acc = max(history_obj.history["val_accuracy"]) * 100.0
    test_acc = test_metrics["accuracy"] * 100.0

    # Record summary metrics in W&B if active
    if bool(config.get("use_wandb", False)) and wandb is not None and wandb.run is not None:
        wandb.summary["best_val_acc"] = best_val_acc
        wandb.summary["test_top1_acc"] = test_acc

    # Step 11: Generate and save training/validation loss & accuracy plot
    plot_path = config.get("plot_path", "outputs/baseline_curves.png")
    plot_training_curves(history_obj.history, save_path=plot_path)

    # Step 12: Save structured experiment results to `outputs/latest_results.json`
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


# ==============================================================================
# 5. COMMAND-LINE ARGUMENT PARSER
# ==============================================================================
def parse_args():
    """
    Defines all command-line flags so you can easily run different experiments from Colab
    without editing the Python code.
    """
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

    # If `--use_wandb` flag is passed, initialize a W&B run with a descriptive name
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
