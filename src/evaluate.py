"""
evaluate.py
Purpose:
    Standalone evaluation module that loads a saved VGG6 TensorFlow/Keras model
    and computes overall Loss, Top-1 Accuracy, and per-class Top-1 Accuracy
    across all 10 CIFAR-10 object categories.
"""

# ==============================================================================
# LIBRARY IMPORTS (AND WHY WE NEED EACH ONE)
# ==============================================================================
import argparse          # Built-in Python module to parse command-line arguments
                         # (e.g., `--model_path`, `--batch_size`, `--seed`).
import numpy as np       # NumPy library used to find the highest-probability class (`argmax`)
                         # and calculate per-class accuracy percentages using boolean masks.
import tensorflow as tf  # TensorFlow library used to load the saved `.keras` model and run inference.

# Import our custom dataset loader and reproducibility seed helper from `src/dataset.py`
from src.dataset import get_cifar10_datasets, set_global_seed


# Human-readable names corresponding to CIFAR-10 integer class labels 0 through 9
CIFAR10_CLASS_NAMES = [
    "airplane",    # Class 0
    "automobile",  # Class 1
    "bird",        # Class 2
    "cat",         # Class 3
    "deer",        # Class 4
    "dog",         # Class 5
    "frog",        # Class 6
    "horse",       # Class 7
    "ship",        # Class 8
    "truck",       # Class 9
]


# ==============================================================================
# 1. OVERALL DATASET EVALUATION (LOSS & TOP-1 ACCURACY)
# ==============================================================================
def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    dataset_name: str = "Test",
) -> dict:
    """
    Evaluates a compiled Keras model on a `tf.data.Dataset` (Validation or Test)
    and prints both Cross-Entropy Loss and Top-1 Accuracy.

    Args:
        model (tf.keras.Model): Compiled VGG6 Keras model.
        dataset (tf.data.Dataset): Dataset pipeline to evaluate on (`val_ds` or `test_ds`).
        dataset_name (str): Display label for console output ("Validation" or "Test").

    Returns:
        dict: Dictionary containing `{"loss": float, "accuracy": float}` (accuracy in 0.0-1.0).
    """
    # Run forward-pass evaluation across all batches in the dataset (verbose=0 suppresses progress bar)
    loss, accuracy = model.evaluate(dataset, verbose=0)
    print(f"[{dataset_name} Evaluation] Loss: {loss:.4f} | Top-1 Accuracy: {accuracy * 100.0:.2f}%")
    return {"loss": float(loss), "accuracy": float(accuracy)}


# ==============================================================================
# 2. GRANULAR PER-CLASS TOP-1 ACCURACY BREAKDOWN
# ==============================================================================
def compute_per_class_accuracy(model: tf.keras.Model, dataset: tf.data.Dataset) -> dict:
    """
    Computes the Top-1 accuracy separately for each of the 10 CIFAR-10 classes
    so we can see which categories (e.g., ship, automobile) are easiest and which
    (e.g., cat vs. dog) are most challenging for the model.
    """
    y_true_list = []  # Will store ground-truth integer labels for every batch
    y_pred_list = []  # Will store predicted integer class indices for every batch

    # Loop through each batch of (images, labels) in the test dataset
    for images, labels in dataset:
        # Predict class probability distributions of shape (batch_size, 10)
        probs = model.predict(images, verbose=0)
        # Pick the index (0..9) with the highest predicted probability for each image
        preds = np.argmax(probs, axis=-1)
        # Append batch ground-truth labels and predictions as NumPy arrays
        y_true_list.append(labels.numpy())
        y_pred_list.append(preds)

    # Concatenate all batch arrays into single 1D arrays of length 10,000
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    # Calculate accuracy for each class 0..9 using a boolean mask
    per_class_acc = {}
    for cls_idx, cls_name in enumerate(CIFAR10_CLASS_NAMES):
        # Select only test samples whose true label equals `cls_idx` (1,000 samples per class)
        mask = (y_true == cls_idx)
        # Compute percentage of those samples where predicted label matches true label
        cls_acc = float(np.mean(y_pred[mask] == y_true[mask])) * 100.0
        per_class_acc[cls_name] = cls_acc

    return per_class_acc


# ==============================================================================
# 3. COMMAND-LINE ENTRY POINT FOR STANDALONE EVALUATION
# ==============================================================================
def main():
    """
    Allows running standalone evaluation from the terminal or a Colab cell via:
    `!python -m src.evaluate --model_path checkpoints/best_vgg6_cifar10.keras`
    """
    parser = argparse.ArgumentParser(description="Evaluate saved VGG6 model on CIFAR-10 Test Set")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/best_vgg6_cifar10.keras",
        help="Path to the saved .keras model file",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split")
    args = parser.parse_args()

    # Set global random seed so the validation split is identical to training
    set_global_seed(args.seed)

    # Load validation and test datasets (with augmentation disabled)
    _, val_ds, test_ds, _ = get_cifar10_datasets(
        batch_size=args.batch_size,
        augment=False,
        seed=args.seed,
    )

    # Load the complete saved Keras model (architecture + trained weights + compilation state)
    print(f"Loading trained model from: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    # Evaluate overall accuracy on Validation (5,000 images) and Test (10,000 images)
    evaluate_model(model, val_ds, dataset_name="Validation")
    evaluate_model(model, test_ds, dataset_name="Test")

    # Compute and print accuracy for each of the 10 classes
    per_class = compute_per_class_accuracy(model, test_ds)
    print("\n--- Per-Class Test Top-1 Accuracy ---")
    for cls_name, acc in per_class.items():
        print(f"  {cls_name:12s}: {acc:.2f}%")


if __name__ == "__main__":
    main()
