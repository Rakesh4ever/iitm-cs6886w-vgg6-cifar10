"""
evaluate.py
Standalone evaluation module for computing Top-1 Accuracy, Loss, and per-class metrics
for a trained VGG6 TensorFlow/Keras model on CIFAR-10.
"""

import argparse
import numpy as np
import tensorflow as tf
from src.dataset import get_cifar10_datasets, set_global_seed


CIFAR10_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    dataset_name: str = "Test",
) -> dict:
    """
    Evaluates a compiled Keras model on the given tf.data.Dataset and prints
    both overall loss and Top-1 accuracy.

    Args:
        model (tf.keras.Model): Compiled VGG6 model.
        dataset (tf.data.Dataset): Dataset to evaluate on (val_ds or test_ds).
        dataset_name (str): Label for logging ("Validation" or "Test").

    Returns:
        dict: {"loss": float, "accuracy": float}
    """
    loss, accuracy = model.evaluate(dataset, verbose=0)
    print(f"[{dataset_name} Evaluation] Loss: {loss:.4f} | Top-1 Accuracy: {accuracy * 100.0:.2f}%")
    return {"loss": float(loss), "accuracy": float(accuracy)}


def compute_per_class_accuracy(model: tf.keras.Model, dataset: tf.data.Dataset) -> dict:
    """
    Computes Top-1 accuracy for each of the 10 CIFAR-10 classes.
    """
    y_true_list = []
    y_pred_list = []

    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=-1)
        y_true_list.append(labels.numpy())
        y_pred_list.append(preds)

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    per_class_acc = {}
    for cls_idx, cls_name in enumerate(CIFAR10_CLASS_NAMES):
        mask = (y_true == cls_idx)
        cls_acc = float(np.mean(y_pred[mask] == y_true[mask])) * 100.0
        per_class_acc[cls_name] = cls_acc

    return per_class_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved VGG6 model on CIFAR-10 Test Set")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/best_vgg6_cifar10.keras",
        help="Path to the saved .keras model file",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    _, val_ds, test_ds, _ = get_cifar10_datasets(
        batch_size=args.batch_size,
        augment=False,
        seed=args.seed,
    )

    print(f"Loading trained model from: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    evaluate_model(model, val_ds, dataset_name="Validation")
    evaluate_model(model, test_ds, dataset_name="Test")

    per_class = compute_per_class_accuracy(model, test_ds)
    print("\n--- Per-Class Test Top-1 Accuracy ---")
    for cls_name, acc in per_class.items():
        print(f"  {cls_name:12s}: {acc:.2f}%")


if __name__ == "__main__":
    main()
