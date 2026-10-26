"""
dataset.py
Module for loading, preprocessing, normalizing, and augmenting the CIFAR-10 dataset
using TensorFlow and tf.data API.
"""

import os
import random
import numpy as np
import tensorflow as tf


def set_global_seed(seed: int = 42) -> None:
    """
    Sets random seeds across Python, NumPy, and TensorFlow for full reproducibility.
    
    Args:
        seed (int): The random seed value (default: 42).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


# CIFAR-10 channel-wise mean and standard deviation (computed over the 50,000 training images)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def normalize_image(image: tf.Tensor, label: tf.Tensor):
    """
    Scales pixel values to [0, 1] and applies channel-wise Z-score standardization
    using CIFAR-10 training set statistics.
    """
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant(CIFAR10_MEAN, dtype=tf.float32, shape=[1, 1, 3])
    std = tf.constant(CIFAR10_STD, dtype=tf.float32, shape=[1, 1, 3])
    image = (image - mean) / std
    label = tf.cast(tf.squeeze(label), tf.int32)
    return image, label


def augment_image(image: tf.Tensor, label: tf.Tensor):
    """
    Applies standard CIFAR-10 data augmentation:
    1. Reflection/zero padding by 4 pixels on each side (32x32 -> 40x40)
    2. Random crop back to 32x32
    3. Random horizontal flip (probability = 0.5)
    """
    # Pad 4 pixels on top, bottom, left, right
    image = tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode="REFLECT")
    # Random crop back to (32, 32, 3)
    image = tf.image.random_crop(image, size=[32, 32, 3])
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    return image, label


def get_cifar10_datasets(
    batch_size: int = 128,
    val_split: float = 0.1,
    augment: bool = True,
    seed: int = 42,
):
    """
    Loads CIFAR-10, splits into train/validation/test sets, and builds optimized
    tf.data.Dataset pipelines with normalization and optional data augmentation.

    Args:
        batch_size (int): Batch size for training and evaluation.
        val_split (float): Fraction of training data reserved for validation (default: 0.1 -> 5,000 samples).
        augment (bool): Whether to apply data augmentation to the training set.
        seed (int): Random seed for reproducible train/val splitting and shuffling.

    Returns:
        tuple: (train_ds, val_ds, test_ds, info_dict)
    """
    set_global_seed(seed)

    # Load raw CIFAR-10 data from Keras datasets
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    num_total_train = len(x_train_full)
    num_val = int(num_total_train * val_split)
    num_train = num_total_train - num_val

    # Shuffle indices deterministically before splitting
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_total_train)
    train_idx, val_idx = indices[:num_train], indices[num_train:]

    x_train, y_train = x_train_full[train_idx], y_train_full[train_idx]
    x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]

    autotune = tf.data.AUTOTUNE

    # Build Training Pipeline
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=num_train, seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.map(normalize_image, num_parallel_calls=autotune)
    if augment:
        train_ds = train_ds.map(augment_image, num_parallel_calls=autotune)
    train_ds = train_ds.batch(batch_size).prefetch(autotune)

    # Build Validation Pipeline
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.map(normalize_image, num_parallel_calls=autotune)
    val_ds = val_ds.batch(batch_size).prefetch(autotune)

    # Build Test Pipeline
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(normalize_image, num_parallel_calls=autotune)
    test_ds = test_ds.batch(batch_size).prefetch(autotune)

    info = {
        "num_train": num_train,
        "num_val": num_val,
        "num_test": len(x_test),
        "num_classes": 10,
        "input_shape": (32, 32, 3),
    }

    return train_ds, val_ds, test_ds, info
