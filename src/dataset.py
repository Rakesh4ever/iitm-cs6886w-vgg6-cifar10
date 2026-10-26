"""
dataset.py
Purpose:
    Loads, preprocesses, normalizes, and augments the CIFAR-10 image dataset
    using TensorFlow and the high-performance `tf.data` input pipeline API.
"""

# ==============================================================================
# LIBRARY IMPORTS (AND WHY WE NEED EACH ONE)
# ==============================================================================
import os                # Built-in Python module to interact with the operating system
                         # (used here to set environment variables for reproducibility).
import random            # Built-in Python random number generator (seeded so random
                         # operations produce the exact same sequence every run).
import numpy as np       # NumPy library for fast array math and deterministic index
                         # shuffling when splitting training and validation sets.
import tensorflow as tf  # Google's deep learning framework used to download CIFAR-10,
                         # manipulate image tensors, and build GPU-accelerated data pipelines.


# ==============================================================================
# 1. REPRODUCIBILITY: SETTING GLOBAL RANDOM SEEDS
# ==============================================================================
def set_global_seed(seed: int = 42) -> None:
    """
    Sets random seeds across Python, NumPy, and TensorFlow so that every time
    you run the code in Colab, you get repeatable weight initializations and data splits.

    Args:
        seed (int): Any fixed integer (default: 42) used to initialize random generators.
    """
    # Fix Python's internal hash seed so dictionary/set iteration order is deterministic
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Instruct TensorFlow GPU kernels to use deterministic algorithms where available
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    # Seed Python's built-in random module
    random.seed(seed)
    # Seed NumPy's legacy global random number generator
    np.random.seed(seed)
    # Seed TensorFlow's graph and eager random number generators
    tf.random.set_seed(seed)
    # Convenience helper in Keras that synchronizes Python, NumPy, and TF random states
    tf.keras.utils.set_random_seed(seed)


# ==============================================================================
# 2. IMAGE NORMALIZATION CONSTANTS & FUNCTION
# ==============================================================================
# Pre-computed RGB channel-wise mean and standard deviation across the 50,000
# CIFAR-10 training images (after scaling pixel values from [0, 255] to [0.0, 1.0]).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)  # Mean for (Red, Green, Blue) channels
CIFAR10_STD = (0.2470, 0.2435, 0.2616)   # Std Dev for (Red, Green, Blue) channels


def normalize_image(image: tf.Tensor, label: tf.Tensor):
    """
    Converts raw uint8 image pixels [0, 255] to float32 [0.0, 1.0] and applies
    channel-wise Z-score standardization: x_normalized = (x - mean) / std.
    Why? Centering inputs around 0 with unit variance prevents exploding/vanishing
    gradients and helps neural network optimizers converge much faster.
    """
    # Step 1: Convert integer pixels (0 to 255) to 32-bit floating point numbers in [0.0, 1.0]
    image = tf.cast(image, tf.float32) / 255.0

    # Step 2: Create 3D broadcastable constant tensors of shape (1, 1, 3) for RGB mean and std
    mean = tf.constant(CIFAR10_MEAN, dtype=tf.float32, shape=[1, 1, 3])
    std = tf.constant(CIFAR10_STD, dtype=tf.float32, shape=[1, 1, 3])

    # Step 3: Subtract channel mean and divide by channel standard deviation
    image = (image - mean) / std

    # Step 4: Convert label from shape (1,) to a flat 32-bit integer scalar (0 through 9)
    label = tf.cast(tf.squeeze(label), tf.int32)
    return image, label


# ==============================================================================
# 3. DATA AUGMENTATION FUNCTION (APPLIED ONLY TO TRAINING IMAGES)
# ==============================================================================
def augment_image(image: tf.Tensor, label: tf.Tensor):
    """
    Applies random spatial transformations to training images on the fly so the
    model sees slightly varied versions of each image every epoch, reducing overfitting:
      1. Reflection padding by 4 pixels on top/bottom/left/right (32x32x3 -> 40x40x3)
      2. Random crop back to the original size (32x32x3)
      3. Random horizontal mirror flip with 50% probability
    """
    # Pad height by [4, 4], width by [4, 4], and keep RGB channels [0, 0] unchanged
    image = tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode="REFLECT")

    # Randomly slice a 32x32x3 patch out of the padded 40x40x3 image
    image = tf.image.random_crop(image, size=[32, 32, 3])

    # Randomly flip the image left-to-right 50% of the time (e.g. a car facing left vs right)
    image = tf.image.random_flip_left_right(image)
    return image, label


# ==============================================================================
# 4. BUILDING THE COMPLETE TRAIN / VALIDATION / TEST PIPELINES
# ==============================================================================
def get_cifar10_datasets(
    batch_size: int = 128,
    val_split: float = 0.1,
    augment: bool = True,
    seed: int = 42,
):
    """
    Downloads CIFAR-10, splits the 50,000 training images into Training (45,000) and
    Validation (5,000) sets, and wraps Train/Val/Test into fast `tf.data.Dataset` pipelines.

    Args:
        batch_size (int): Number of images processed together in a single training step.
        val_split (float): Fraction of training data held out for validation (0.1 = 10%).
        augment (bool): If True, applies random crop + flip augmentation to training images.
        seed (int): Random seed for reproducible train/validation splitting and shuffling.

    Returns:
        tuple: (train_ds, val_ds, test_ds, info_dict)
    """
    # Ensure all random operations use our fixed seed
    set_global_seed(seed)

    # Download CIFAR-10 from Keras: 50,000 train images and 10,000 test images of shape (32, 32, 3)
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Calculate exact sample counts for training (45,000) and validation (5,000)
    num_total_train = len(x_train_full)
    num_val = int(num_total_train * val_split)
    num_train = num_total_train - num_val

    # Create a seeded NumPy random generator to shuffle indices 0..49,999 reproducibly
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_total_train)
    train_idx, val_idx = indices[:num_train], indices[num_train:]

    # Slice the full training arrays into separate training and validation arrays
    x_train, y_train = x_train_full[train_idx], y_train_full[train_idx]
    x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]

    # AUTOTUNE lets TensorFlow dynamically choose the optimal number of CPU threads
    autotune = tf.data.AUTOTUNE

    # --- Build Training Data Pipeline ---
    # 1. Convert NumPy arrays into a TensorFlow Dataset of (image, label) pairs
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # 2. Shuffle training samples at the start of every epoch so batches vary each epoch
    train_ds = train_ds.shuffle(buffer_size=num_train, seed=seed, reshuffle_each_iteration=True)
    # 3. Normalize pixel values in parallel across CPU cores
    train_ds = train_ds.map(normalize_image, num_parallel_calls=autotune)
    # 4. Apply random data augmentation ONLY to the training set (never to val or test!)
    if augment:
        train_ds = train_ds.map(augment_image, num_parallel_calls=autotune)
    # 5. Group individual images into mini-batches (e.g., 128 images per batch) and
    #    prefetch the next batch in the background while the GPU trains on the current batch
    train_ds = train_ds.batch(batch_size).prefetch(autotune)

    # --- Build Validation Data Pipeline (No Shuffling, No Augmentation) ---
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.map(normalize_image, num_parallel_calls=autotune)
    val_ds = val_ds.batch(batch_size).prefetch(autotune)

    # --- Build Test Data Pipeline (No Shuffling, No Augmentation) ---
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(normalize_image, num_parallel_calls=autotune)
    test_ds = test_ds.batch(batch_size).prefetch(autotune)

    # Store dataset metadata in a dictionary for reference/logging
    info = {
        "num_train": num_train,
        "num_val": num_val,
        "num_test": len(x_test),
        "num_classes": 10,
        "input_shape": (32, 32, 3),
    }

    return train_ds, val_ds, test_ds, info
