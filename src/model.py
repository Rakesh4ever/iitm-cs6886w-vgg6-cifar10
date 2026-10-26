"""
model.py
Purpose:
    Defines the configurable VGG6 Convolutional Neural Network (CNN) architecture
    in TensorFlow/Keras for classifying 32x32 RGB images from CIFAR-10 into 10 classes.
    Allows switching activation functions, Batch Normalization, Dropout, and L2 Weight Decay.
"""

# ==============================================================================
# LIBRARY IMPORTS (AND WHY WE NEED EACH ONE)
# ==============================================================================
import tensorflow as tf  # Core TensorFlow library for building and running neural networks.
from tensorflow.keras import (
    layers,        # Provides neural network building blocks (Conv2D, Dense, BatchNormalization, Dropout, etc.).
    models,        # Provides `models.Model` to assemble input and output tensors into a trainable Keras Model.
    regularizers,  # Provides L2 weight decay (`regularizers.l2`) to penalize overly large weights and curb overfitting.
)


# ==============================================================================
# 1. ACTIVATION FUNCTION FACTORY
# ==============================================================================
def get_activation_layer(activation_name: str) -> layers.Layer:
    """
    Returns the requested non-linear Keras activation layer based on a string name.
    Why do we need activation functions? Without non-linear activations, stacking
    multiple convolution or dense layers would mathematically collapse into a single
    linear transformation and could never learn complex visual patterns!

    Supported activations:
    - relu: Rectified Linear Unit -> max(0, x). Fast, standard baseline.
    - sigmoid: Logistic Sigmoid -> 1 / (1 + exp(-x)). Squashes values to (0, 1).
    - tanh: Hyperbolic Tangent -> squashes values to (-1, 1), zero-centered.
    - silu: Sigmoid Linear Unit (Swish) -> x * sigmoid(x). Smooth, non-monotonic.
    - gelu: Gaussian Error Linear Unit -> smooth probabilistic gating, great accuracy.
    - leaky_relu: Allows a small gradient (0.1 * x) when x < 0 to prevent dead neurons.
    - selu: Scaled Exponential Linear Unit -> self-normalizing activation.
    """
    # Normalize the input string to lowercase and strip accidental whitespace
    name = activation_name.lower().strip()

    if name == "relu":
        return layers.Activation("relu")
    elif name == "sigmoid":
        return layers.Activation("sigmoid")
    elif name == "tanh":
        return layers.Activation("tanh")
    elif name in ("silu", "swish"):
        return layers.Activation("silu")
    elif name == "gelu":
        return layers.Activation("gelu")
    elif name in ("leaky_relu", "l_relu"):
        return layers.LeakyReLU(negative_slope=0.1)
    elif name == "selu":
        return layers.Activation("selu")
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")


# ==============================================================================
# 2. VGG6 ARCHITECTURE BUILDER (6 LEARNABLE WEIGHT LAYERS: 4 CONV + 2 DENSE)
# ==============================================================================
def build_vgg6(
    input_shape: tuple = (32, 32, 3),
    num_classes: int = 10,
    activation: str = "relu",
    batch_norm: bool = True,
    dropout_rate: float = 0.2,
    weight_decay: float = 1e-4,
) -> tf.keras.Model:
    """
    Constructs the VGG6 Convolutional Neural Network architecture for CIFAR-10.

    Why is it called VGG6?
    In VGG naming conventions, only layers with learnable weights (Conv2D and Dense)
    are counted. Our network has 4 Conv2D layers + 2 Dense layers = 6 weight layers:
      - Block 1 (32x32 -> 16x16):
          Conv2D(64, 3x3) -> [BatchNorm] -> Activation
          Conv2D(64, 3x3) -> [BatchNorm] -> Activation
          MaxPooling2D(2x2) -> [Dropout]
      - Block 2 (16x16 -> 8x8):
          Conv2D(128, 3x3) -> [BatchNorm] -> Activation
          Conv2D(128, 3x3) -> [BatchNorm] -> Activation
          MaxPooling2D(2x2) -> [Dropout]
      - Classifier Head (8x8x128 = 8192 -> 256 -> 10):
          Flatten()
          Dense(256) -> [BatchNorm] -> Activation -> [Dropout]
          Dense(10, activation='softmax')
    """
    # Create an L2 weight regularizer if weight_decay > 0; otherwise None.
    # L2 regularization adds `weight_decay * sum(w^2)` to the loss to prevent weights from growing too large.
    reg = regularizers.l2(weight_decay) if weight_decay and weight_decay > 0 else None

    # Define the input placeholder expecting 32x32 RGB images: shape = (32, 32, 3)
    inputs = layers.Input(shape=input_shape, name="cifar10_input")
    x = inputs

    # --- Convolutional Block 1: Extracts low-level features (edges, colors, textures) ---
    # Contains 2 Convolutional layers with 64 filters each, kernel size 3x3.
    # Spatial resolution stays 32x32 inside the conv layers because padding="same".
    for i in range(1, 3):
        x = layers.Conv2D(
            filters=64,                      # Learn 64 different 3x3 feature detectors
            kernel_size=(3, 3),              # Each filter scans a 3x3 pixel neighborhood
            padding="same",                  # Zero-pad borders so output height/width stays 32x32
            use_bias=not batch_norm,         # Bias is redundant when BatchNormalization follows immediately
            kernel_initializer="he_normal",  # He initialization keeps variance stable across layers
            kernel_regularizer=reg,          # Apply L2 weight decay penalty to filter weights
            name=f"block1_conv{i}",
        )(x)
        # Optional Batch Normalization: normalizes activations across the mini-batch
        # to stabilize training and allow higher learning rates
        if batch_norm:
            x = layers.BatchNormalization(name=f"block1_bn{i}")(x)
        # Apply the chosen non-linear activation function (e.g., GELU, ReLU, SiLU)
        x = get_activation_layer(activation)(x)

    # Downsample spatial dimensions by 2x (from 32x32x64 down to 16x16x64) by taking max in each 2x2 window
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block1_pool")(x)
    # Optional Dropout: randomly zeroes a fraction of activations during training to prevent co-adaptation
    if dropout_rate > 0:
        x = layers.Dropout(rate=dropout_rate, name="block1_dropout")(x)

    # --- Convolutional Block 2: Extracts mid-level shapes and object parts ---
    # Contains 2 Convolutional layers with 128 filters each, kernel size 3x3.
    # Spatial resolution stays 16x16 inside the conv layers, then halves to 8x8 after MaxPooling.
    for i in range(1, 3):
        x = layers.Conv2D(
            filters=128,                     # Double the filter count to 128 to capture richer patterns
            kernel_size=(3, 3),
            padding="same",
            use_bias=not batch_norm,
            kernel_initializer="he_normal",
            kernel_regularizer=reg,
            name=f"block2_conv{i}",
        )(x)
        if batch_norm:
            x = layers.BatchNormalization(name=f"block2_bn{i}")(x)
        x = get_activation_layer(activation)(x)

    # Downsample spatial dimensions by 2x (from 16x16x128 down to 8x8x128)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block2_pool")(x)
    if dropout_rate > 0:
        x = layers.Dropout(rate=dropout_rate, name="block2_dropout")(x)

    # --- Classifier Head: Combines spatial features to predict the 10 CIFAR-10 classes ---
    # Flatten the 3D feature map (8 x 8 x 128) into a 1D vector of length 8,192
    x = layers.Flatten(name="flatten")(x)

    # Layer 5 (First Fully Connected / Dense Layer): compresses 8,192 features into 256 hidden units
    x = layers.Dense(
        units=256,
        use_bias=not batch_norm,
        kernel_initializer="he_normal",
        kernel_regularizer=reg,
        name="fc1",
    )(x)
    if batch_norm:
        x = layers.BatchNormalization(name="fc1_bn")(x)
    x = get_activation_layer(activation)(x)
    # Apply slightly stronger dropout on the dense layer (up to 0.5) since dense layers have the most parameters
    if dropout_rate > 0:
        x = layers.Dropout(rate=min(0.5, dropout_rate * 1.5), name="fc1_dropout")(x)

    # Layer 6 (Output Dense Layer): maps 256 features to 10 class probabilities using Softmax
    # Softmax ensures all 10 output probabilities are positive and sum to 1.0
    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",
        kernel_regularizer=reg,
        name="predictions",
    )(x)

    # Package the input tensor and output tensor into a Keras Model object
    model = models.Model(inputs=inputs, outputs=outputs, name=f"VGG6_{activation}")
    return model
