"""
model.py
Configurable VGG6 model architecture in TensorFlow/Keras for CIFAR-10 classification.
Supports multiple activation functions, optional Batch Normalization, Dropout, and L2 Weight Decay.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def get_activation_layer(activation_name: str) -> layers.Layer:
    """
    Returns the corresponding Keras activation layer for a given name.

    Supported activations:
    - relu: Rectified Linear Unit
    - sigmoid: Logistic Sigmoid
    - tanh: Hyperbolic Tangent
    - silu: Sigmoid Linear Unit (Swish)
    - gelu: Gaussian Error Linear Unit
    - leaky_relu: Leaky ReLU (negative_slope=0.1)
    - selu: Scaled Exponential Linear Unit
    """
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


def build_vgg6(
    input_shape: tuple = (32, 32, 3),
    num_classes: int = 10,
    activation: str = "relu",
    batch_norm: bool = True,
    dropout_rate: float = 0.2,
    weight_decay: float = 1e-4,
) -> tf.keras.Model:
    """
    Builds the VGG6 Convolutional Neural Network architecture for CIFAR-10.
    
    VGG6 Layer Breakdown (6 parameterized weight layers: 4 Conv2D + 2 Dense):
    - Block 1:
        Conv2D(64, 3x3, padding='same') -> [BatchNorm] -> Activation
        Conv2D(64, 3x3, padding='same') -> [BatchNorm] -> Activation
        MaxPooling2D(2x2) -> [Dropout]
    - Block 2:
        Conv2D(128, 3x3, padding='same') -> [BatchNorm] -> Activation
        Conv2D(128, 3x3, padding='same') -> [BatchNorm] -> Activation
        MaxPooling2D(2x2) -> [Dropout]
    - Classifier Head:
        Flatten()
        Dense(256) -> [BatchNorm] -> Activation -> [Dropout]
        Dense(num_classes, activation='softmax')
    """
    reg = regularizers.l2(weight_decay) if weight_decay and weight_decay > 0 else None

    inputs = layers.Input(shape=input_shape, name="cifar10_input")
    x = inputs

    # --- Convolutional Block 1 (2 Conv layers, 64 filters) ---
    for i in range(1, 3):
        x = layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            use_bias=not batch_norm,
            kernel_initializer="he_normal",
            kernel_regularizer=reg,
            name=f"block1_conv{i}",
        )(x)
        if batch_norm:
            x = layers.BatchNormalization(name=f"block1_bn{i}")(x)
        x = get_activation_layer(activation)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block1_pool")(x)
    if dropout_rate > 0:
        x = layers.Dropout(rate=dropout_rate, name="block1_dropout")(x)

    # --- Convolutional Block 2 (2 Conv layers, 128 filters) ---
    for i in range(1, 3):
        x = layers.Conv2D(
            filters=128,
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

    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block2_pool")(x)
    if dropout_rate > 0:
        x = layers.Dropout(rate=dropout_rate, name="block2_dropout")(x)

    # --- Classifier Head (2 Dense layers: 256 -> 10) ---
    x = layers.Flatten(name="flatten")(x)
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
    if dropout_rate > 0:
        x = layers.Dropout(rate=min(0.5, dropout_rate * 1.5), name="fc1_dropout")(x)

    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",
        kernel_regularizer=reg,
        name="predictions",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"VGG6_{activation}")
    return model
