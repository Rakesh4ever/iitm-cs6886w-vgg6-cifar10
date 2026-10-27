# CS6886W — Systems Engineering for Deep Learning (Assignment 1)
## Exploring VGG6 on CIFAR-10 in TensorFlow

- **GitHub Repository Link**: [https://github.com/Rakesh4ever/iitm-cs6886w-vgg6-cifar10](https://github.com/Rakesh4ever/iitm-cs6886w-vgg6-cifar10)
- **Submission Report PDF**: ([cs25m530_Rakesh_Kumar.pdf](https://github.com/Rakesh4ever/iitm-cs6886w-vgg6-cifar10/blob/main/cs25m530_Rakesh_Kumar.pdf))

This repository contains a clean, modular, and reproducible implementation of the **VGG6** Convolutional Neural Network trained and evaluated on the **CIFAR-10** dataset using **TensorFlow (Keras)** and **Weights & Biases (W&B)**.

---

## 1. Repository Structure (Question 5a)

To satisfy Question 5(a) (*"Provide clean, modular, and well-commented code with clear separation of training, evaluation. Don't submit a single .ipynb file or .py file"*), the core implementation is separated into modular Python packages under `src/`, accompanied by `driver.ipynb` as an interactive Google Colab launcher for evaluators:

```text
iitm-cs6886w-vgg6-cifar10/
├── src/
│   ├── __init__.py       # Marks src/ as an importable Python package
│   ├── dataset.py        # CIFAR-10 loading, channel-wise Z-score normalization, and data augmentation
│   ├── model.py          # Configurable VGG6 architecture (activations, BatchNorm, Dropout, L2 decay)
│   ├── train.py          # Training loop, optimizer factory, callbacks, plotting, and W&B metric logging
│   ├── evaluate.py       # Standalone evaluation script for Top-1 Test/Val accuracy & per-class accuracy
│   └── sweep.py          # W&B hyperparameter sweep runner for automated parallel-coordinates & curves
├── driver.ipynb          # Interactive Google Colab / VS Code driver notebook (Steps 1 to 6)
├── checkpoints/          # Saved best model weights (.weights.h5) and full trained model (.keras)     
├── requirements.txt      # Pinned Python dependency versions
├── cs25m530_Rakesh_Kumar.pdf  # Final report with detailed answers, plots & GitHub link
└── README.md             # Reproducibility guide & analysis of performance across configurations
```

---

## 2. Environment, Hardware & Seed Details (Question 5b)

- **Execution Platform**: Google Colab (NVIDIA Tesla T4 GPU, 15 GB VRAM) / Linux x86_64 & macOS
- **Python Version**: `Python 3.10+`
- **Required Libraries (`requirements.txt`)**:
  - `tensorflow>=2.15.0`
  - `numpy>=1.24.0`
  - `matplotlib>=3.7.0`
  - `wandb>=0.16.0`

### Determinism & Global Random Seed (`seed = 42`)
To guarantee reproducible train/validation splits, weight initializations, and mini-batch shuffling across runs, `set_global_seed(seed=42)` in `src/dataset.py` sets:
- `os.environ["PYTHONHASHSEED"] = "42"`
- `os.environ["TF_DETERMINISTIC_OPS"] = "1"`
- `random.seed(42)`
- `np.random.seed(42)`
- `tf.random.set_seed(42)`
- `tf.keras.utils.set_random_seed(42)`
- Deterministic train/validation index permutation via `np.random.default_rng(42)` (splitting the 50,000 CIFAR-10 training images into **45,000 Training** and **5,000 Validation** images, with **10,000 Test** images held out).

---

## 3. Exact Instructions to Re-Run the Code & Reproduce Results (Question 5b)

Evaluators can reproduce all results using **either** the interactive **`driver.ipynb`** notebook (in Google Colab or VS Code) **or** standard command-line execution.

### Option A: Re-Running Step-by-Step via `driver.ipynb` (Google Colab / VS Code Colab Extension)

Open `driver.ipynb` in Google Colab (or in VS Code connected to the Google Colab GPU kernel) and execute the 6 cells in order:

#### Step 1: Clone the Modular `.py` Repository & Verify GPU
```python
import os

if os.path.exists("/content"):
    os.chdir("/content")
    !rm -rf /content/iitm-cs6886w-vgg6-cifar10
    !git clone https://github.com/Rakesh4ever/iitm-cs6886w-vgg6-cifar10.git
    os.chdir("/content/iitm-cs6886w-vgg6-cifar10")

print("Current Working Directory:", os.getcwd())
!pip install -q -r requirements.txt
!ls -la src/

import tensorflow as tf
print("TensorFlow Version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
```

#### Step 2: Authenticate Weights & Biases (Optional for Local Runs)
```python
import os
import wandb

WANDB_KEY = os.environ.get("WANDB_API_KEY", "YOUR_WANDB_API_KEY")
if WANDB_KEY != "YOUR_WANDB_API_KEY" and len(WANDB_KEY.strip()) >= 32:
    os.environ["WANDB_API_KEY"] = WANDB_KEY.strip()
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
else:
    print("W&B API Key not set. Step 3 will train locally and still save all plots and checkpoints.")
```

#### Step 3: Train VGG6 with the Best Configuration (`src/train.py`)
```python
import os
if os.path.exists("/content/iitm-cs6886w-vgg6-cifar10"):
    os.chdir("/content/iitm-cs6886w-vgg6-cifar10")

!python -m src.train \
    --activation gelu \
    --optimizer nadam \
    --batch_size 128 \
    --epochs 80 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --dropout_rate 0.2 \
    --batch_norm True \
    --augment True \
    --seed 42 \
    --use_wandb \
    --wandb_project cs6886w-vgg6-cifar10
```
*(Note: For a quick verification run, set `--epochs 20`, which reaches ~83–85% validation accuracy in ~2 minutes on a T4 GPU; full `--epochs 80` reaches ~88.9% validation accuracy).*

#### Step 4: Visualize the Saved Training & Validation Plot (`outputs/baseline_curves.png`)
```python
import os
from IPython.display import Image, display

if os.path.exists("/content/iitm-cs6886w-vgg6-cifar10"):
    os.chdir("/content/iitm-cs6886w-vgg6-cifar10")

display(Image(filename="outputs/baseline_curves.png"))
```

#### Step 5: Standalone Evaluation of the Saved Model on the Test Set (`src/evaluate.py`)
```python
import os
if os.path.exists("/content/iitm-cs6886w-vgg6-cifar10"):
    os.chdir("/content/iitm-cs6886w-vgg6-cifar10")

!python -m src.evaluate \
    --model_path checkpoints/best_vgg6_cifar10.keras \
    --batch_size 128 \
    --seed 42
```

#### Step 6: Launch Automated W&B Hyperparameter Sweep (`src/sweep.py`)
```python
import os
if os.path.exists("/content/iitm-cs6886w-vgg6-cifar10"):
    os.chdir("/content/iitm-cs6886w-vgg6-cifar10")

!python -m src.sweep --project cs6886w-vgg6-cifar10 --count 20
```

---

### Option B: Direct Command-Line / Terminal Execution

```bash
# 1. Clone repository and install dependencies
git clone https://github.com/Rakesh4ever/iitm-cs6886w-vgg6-cifar10.git
cd iitm-cs6886w-vgg6-cifar10
pip install -r requirements.txt

# 2. Train VGG6 with the best configuration
python -m src.train \
  --activation gelu \
  --optimizer nadam \
  --batch_size 128 \
  --epochs 80 \
  --learning_rate 0.001 \
  --weight_decay 0.0001 \
  --dropout_rate 0.2 \
  --batch_norm True \
  --augment True \
  --seed 42

# 3. Evaluate the saved checkpoint on Validation and Test sets (without retraining)
python -m src.evaluate \
  --model_path checkpoints/best_vgg6_cifar10.keras \
  --batch_size 128 \
  --seed 42

# 4. Run W&B Bayesian hyperparameter sweep across configurations
wandb login
python -m src.sweep --project cs6886w-vgg6-cifar10 --count 20
```

---

## 4. Data Normalization & Augmentation Pipeline (Question 1a)

Implemented in `src/dataset.py`:
1. **Pixel Scaling**: Raw `uint8` RGB images in `[0, 255]` are cast to `tf.float32` and divided by `255.0` to lie in `[0.0, 1.0]`.
2. **Channel-Wise Z-Score Standardization**: Each RGB channel is standardized using the CIFAR-10 training set mean and standard deviation:
   - `CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)`
   - `CIFAR10_STD  = (0.2470, 0.2435, 0.2616)`
   - Formula: `image = (image - CIFAR10_MEAN) / CIFAR10_STD`
3. **Training Data Augmentation** (applied dynamically via `tf.data` only to the 45,000 training images):
   - **Reflection Padding**: 4 pixels on top, bottom, left, and right borders (`32x32x3 -> 40x40x3`).
   - **Random Crop**: Randomly cropped back to `32x32x3` (`tf.image.random_crop`).
   - **Random Horizontal Flip**: Applied with probability `p = 0.5` (`tf.image.random_flip_left_right`).
   - Validation (5,000 images) and Test (10,000 images) pipelines only apply normalization (no augmentation, no shuffling).

---

## 5. Single Best Configuration & Reported Results (Question 1 & Question 4)

| Hyperparameter / Metric | Best Configuration Value |
| :--- | :--- |
| **Architecture** | VGG6 (4 `Conv2D` + 2 `Dense` layers, `BatchNorm`, `Dropout`) |
| **Activation Function** | `GELU` |
| **Optimizer** | `Nadam` |
| **Initial Learning Rate** | `0.001` (with `ReduceLROnPlateau` factor=`0.5`, patience=`5`, min\_lr=`1e-6`) |
| **Batch Size** | `128` |
| **Epochs** | `80` (reaches `~83.5%` at 10 epochs, `~85.4%` at 20 epochs, `88.94%` at 80 epochs) |
| **Batch Normalization** | `True` |
| **Weight Decay (L2)** | `0.0001` (`1e-4`) |
| **Dropout Rate** | `0.2` (`0.3` on dense classifier head) |
| **Random Seed** | `42` |
| **Best Validation Accuracy** | **88.94%** |
| **Final Test Top-1 Accuracy** | **88.45%** |

---

## 6. How Performance Varies Across Different Configurations (Question 2 Detailed Analysis)

### (a) Varying the Activation Function (`ReLU`, `Sigmoid`, `Tanh`, `SiLU`, `GELU`)

| Activation Function | Typical Val Accuracy (with BN) | Convergence Behavior & Mathematical Explanation |
| :--- | :---: | :--- |
| **GELU** | **88.94%** | Smooth, non-monotonic gating (`x * Phi(x)`) that permits small negative gradients for slightly negative inputs, preventing dead neurons and smoothing the loss landscape. |
| **SiLU (Swish)** | **88.40%** | Smooth self-gated activation (`x * sigmoid(x)`) with a bounded negative trough; closely matches GELU in convergence speed and generalization. |
| **ReLU** | **87.62%** | Non-saturating positive gradient (`f'(x) = 1` for `x > 0`) enables fast training, but hard zeroing for `x < 0` causes a subset of filters to suffer from the "dying ReLU" issue. |
| **Leaky ReLU** | **87.90%** | Retains a small slope (`0.1`) for negative inputs, mitigating dying ReLU while maintaining piecewise-linear speed. |
| **Tanh** | **81.20%** | Zero-centered output in `(-1, 1)` aids layer conditioning compared to Sigmoid, but saturates at both positive and negative tails (`f'(x) = 1 - tanh^2(x) -> 0`), slowing deeper gradient flow. |
| **Sigmoid** | **68.40%** *(or `<25%` without BN)* | Outputs lie in `(0, 1)` (non-zero-centered, inducing zig-zag weight updates) and its maximum derivative is only `0.25`. Multiplying factors `<= 0.25` across 6 layers causes severe vanishing gradients. |

### (b) Varying the Optimizer (`SGD`, `Nesterov-SGD`, `Adam`, `Adagrad`, `RMSprop`, `Nadam`)

| Optimizer | Best Learning Rate | Typical Val Accuracy | Convergence Speed & Optimization Dynamics |
| :--- | :---: | :---: | :--- |
| **Nadam** | `1e-3` | **88.94%** | Combines Adam's per-parameter second-moment scaling with **Nesterov look-ahead momentum**, yielding the fastest early convergence and highest final accuracy. |
| **Adam** | `1e-3` | **88.35%** | Bias-corrected exponential moving averages of first (`m_t`) and second (`v_t`) gradient moments provide rapid, stable convergence across hyperparameter settings. |
| **Nesterov-SGD** | `1e-2` (`momentum=0.9`) | **88.10%** | Evaluates gradients at the projected look-ahead position (`theta + mu * v_t`), damping oscillations in high-curvature ravines and generalizing strongly once LR decays. |
| **RMSprop** | `1e-3` | **86.75%** | Normalizes updates by an exponentially decaying average of squared gradients, preventing step-size vanishing while controlling oscillations. |
| **Standard SGD** | `1e-2` (`momentum=0.0`) | **81.90%** | Updates strictly along the noisy instantaneous mini-batch gradient; progresses slowly along flat directions and oscillates across steep canyon walls. |
| **Adagrad** | `1e-2` | **74.80%** | Accumulates squared gradients monotonically in the denominator (`G_t = G_{t-1} + g_t^2`), shrinking the effective learning rate toward zero prematurely and stalling early. |

### (c) Varying Batch Size, Learning Rate, and Number of Epochs
- **Batch Size (`32, 64, 128, 256, 512`)**:
  - **Small Batches (`32, 64`)**: Perform more gradient updates per epoch (`1,406` and `703` steps/epoch) and inject beneficial stochastic gradient noise that acts as an implicit regularizer (~88.3% val acc), but require longer wall-clock time per epoch due to lower GPU utilization.
  - **Moderate Batch Size (`128`)**: Optimal balance between GPU throughput (`352` steps/epoch) and generalization (**88.94%** val acc).
  - **Large Batches (`256, 512`)**: Perform fewer parameter updates per epoch (`88` steps/epoch at `bs=512`), converging slower per epoch (~85.6% val acc) and tending toward sharper minima with a larger generalization gap unless the learning rate is scaled linearly.
- **Learning Rate (`1e-4` to `1e-2`)**:
  - **Too High (`1e-2` with Adam/Nadam)**: Large adaptive steps overshoot narrow valleys in the loss landscape, causing high validation loss variance or divergence.
  - **Optimal (`1e-3` for Adam/Nadam, `1e-2` for Nesterov-SGD)**: Fast initial descent during the first 20 epochs, followed by fine-grained convergence when `ReduceLROnPlateau` halves the step size at plateaus.
  - **Too Low (`1e-4`)**: Gradient updates are too small, causing slow progress and underfitting (~79.5% at 50 epochs).
- **Number of Epochs (`20, 30, 50, 80, 100`)**:
  - Because random crop and flip augmentation increase training diversity, `20–30` epochs (~84–86% val acc) is still improving.
  - Training for `60–80` epochs allows validation accuracy to plateau near **89%**, while `Dropout(0.2)` and `L2 Weight Decay (1e-4)` prevent severe overfitting at `80–100` epochs.
