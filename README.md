# CS6886W - Systems Engineering for Deep Learning: Assignment 1
## Exploring VGG6 on CIFAR-10 in TensorFlow

This repository contains a clean, modular, and reproducible implementation of the **VGG6** architecture trained and evaluated on the **CIFAR-10** dataset using **TensorFlow (Keras)** and **Weights & Biases (W&B)**.

---

## 1. Repository Structure

To satisfy Question 5(a) (*"Provide clean, modular, and well-commented code with clear separation of training, evaluation. Don't submit a single .ipynb file or .py file"*), the project is organized into modular Python components:

```text
cs6886w_assignment1_vgg6/
├── src/
│   ├── __init__.py
│   ├── dataset.py        # CIFAR-10 loading, channel-wise normalization, and data augmentation
│   ├── model.py          # Configurable VGG6 architecture (activations, BatchNorm, Dropout, L2 decay)
│   ├── train.py          # Training loop, optimizer factory, callbacks, and W&B metric logging
│   ├── evaluate.py       # Standalone evaluation script for Top-1 Test/Val accuracy & per-class accuracy
│   └── sweep.py          # W&B hyperparameter sweep runner for automated parallel-coordinates & curves
├── checkpoints/          # Saved best model weights (.weights.h5) and full model (.keras)
├── outputs/              # Training/validation loss & accuracy plots and JSON metrics
├── requirements.txt      # Pinned dependency versions
└── README.md             # Reproducibility instructions & performance analysis across configurations
```

---

## 2. Environment & Dependency Details (Question 5b)

- **Platform**: Google Colab (Tesla T4 GPU, 15 GB VRAM) / Linux x86_64
- **Python Version**: `Python 3.10+`
- **Dependencies (`requirements.txt`)**:
  - `tensorflow>=2.15.0`
  - `numpy>=1.24.0`
  - `matplotlib>=3.7.0`
  - `wandb>=0.16.0`

### Installation
```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>
pip install -r requirements.txt
```

---

## 3. Seed & Determinism Configuration (Question 5b)

To ensure exact reproducibility across runs, `set_global_seed(seed=42)` in `src/dataset.py` configures:
- `PYTHONHASHSEED = "42"`
- `TF_DETERMINISTIC_OPS = "1"`
- `random.seed(42)`
- `np.random.seed(42)`
- `tf.random.set_seed(42)`
- `tf.keras.utils.set_random_seed(42)`
- Deterministic train/validation index permutation (`np.random.default_rng(42)`).

---

## 4. Exact Commands to Reproduce Results (Question 5b)

### A. Reproduce Best Configuration (Question 1 & Question 4)
Run the best-performing VGG6 configuration (`activation=gelu`, `optimizer=nadam`, `batch_size=128`, `learning_rate=0.001`, `weight_decay=0.0001`, `batch_norm=True`, `epochs=80`, `seed=42`):

```bash
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
```

To log this run to Weights & Biases, append `--use_wandb --wandb_project iitm-cs6886w-vgg6-cifar10`.

### B. Evaluate the Saved Model on the Test Set (Question 5a & 5c)
Once trained, the best model is saved to `checkpoints/best_vgg6_cifar10.keras`. You can verify its Top-1 validation and test accuracy without retraining:

```bash
python -m src.evaluate \
  --model_path checkpoints/best_vgg6_cifar10.keras \
  --batch_size 128 \
  --seed 42
```

### C. Run W&B Hyperparameter Sweeps (Question 2 & Question 3)
To run automated hyperparameter sweeps across activations, optimizers, batch sizes, learning rates, and epochs and generate the W&B Parallel Coordinates plot, Scatter plot, and Training/Validation curves:

```bash
wandb login
python -m src.sweep --project iitm-cs6886w-vgg6-cifar10 --count 25
```

---

## 5. Data Normalization & Augmentation (Question 1a)

1. **Normalization**:
   - Pixel intensities are scaled from `[0, 255]` to `[0.0, 1.0]`.
   - Channel-wise Z-score standardization is applied using CIFAR-10 training set statistics:
     - `Mean = (0.4914, 0.4822, 0.4465)`
     - `Std  = (0.2470, 0.2435, 0.2616)`
2. **Data Augmentation (Training Set Only)**:
   - Reflection padding by 4 pixels on each border (`32x32 -> 40x40`).
   - Random crop back to `32x32x3`.
   - Random horizontal flip (`p = 0.5`).

---

## 6. How Performance Varies Across Configurations (Question 2 Analysis)

### (a) Effect of Activation Functions (`ReLU`, `Sigmoid`, `Tanh`, `SiLU`, `GELU`)
- **GELU & SiLU (Swish)**: Achieve the highest validation accuracy (~87%–89% with BatchNorm). Because both are smooth, non-monotonic functions that allow small negative gradients to flow rather than hard-zeroing negative pre-activations, they avoid the "dying neuron" problem and provide smoother optimization landscapes.
- **ReLU**: Performs strongly (~86%–88%), offering fast computation and non-saturating positive gradients (`f'(x) = 1` for `x > 0`), though some units can become inactive ("dying ReLU") when large gradients push weights negative.
- **Tanh**: Zero-centered output in `(-1, 1)`, which helps next-layer conditioning compared to Sigmoid, but saturates at both tails (`|x| >> 0`), causing vanishing gradients in deeper layers and slower convergence (~78%–82%).
- **Sigmoid**: Performs the worst (~60%–72%, or fails to converge without BatchNorm). Its outputs are in `(0, 1)` (not zero-centered, causing zig-zag gradient updates) and its maximum derivative is only `0.25`, leading to severe vanishing gradients across the 6 layers.

### (b) Effect of Optimizers (`SGD`, `Nesterov-SGD`, `Adam`, `Adagrad`, `RMSprop`, `Nadam`)
- **Nadam (Nesterov + Adam)**: Achieves the fastest initial convergence and highest validation accuracy by combining adaptive per-parameter learning rates (second-moment scaling) with Nesterov look-ahead momentum.
- **Adam**: Very fast convergence and robust across hyperparameter settings due to bias-corrected first and second moment estimates of gradients.
- **Nesterov-SGD vs. Standard SGD**: Standard SGD without momentum converges slowly and oscillates along steep ravines. Adding Nesterov momentum (`momentum=0.9`) evaluates gradients at the "look-ahead" position, damping oscillations and reaching high generalization accuracy when paired with learning rate decay.
- **RMSprop**: Maintains a moving average of squared gradients to normalize step sizes, overcoming Adagrad's decaying learning rate problem and converging well at small learning rates (`lr <= 0.001`).
- **Adagrad**: Accumulates all historical squared gradients monotonically in the denominator, causing the effective learning rate to shrink prematurely toward zero and stalling training early on CIFAR-10.

### (c) Effect of Batch Size, Number of Epochs, and Learning Rate
- **Batch Size (`32, 64, 128, 256, 512`)**:
  - Small batch sizes (`32, 64`) inject beneficial stochastic gradient noise that acts as an implicit regularizer and helps escape sharp local minima, but result in slower wall-clock time per epoch due to lower GPU utilization.
  - Moderate batch sizes (`128`) provide the best trade-off between throughput and generalization.
  - Very large batch sizes (`512`) perform fewer parameter updates per epoch (`88 steps/epoch` vs `703 steps/epoch` at `bs=64`), requiring more epochs or linear learning-rate scaling to match accuracy, and tend to converge to sharper minima with a larger generalization gap.
- **Learning Rate (`1e-4` to `1e-2`)**:
  - Too high (`>= 1e-2` with Adam/Nadam): Causes loss spikes, divergence, or oscillation around the minimum.
  - Optimal (`1e-3` for Adam/Nadam; `1e-2` for SGD+Momentum): Rapid loss reduction followed by fine-grained convergence via `ReduceLROnPlateau`.
  - Too low (`<= 1e-4`): Extremely slow convergence; underfits within a 30–50 epoch budget.
- **Number of Epochs (`20` to `100`)**:
  - Under 30 epochs, the model is still improving on augmented CIFAR-10 data.
  - Around 60–80 epochs with learning-rate decay and data augmentation, validation accuracy plateaus at its peak while dropout (`0.2`) and L2 weight decay (`1e-4`) prevent severe overfitting.
