"""
sweep.py
Purpose:
    Automates Weights & Biases (W&B) Bayesian hyperparameter sweeps to explore how
    different activations, optimizers, batch sizes, epochs, learning rates, batch norm,
    and weight decay affect VGG6 validation accuracy on CIFAR-10.
    Automatically generates the W&B Parallel Coordinates plot, Scatter plot, and Training curves.
"""

# ==============================================================================
# LIBRARY IMPORTS (AND WHY WE NEED EACH ONE)
# ==============================================================================
import argparse  # Built-in Python module to parse `--project` and `--count` command-line arguments.
import wandb     # Weights & Biases SDK used to register the sweep configuration on the W&B server
                 # and run the automated sweep agent trials.

# Import our core training function from `src/train.py` so each sweep trial runs the exact same pipeline
from src.train import train_and_evaluate


# ==============================================================================
# 1. HYPERPARAMETER SWEEP CONFIGURATION DICTIONARY
# ==============================================================================
# Tells Weights & Biases to use Bayesian Optimization ("bayes") to intelligently search
# combinations of hyperparameters with the goal of maximizing validation accuracy ("val_acc").
SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "activation": {
            "values": ["relu", "sigmoid", "tanh", "silu", "gelu", "leaky_relu", "selu"]
        },
        "batch_norm": {
            "values": [True, False]
        },
        "batch_size": {
            "values": [32, 64, 128, 256, 512]
        },
        "epochs": {
            "values": [20, 30, 50, 80, 100]
        },
        "learning_rate": {
            "values": [1e-4, 2e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        },
        "momentum": {
            "values": [0.0, 0.5, 0.8, 0.9]
        },
        "optimizer": {
            "values": ["sgd", "nesterov", "adam", "adagrad", "rmsprop", "nadam"]
        },
        "weight_decay": {
            "values": [0.0, 5e-5, 1e-4, 2e-4, 5e-4]
        },
        "dropout_rate": {
            "value": 0.2
        },
        "seed": {
            "value": 42
        },
    },
}


# ==============================================================================
# 2. SINGLE SWEEP TRIAL EXECUTION FUNCTION
# ==============================================================================
def sweep_train_fn():
    """
    Callback executed by `wandb.agent` for each trial in the hyperparameter sweep.
    Receives a set of hyperparameters sampled by W&B, formats a descriptive run name
    (e.g. `act_gelu_bs_128_opt_nadam_lr_0.001_wd_0.0001_bn_True`), and trains VGG6.
    """
    with wandb.init() as run:
        # Convert the sampled W&B hyperparameters into a standard Python dictionary
        cfg = dict(wandb.config)

        # Assign a clear run name matching the assignment's W&B plot legend convention
        run.name = (
            f"act_{cfg['activation']}_bs_{cfg['batch_size']}_opt_{cfg['optimizer']}_"
            f"lr_{cfg['learning_rate']}_wd_{cfg['weight_decay']}_bn_{cfg['batch_norm']}"
        )

        # Enable W&B epoch logging, skip saving heavy full `.keras` files for every trial,
        # and store temporary trial weights/plots using the unique `run.id`
        cfg["use_wandb"] = True
        cfg["save_full_model"] = False
        cfg["checkpoint_path"] = f"checkpoints/sweep_{run.id}.weights.h5"
        cfg["plot_path"] = f"outputs/sweep_{run.id}_curves.png"

        # Launch training and evaluation for this sampled configuration
        train_and_evaluate(cfg)


# ==============================================================================
# 3. COMMAND-LINE ENTRY POINT TO LAUNCH THE SWEEP
# ==============================================================================
def main():
    """
    Initializes the W&B sweep on the server and runs `--count` sequential trials in Colab.
    """
    parser = argparse.ArgumentParser(description="Run W&B Sweep for VGG6 on CIFAR-10")
    parser.add_argument("--project", type=str, default="iitm-cs6886w-vgg6-cifar10")
    parser.add_argument("--count", type=int, default=20, help="Number of sweep runs to execute")
    args = parser.parse_args()

    # Register the sweep configuration with Weights & Biases and obtain a `sweep_id`
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=args.project)
    print(f"Initialized W&B Sweep ID: {sweep_id}")

    # Start the W&B agent to execute `args.count` training runs automatically
    wandb.agent(sweep_id, function=sweep_train_fn, count=args.count)


if __name__ == "__main__":
    main()
