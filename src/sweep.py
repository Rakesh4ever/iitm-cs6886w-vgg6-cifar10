"""
sweep.py
Weights & Biases (W&B) hyperparameter sweep runner for exploring VGG6 configurations
on CIFAR-10 (activations, optimizers, batch sizes, epochs, learning rates, batch norm, weight decay).
Automatically generates the parallel-coordinates plot, val_acc vs. created scatter plot,
and train/val loss & accuracy curves in the W&B dashboard.
"""

import argparse
import wandb
from src.train import train_and_evaluate


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


def sweep_train_fn():
    """
    Single W&B sweep trial entry point.
    Formats the run name to match the assignment specification:
    e.g. act_gelu_bs_128_opt_nadam_lr_0.001_wd_0.0001_bn_True
    """
    with wandb.init() as run:
        cfg = dict(wandb.config)
        run.name = (
            f"act_{cfg['activation']}_bs_{cfg['batch_size']}_opt_{cfg['optimizer']}_"
            f"lr_{cfg['learning_rate']}_wd_{cfg['weight_decay']}_bn_{cfg['batch_norm']}"
        )
        cfg["use_wandb"] = True
        cfg["save_full_model"] = False
        cfg["checkpoint_path"] = f"checkpoints/sweep_{run.id}.weights.h5"
        cfg["plot_path"] = f"outputs/sweep_{run.id}_curves.png"

        train_and_evaluate(cfg)


def main():
    parser = argparse.ArgumentParser(description="Run W&B Sweep for VGG6 on CIFAR-10")
    parser.add_argument("--project", type=str, default="iitm-cs6886w-vgg6-cifar10")
    parser.add_argument("--count", type=int, default=20, help="Number of sweep runs to execute")
    args = parser.parse_args()

    sweep_id = wandb.sweep(SWEEP_CONFIG, project=args.project)
    print(f"Initialized W&B Sweep ID: {sweep_id}")
    wandb.agent(sweep_id, function=sweep_train_fn, count=args.count)


if __name__ == "__main__":
    main()
