from aligntune.src.train import train
from aligntune.src.test import test
import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with AlignTune")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of epochs for training"
    )
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="Use Weights & Biases for logging",
    )

    parser.add_argument(
        "--project_name",
        type=str,
        default="aligntune",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="paligemma-3b-pt-224-cleaned_all_data_r8_replace4",
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--offline",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate for training"
    )
    parser.add_argument(
        "--test",
        type=bool,
        default=False,
        help="Run in test mode (load a checkpoint and evaluate)",
    )
    parser.add_argument(
        "--test-checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint for testing",
    )
    args = parser.parse_args(
        [
            "--log_wandb",
            "--project_name",
            "aligntune",
            "--run_name",
            "zvk8ks4u-test-lastckpt-withbleu",
            "--learning_rate",
            "1e-5",
            "--offline",
            False,
            "--test",
            "True",
            "--test-checkpoint",
            "aligntune/logs/aligntune/zvk8ks4u/checkpoints/last.ckpt",
        ]
    )

    args = vars(args)

    if not args["test"]:
        train(args)
    else:
        test(args)
