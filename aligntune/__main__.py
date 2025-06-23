from aligntune.src.train import train
import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    # get if need logger, if it is yes, get project,name,offline, inputs

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
        action="store_true",
        help="Run Weights & Biases in offline mode",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate for training"
    )
    args = parser.parse_args(["--batch_size", "8", 
                              "--num_epochs", "15",
                              "--log_wandb", 
                              "--project_name", "aligntune", 
                              "--run_name", "paligemma-3b-pt-224-cleaned_r8_replace2-15epochs", 
                              "--learning_rate", "1e-5"])
   
    args = vars(args)

    train(args)
