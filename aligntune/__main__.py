from aligntune.src.train import train
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if __name__ == "__main__":
    train()
