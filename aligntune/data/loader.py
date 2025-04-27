import os
import lightning as L
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from aligntune.utils.processor import PaliGemmaProcessor
import random
import torch


class AlignTuneDataset(Dataset):
    def __init__(self, path: str, processor: PaliGemmaProcessor, task: str = "train"):
        self.data = pd.read_csv(os.path.join(path, "captions.csv"))
        self.data = self.data[self.data["split"] == task]
        # add os.path.join(path, "image") for each image
        self.data["image_path"] = self.data["image"].apply(
            lambda x: os.path.join(path, "resized", x)
        )

        self.task = task
        self.data = self.data.drop(columns=["image", "split", "source"])
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data.iloc[idx]["image_path"])
        image = Image.open(image_path).convert("RGB")

        caption_idx = random.randint(1, 5)
        target_caption = self.data.iloc[idx][f"caption_{caption_idx}"]

        prompt = "Describe the image in detail"
        model_inputs = self.processor(
            text=[
                prompt
            ],  # Single prompt for all images, "Describe the image in detail"
            images=[image],
            padding=False,
            truncation=False,
            max_length=512,
        )

        target_encoding = self.processor.tokenizer(
            target_caption,
            padding=False,
            truncation=False,
            max_length=512,
            return_tensors="pt",
        )
        model_inputs["labels"] = target_encoding.input_ids

        return model_inputs


def custom_collate_fn(batch):
    """Custom collate function to handle different sized tensors."""
    result = {}
    keys = list(batch[0].keys())
    # remove labels from keys
    keys.remove("labels")
    for key in keys:
        result[key] = torch.stack([item[key] for item in batch])

    labels = [batch[i]["labels"] for i in range(len(batch))]
    result["labels"] = labels

    return result


class AlignTuneAnalysisDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule class for loading
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 12,
        processor: PaliGemmaProcessor = None,
    ):
        """
        Initializes the DataModule by setting the data directory and batch size.

        Args:
            data_path (str): Directory containing the preprocessed CSV files.
            batch_size (int): Number of samples per batch.

        """
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.processor = processor

        self.train = AlignTuneDataset(self.data_path, self.processor, task="train")
        self.val = AlignTuneDataset(self.data_path, self.processor, task="val")
        self.test = AlignTuneDataset(self.data_path, self.processor, task="test")

    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset.

        Returns:
            DataLoader: Training DataLoader.
        """
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            # pin_memory=True,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=custom_collate_fn,
        )

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: Validation DataLoader.
        """
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            # pin_memory=True,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

    def test_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: Validation DataLoader.
        """
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            # pin_memory=True,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            collate_fn=custom_collate_fn,
        )
