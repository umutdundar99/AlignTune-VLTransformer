import os
import lightning as L
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from aligntune.utils.processor import PaliGemmaProcessor
import torch
from aligntune.data.sampler import LengthGroupedSampler


class AlignTuneDataset(Dataset):
    prompt = "describe the image in detail"
    patch_size = 14
    image_size = 224

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

    def step(self, idx):
        image_path = os.path.join(self.data.iloc[idx]["image_path"])
        image = Image.open(image_path).convert("RGB")

        caption_idx = 1
        target_caption = self.data.iloc[idx][f"caption_{caption_idx}"]

        prompt_tokens = self.processor.tokenizer(
            self.prompt,
            add_special_tokens=False,
        ).input_ids

        caption_tokens = self.processor.tokenizer(
            target_caption,
            add_special_tokens=False,
            truncation=False,
            max_length=512,
        ).input_ids

        sep_token_id = self.processor.tokenizer.eos_token_id

        input_ids = prompt_tokens + [sep_token_id] + caption_tokens

        attention_mask = [1] * len(input_ids)

        labels = [-100] * (len(prompt_tokens) + 1) + caption_tokens

        model_inputs = self.processor(
            text=None,
            images=[image],
            padding=False,
            truncation=False,
            max_length=512,
        )

        model_inputs["input_ids"] = torch.tensor(input_ids, dtype=torch.long)
        model_inputs["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
        model_inputs["labels"] = torch.tensor(labels, dtype=torch.long)

        return model_inputs

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.step(i) for i in idx]
        else:
            return self.step(idx)

    def get_input_length(self, idx):
        """
        Get the total input length (prompt + image token + caption token) for the given index.
        """

        prompt_tokens = self.processor.tokenizer(
            self.prompt,
            add_special_tokens=False,
        ).input_ids
        num_prompt_tokens = len(prompt_tokens)

        num_image_tokens = (self.image_size // self.patch_size) ** 2
        caption_idx = 1
        target_caption = self.data.iloc[idx][f"caption_{caption_idx}"]
        target_tokens = self.processor.tokenizer(
            target_caption,
            add_special_tokens=False,
            max_length=512,
            truncation=False,
        ).input_ids
        num_caption_tokens = len(target_tokens)

        # num_prompt + num_image = 261, it is fixed
        # do not change the input prompt
        total_length = num_prompt_tokens + num_image_tokens + num_caption_tokens

        return total_length


def custom_collate_fn(batch):
    """Custom collate function to handle different sized tensors."""
    result = {}
    keys = list(batch[0].keys())
    # remove labels from keys
    keys.remove("labels")
    keys.remove("input_ids")
    for key in keys:
        result[key] = torch.stack([item[key] for item in batch])

    labels = [batch[i]["labels"] for i in range(len(batch))]
    input_ids = [batch[i]["input_ids"] for i in range(len(batch))]
    result["labels"] = labels
    result["input_ids"] = input_ids

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
            shuffle=False,
            # collate_fn=custom_collate_fn,
            sampler=LengthGroupedSampler(self.train, self.batch_size),
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
            # collate_fn=custom_collate_fn,
            sampler=LengthGroupedSampler(self.val, self.batch_size),
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
            # collate_fn=custom_collate_fn,
            sampler=LengthGroupedSampler(self.test, self.batch_size),
        )
