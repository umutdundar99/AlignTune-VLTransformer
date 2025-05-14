import os
import lightning as L
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from aligntune.utils.processor import PaliGemmaProcessor
import torch
from torch.nn.utils.rnn import pad_sequence


class AlignTuneDataset(Dataset):
    prompt = "describe the image in details"
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
        self.data["caption"] = self.data["caption_1"]
        self.data = self.data.drop(
            columns=[
                "image",
                "split",
                "source",
                "caption_1",
                "caption_2",
                "caption_3",
                "caption_4",
                "caption_5",
            ]
        )
        #use only %30 of the data
        self.data = self.data.sample(frac=0.3, random_state=42)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data.iloc[idx]["image_path"])
        image = Image.open(image_path).convert("RGB")
        prompt = self.data.iloc[idx]["caption"]
        # Convert the image to a tensor
        model_inputs = self.processor(
            text=[self.prompt],
            images=[image],
            padding="False",
            truncation=False,
        )
        labels = self.processor.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            max_length=512,
            truncation=False,
        )
        model_inputs["labels"] = labels.input_ids[0]
        return model_inputs

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
    result = {}
    keys = list(batch[0].keys())
    keys.remove("labels")
    for key in keys:
        result[key] = torch.stack([item[key] for item in batch])

    labels = [item["labels"] for item in batch]
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
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
            shuffle=False,
            collate_fn=custom_collate_fn,
            # sampler=LengthGroupedSampler(self.train, self.batch_size),
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
            # sampler=LengthGroupedSampler(self.val, self.batch_size),
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
            # sampler=LengthGroupedSampler(self.test, self.batch_size),
        )
