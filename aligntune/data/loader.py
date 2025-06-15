import os
import lightning as L
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import AutoProcessor as PaliGemmaProcessor
import torch
from transformers import AutoProcessor
import random
from nltk.corpus import wordnet
import nltk

processor = AutoProcessor.from_pretrained("paligemma-3b-pt-224")


nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")
possible_prompts = [
    "<image> <bos> describe this image.",
    # "<image> <bos> describe this image in detail.",
    # "<image> <bos> give a detailed description of this image.",
    # "<image> <bos> provide a detailed description of this image.",
    # "<image> <bos> give comprehensive details about this image.",
    # "<image> <bos> provide a comprehensive description of this image.",
    # "<image> <bos> describe this image in a detailed manner.",
    # "<image> <bos> describe this image thoroughly.",
    # "<image> <bos> provide a thorough description of this image.",
    # "<image> <bos> give a thorough description of this image.",
    # "<image> <bos> describe this image with as much detail as possible.",
]


class AlignTuneDataset(Dataset):
    patch_size = 14
    image_size = 224

    def __init__(
        self,
        path: str,
        processor: PaliGemmaProcessor,
        task: str = "train",
        num_replacement: int = 2,
    ):
        self.data = pd.read_csv(os.path.join(path, "captions_cleaned.csv"))
        self.data = self.data[self.data["split"] == task]
        self.data["image_path"] = self.data["image"].apply(
            lambda x: os.path.join(path, "resized", x)
        )
        self.num_replacement = num_replacement
        self.task = task
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data.iloc[idx]["image_path"])
        image = Image.open(image_path).convert("RGB")
        caption = self.data.iloc[idx]["caption"]

        if self.num_replacement > 0:
            # print(f"Original caption: {caption}")
            caption = self.synonym_replacement(
                caption, num_replacement=self.num_replacement
            )
            # print(f"Processed caption: {caption}")
            # print("-----------------------------------------------")

        return {"image": image, "caption": caption}

    def get_synonyms(self, word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        return synonyms

    def synonym_replacement(self, text, num_replacement=1):
        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)

        candidates = [word for word, pos in pos_tags if pos in ["RB", "JJ"]]

        if len(candidates) < num_replacement:
            return " ".join(words)

        words_to_replace = random.sample(candidates, num_replacement)

        for word in words_to_replace:
            synonyms = self.get_synonyms(word)
            if synonyms:
                synonym = random.choice(synonyms)
                text = text.replace(word, synonym, 1)

        return text


def custom_collate_fn(examples):
    texts = [possible_prompts[0] for example in examples]
    labels = [example["caption"] for example in examples]
    images = [example["image"] for example in examples]
    tokens = processor(
        text=texts, images=images, suffix=labels, return_tensors="pt", padding="longest"
    )
    tokens["image"] = images
    tokens = tokens.to(torch.bfloat16)

    return tokens


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
        num_replacement: int = 3,
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

        self.train = AlignTuneDataset(
            self.data_path,
            self.processor,
            task="train",
            num_replacement=num_replacement,
        )
        self.val = AlignTuneDataset(
            self.data_path, self.processor, task="val", num_replacement=0
        )
        self.test = AlignTuneDataset(
            self.data_path, self.processor, task="test", num_replacement=0
        )

    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset.

        Returns:
            DataLoader: Training DataLoader.
        """
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
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
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            collate_fn=custom_collate_fn,
        )
