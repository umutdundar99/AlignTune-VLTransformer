import torch
import lightning as L
from torchmetrics.text import ROUGEScore, BLEUScore
from pycocoevalcap.cider.cider import Cider
from transformers import PaliGemmaForConditionalGeneration
from nltk.translate.bleu_score import SmoothingFunction
from aligntune.utils.processor import PaliGemmaProcessor
import pandas as pd


class CIDErWrapper:
    def __init__(self):
        self.cider = Cider()

    def __call__(self, predictions, references):
        gts = {i: [ref] for i, ref in enumerate(references)}
        res = {i: [pred] for i, pred in enumerate(predictions)}
        score, _ = self.cider.compute_score(gts, res)
        return score


class PaliGemmaModule(L.LightningModule):
    def __init__(
        self,
        model: PaliGemmaForConditionalGeneration,
        processor: PaliGemmaProcessor,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        max_tokens_to_generate: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_tokens_to_generate = max_tokens_to_generate
        self.temperature = temperature
        self.top_p = top_p
        self.rouge = ROUGEScore()
        self.CIDEr= CIDErWrapper()
        
        self.smooth = SmoothingFunction().method1
        self.save_hyperparameters(ignore=["model", "processor"])
        self.df = pd.DataFrame(columns=["generated", "actual"])

    def forward(self, batch):
        generated_tokens = []
        outputs = self.model(**batch)
        prompt = [
            "<image> <bos> describe this image." for _ in range(len(batch["image"]))
        ]
        inputs = self.processor(batch["image"], prompt, return_tensors="pt").to("cuda")
        generated_tokens = self.model.generate(**inputs, max_new_tokens=20)
        return outputs.loss, generated_tokens

    def training_step(self, batch, batch_idx):
        loss, generated_tokens = self(batch)

        generated_captions = [
            self.processor.tokenizer.decode(g, skip_special_tokens=True)[23:]
            for g in generated_tokens
        ]
        actual_captions = [
            self.processor.tokenizer.decode(seq[seq != -100], skip_special_tokens=True)
            for seq in batch["labels"]
        ]

        cider_score = self.CIDEr(generated_captions, actual_captions)
        rouge_scores = self.rouge(generated_captions, actual_captions)

        self.log_dict(
            {
                "train/loss": loss,
                "train/cider": cider_score,
                "train/rouge1": rouge_scores["rouge1_fmeasure"],
                "train/rouge2": rouge_scores["rouge2_fmeasure"],
                "train/rougeL": rouge_scores["rougeL_fmeasure"],
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, generated_tokens = self(batch)
        generated_captions = [
            self.processor.tokenizer.decode(g, skip_special_tokens=True)[23:]
            for g in generated_tokens
        ]
        actual_captions = [
            self.processor.tokenizer.decode(seq[seq != -100], skip_special_tokens=True)
            for seq in batch["labels"]
        ]

        self.df = pd.concat(
            [
                self.df,
                pd.DataFrame(
                    {"generated": generated_captions, "actual": actual_captions}
                ),
            ],
            ignore_index=True,
        )

        cider_score = self.CIDEr(generated_captions, actual_captions)
        rouge_scores = self.rouge(generated_captions, actual_captions)

        self.log_dict(
            {
                "val/loss": loss,
                "val/cider": cider_score,
                "val/rouge1": rouge_scores["rouge1_fmeasure"],
                "val/rouge2": rouge_scores["rouge2_fmeasure"],
                "val/rougeL": rouge_scores["rougeL_fmeasure"],
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def on_validation_epoch_end(self):
        self.df = pd.concat(
            [
                self.df,
                pd.DataFrame(
                    {
                        "generated": [f"-----{self.current_epoch}-----"],
                        "actual": [f"-----{self.current_epoch}-----"],
                    }
                ),
            ],
            ignore_index=True,
        )
        self.df.to_csv("generated_captions.csv", index=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
