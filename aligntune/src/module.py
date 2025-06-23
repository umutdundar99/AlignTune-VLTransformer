import torch
import lightning as L
from torchmetrics.text import ROUGEScore, BLEUScore
from pycocoevalcap.cider.cider import Cider
from transformers import PaliGemmaForConditionalGeneration
from aligntune.utils.processor import PaliGemmaProcessor
import pandas as pd
import torchmetrics as tm
from torchmetrics import Metric
from nltk.translate.meteor_score import meteor_score


class CIDErMetric(Metric):
    full_state_update = True  # accumulate all

    def __init__(self):
        super().__init__(dist_sync_on_step=False)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.cider = Cider()

    def update(self, preds, targets):
        self.preds.extend(preds)
        self.targets.extend(targets)

    def compute(self):
        res = {i: [p] for i, p in enumerate(self.preds)}
        gts = {i: [t] for i, t in enumerate(self.targets)}
        score, _ = self.cider.compute_score(gts, res)
        return score
class METEORMetric(Metric):
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.preds = []
        self.targets = []

    def update(self, preds, targets):
        self.preds.extend(preds)
        self.targets.extend(targets)

    def compute(self):
        scores = [meteor_score(refs, pred) for pred, refs in zip(self.preds, self.targets)]
        return sum(scores) / len(scores)

    def reset(self):
        self.preds.clear()
        self.targets.clear()

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
  
        self.training_metrics_rouge_cider = tm.MetricCollection({
            "rouge": ROUGEScore(),
           
            "CIDEr": CIDErMetric(),    
        })
        self.validation_metrics_rouge_cider  = tm.MetricCollection({
            "rouge": ROUGEScore(),
            
            "CIDEr": CIDErMetric(),
        })
        self.training_metric_bleu = tm.MetricCollection({
            "bleu": BLEUScore(),
        })
        self.validation_metric_bleu = tm.MetricCollection({
            "bleu": BLEUScore(),
        })

        
        self.save_hyperparameters(ignore=["model", "processor"])
        self.df = pd.DataFrame(columns=["generated", "actual"])

    def forward(self, batch):
        generated_tokens = []
        outputs = self.model(**batch)
        prompt = [
            "<image> <bos> describe this image." for _ in range(len(batch["image"]))
        ]
        inputs = self.processor(batch["image"], prompt, return_tensors="pt").to("cuda")
        generated_tokens = self.model.generate(**inputs, max_new_tokens=self.max_tokens_to_generate)
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
        actual_captions_list = [[ref] for ref in actual_captions]

        metrics_rouge_cider = self.training_metrics_rouge_cider(
            generated_captions, actual_captions
        )
        metrics_bleu = self.training_metric_bleu(generated_captions, actual_captions_list)


        self.log_dict(
            {
                "train/loss": loss,
                "train/rouge1": metrics_rouge_cider["rouge1_fmeasure"],
                "train/rouge2": metrics_rouge_cider["rouge2_fmeasure"],
                "train/rougeL": metrics_rouge_cider["rougeL_fmeasure"],
                "train/bleu": metrics_bleu["bleu"],
                "train/cider": metrics_rouge_cider["CIDEr"],
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

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

        actual_captions_list = [[ref] for ref in actual_captions]
        

        self.df = pd.concat(
            [
                self.df,
                pd.DataFrame(
                    {"generated": generated_captions, "actual": actual_captions}
                ),
            ],
            ignore_index=True,
        )

        self.validation_metrics_rouge_cider.update(
            generated_captions, actual_captions
        )
        self.validation_metric_bleu.update(
            generated_captions, actual_captions_list
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

        # log validation metrics
        metrics_rouge_cider = self.validation_metrics_rouge_cider.compute()
        metrics_bleu = self.validation_metric_bleu.compute()
        self.log_dict(
            {
                "val/rouge1": metrics_rouge_cider["rouge1_fmeasure"],
                "val/rouge2": metrics_rouge_cider["rouge2_fmeasure"],
                "val/rougeL": metrics_rouge_cider["rougeL_fmeasure"],
                "val/bleu": metrics_bleu["bleu"],
                "val/cider": metrics_rouge_cider["CIDEr"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        
        self.validation_metrics_rouge_cider.reset()
        self.validation_metric_bleu.reset()

    def on_train_epoch_end(self):
        # reset train metrics
        self.training_metrics_rouge_cider.reset()
        self.training_metric_bleu.reset()

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
