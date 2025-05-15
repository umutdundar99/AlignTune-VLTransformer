import torch
import lightning as L
import numpy as np
from torchmetrics.text import ROUGEScore, BLEUScore
from pycocoevalcap.cider.cider import Cider

from nltk.translate.bleu_score import SmoothingFunction
import nltk
from aligntune.src.nn.gemma import PaliGemmaForConditionalGeneration
from aligntune.utils.processor import PaliGemmaProcessor
import torch.nn.functional as F
from aligntune.src.nn.gemma import KVCache
from aligntune.src.nn.gemma import GemmaRMSNorm


class CIDErWrapper:
    def __init__(self):
        self.cider = Cider()

    def __call__(self, predictions, references):
        gts = {i: [ref] for i, ref in enumerate(references)}
        res = {i: [pred] for i, pred in enumerate(predictions)}
        score, _ = self.cider.compute_score(gts, res)
        return score

cider_metric = CIDErWrapper()

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
        self.model =model#.to(dtype=torch.float16)
        self.processor = processor
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_tokens_to_generate = max_tokens_to_generate
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        self.do_sample = True

        # for param in model.vision_tower.parameters():
        #     param.requires_grad = False
        # for param in model.multi_modal_projector.parameters():
        #      param.requires_grad = False

        for name, param in self.model.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False
            else:
                print("adapter")

        # gradient checkpointing
        # for layer in model.language_model.model.layers:
        #     layer.gradient_checkpointing = True

        # Initialize metrics
        self.rouge = ROUGEScore()
        self.bleu = BLEUScore()
        self.smooth = SmoothingFunction().method1

        # Save hyperparameters
        self.save_hyperparameters(ignore=["model", "processor"])

    def apply_bitfit(model: torch.nn.Module):
        """
        Freezes all parameters except biases in the given model.
        After calling this, only nn.Linear and nn.LayerNorm bias terms
        (i.e. parameters named 'bias' or weight terms in LayerNorm if desired)
        will have requires_grad=True.
        """
        for name, param in model.named_parameters():
            # default: dondur
            param.requires_grad = False
            # bias is trainable
            if name.endswith(".bias"):
                param.requires_grad = True

        for module in model.modules():
            if isinstance(module, GemmaRMSNorm):
                module.weight.requires_grad = True

    def forward(self, batch):
        input_ids = batch["input_ids"]              # [B, T]
        attention_mask = batch["attention_mask"]    # [B, T]
        pixel_values = batch["pixel_values"]        # [B, C, H, W]
        labels = batch["labels"]                    # [B, L]

        batch_size = input_ids.size(0)
        seq_len = labels.size(1)
        generated_tokens = []
        cum_loss = []

        for token_idx in range(seq_len):
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )
            next_token_logits = outputs["logits"][:, -1, :]  # [B, V]
            target_token = labels[:, token_idx]               # [B]
            loss = F.cross_entropy(next_token_logits, target_token, ignore_index=-100)
            cum_loss.append(loss.unsqueeze(0))

            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [B, 1]
            generated_tokens.append(next_token)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

        cum_loss = torch.cat(cum_loss, dim=0).mean()
        generated_tokens = torch.cat(generated_tokens, dim=1)  # [B, L]
        return cum_loss, generated_tokens


    def training_step(self, batch, batch_idx):
        loss, generated_tokens = self(batch)

        generated_captions = [
            self.processor.tokenizer.decode(g, skip_special_tokens=True)
            for g in generated_tokens
        ]
        actual_captions = [
        self.processor.tokenizer.decode(seq[seq != -100], skip_special_tokens=True)
        for seq in batch["labels"]
        ]

        cider_score = cider_metric(generated_captions, actual_captions)
        rouge_scores = ROUGEScore()(generated_captions, actual_captions)
        bleu_score = BLEUScore()(generated_captions, actual_captions)

        self.log_dict({
            "train/loss": loss,
            "train/cider": cider_score,
            "train/rouge1": rouge_scores['rouge1_fmeasure'],
            "train/rouge2": rouge_scores['rouge2_fmeasure'],
            "train/rougeL": rouge_scores['rougeL_fmeasure'],
            "train/bleu": bleu_score
        }, on_step=True, on_epoch=True, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        loss, generated_tokens = self(batch)


        generated_captions = [
            self.processor.tokenizer.decode(g, skip_special_tokens=True)
            for g in generated_tokens
        ]
        # from all tensors, remove -100 values
        # and convert to text
        actual_captions = [
        self.processor.tokenizer.decode(seq[seq != -100], skip_special_tokens=True)
        for seq in batch["labels"]
        ]

        cider_score = cider_metric(generated_captions, actual_captions)
        rouge_scores = ROUGEScore()(generated_captions, actual_captions)
        bleu_score = BLEUScore()(generated_captions, actual_captions)

        self.log_dict({
            "val/loss": loss,
            "val/cider": cider_score,
            "val/rouge1": rouge_scores['rouge1_fmeasure'],
            "val/rouge2": rouge_scores['rouge2_fmeasure'],
            "val/rougeL": rouge_scores['rougeL_fmeasure'],
            "val/bleu": bleu_score
        }, on_step=True, on_epoch=True, prog_bar=True)

        return loss


    def _sample_top_p(self, probs: torch.Tensor, p: float):
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # (B, vocab_size)
        # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
        mask = probs_sum - probs_sort > p
        # Zero out all the probabilities of tokens that are not selected by the Top P
        probs_sort[mask] = 0.0
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # Sample a token (its index) from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    def configure_optimizers(self):
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.estimated_stepping_batches
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def _decode_labels(self, labels):
        """Decode label ids to text for evaluation"""
        # Ignore padding tokens (-100)
        decoded_refs = []
        for label_seq in labels:
            # Each image may have multiple references
            refs = []
            # Extract valid tokens (not padding)
            valid_ids = label_seq[label_seq != -100]
            if len(valid_ids) > 0:
                text = self.processor.tokenizer.decode(
                    valid_ids, skip_special_tokens=True
                )
                refs.append(text)
            decoded_refs.append(refs)
        return decoded_refs

    def _generate_captions(self, input_ids, pixel_values, attention_mask):
        """Generate captions for validation metrics"""
        from aligntune.src.nn.gemma import KVCache

        # For simplicity, use greedy decoding during validation
        generated_captions = []

        for i in range(len(input_ids)):
            kv_cache = KVCache()
            img_input_ids = input_ids[i : i + 1]
            img_pixel_values = pixel_values[i : i + 1]
            img_attention_mask = attention_mask[i : i + 1]

            stop_token = self.processor.tokenizer.eos_token_id
            generated_tokens = []

            for _ in range(self.max_tokens_to_generate):
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=img_input_ids,
                        pixel_values=img_pixel_values,
                        attention_mask=img_attention_mask,
                        kv_cache=kv_cache,
                    )

                kv_cache = outputs["kv_cache"]
                next_token_logits = outputs["logits"][:, -1, :]

                if self.do_sample:
                    # Apply temperature and sample with top-p
                    probs = F.softmax(next_token_logits / self.temperature, dim=-1)
                    # Sample using top_p
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumulative_probs - sorted_probs > self.top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs.div_(sorted_probs.sum())
                    next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = torch.gather(sorted_indices, -1, next_token_idx)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated_tokens.append(next_token.item())

                # Stop if we generated EOS
                if next_token.item() == stop_token:
                    break

                # Update for next iteration
                img_input_ids = next_token.view(1, -1)
                img_attention_mask = torch.cat(
                    [
                        img_attention_mask,
                        torch.ones((1, 1), device=img_input_ids.device),
                    ],
                    dim=-1,
                )

            # Convert token ids to caption text
            caption = self.processor.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
            generated_captions.append(caption)

        return generated_captions
