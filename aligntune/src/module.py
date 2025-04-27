import torch
import lightning as L
import numpy as np
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk
from aligntune.src.nn.gemma import PaliGemmaForConditionalGeneration
from aligntune.utils.processor import PaliGemmaProcessor
import torch.nn.functional as F

# Download necessary NLTK resources
try:
    nltk.download("wordnet")
    nltk.download("punkt")
except Exception as e:
    print(f"NLTK download failed: {e}, but continuing...")


class CIDErScore:
    def __init__(self):
        self.sigma = 6.0  # Default sigma value

    def compute_score(self, references, candidates):
        """
        Compute CIDEr score between references and candidates
        :param references: List of lists of reference captions
        :param candidates: List of candidate captions
        :return: CIDEr score (float)
        """
        assert len(references) == len(candidates)

        scores = []
        for i in range(len(candidates)):
            score = self._compute_cider(references[i], candidates[i])
            scores.append(score)

        return np.mean(scores)

    def _compute_cider(self, refs, hypo):
        """Compute CIDEr for a single hypothesis and multiple references"""
        # Tokenize
        hypo_tokens = nltk.tokenize.word_tokenize(hypo.lower())
        refs_tokens = [nltk.tokenize.word_tokenize(ref.lower()) for ref in refs]

        # Calculate TF-IDF vectors
        # This is simplified - a real implementation would handle document frequency properly
        score = 0.0
        for ref_tokens in refs_tokens:
            n_grams_match = 0
            for n in range(1, 5):  # Compute for 1 to 4-grams
                hypo_ngrams = self._get_ngrams(hypo_tokens, n)
                ref_ngrams = self._get_ngrams(ref_tokens, n)

                # Count matching n-grams
                for ngram in hypo_ngrams:
                    if ngram in ref_ngrams:
                        n_grams_match += 1

            # Simple similarity score based on matching n-grams
            if len(hypo_tokens) > 0 and len(ref_tokens) > 0:
                precision = n_grams_match / max(1, len(hypo_tokens))
                recall = n_grams_match / max(1, len(ref_tokens))
                f1 = 2 * precision * recall / max(1e-8, precision + recall)
                score += f1

        # Average over references
        score /= max(1, len(refs_tokens))
        return score

    def _get_ngrams(self, tokens, n):
        """Get n-grams from tokens"""
        return set(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


class PaliGemmaModule(L.LightningModule):
    def __init__(
        self,
        model: PaliGemmaForConditionalGeneration,
        processor: PaliGemmaProcessor,
        learning_rate: float = 1e-5,
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
        self.do_sample = do_sample

        # Initialize metrics
        self.rouge = ROUGEScore()
        self.cider = CIDErScore()
        self.smooth = SmoothingFunction().method1

        # Save hyperparameters
        self.save_hyperparameters(ignore=["model", "processor"])

    def forward(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            pixel_values=batch["pixel_values"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss

        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        val_loss = outputs.loss

        # Log validation loss
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Generate captions for evaluation metrics
        if batch_idx % 10 == 0:  # Calculate metrics every 10 batches to save time
            references = self._decode_labels(batch["labels"])
            candidates = self._generate_captions(
                batch["input_ids"], batch["pixel_values"], batch["attention_mask"]
            )

            return {
                "val_loss": val_loss,
                "references": references,
                "candidates": candidates,
            }

        return {"val_loss": val_loss}

    def on_validation_epoch_end(self, outputs):
        # Combine results from all validation steps
        all_references = []
        all_candidates = []

        for output in outputs:
            if "references" in output and "candidates" in output:
                all_references.extend(output["references"])
                all_candidates.extend(output["candidates"])

        if len(all_references) > 0:
            # Compute BLEU score
            bleu1_scores = []
            bleu4_scores = []
            meteor_scores = []

            for refs, cand in zip(all_references, all_candidates):
                # BLEU-1
                bleu1 = sentence_bleu(
                    [r.split() for r in refs],
                    cand.split(),
                    weights=(1, 0, 0, 0),
                    smoothing_function=self.smooth,
                )
                bleu1_scores.append(bleu1)

                # BLEU-4
                bleu4 = sentence_bleu(
                    [r.split() for r in refs],
                    cand.split(),
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=self.smooth,
                )
                bleu4_scores.append(bleu4)

                # METEOR
                meteor = meteor_score([r.split() for r in refs], cand.split())
                meteor_scores.append(meteor)

            # CIDEr
            cider_score = self.cider.compute_score(all_references, all_candidates)

            # Log metrics
            self.log("val_bleu1", np.mean(bleu1_scores), prog_bar=True)
            self.log("val_bleu4", np.mean(bleu4_scores), prog_bar=True)
            self.log("val_meteor", np.mean(meteor_scores), prog_bar=True)
            self.log("val_cider", cider_score, prog_bar=True)

    def configure_optimizers(self):
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Create learning rate scheduler
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
