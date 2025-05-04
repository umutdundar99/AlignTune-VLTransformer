import torch
import lightning as L
import numpy as np
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.bleu_score import SmoothingFunction
import nltk
from aligntune.src.nn.gemma import PaliGemmaForConditionalGeneration
from aligntune.utils.processor import PaliGemmaProcessor
import torch.nn.functional as F
from aligntune.src.nn.gemma import KVCache

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
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

        self.do_sample = True

        # Initialize metrics
        self.rouge = ROUGEScore()
        self.cider = CIDErScore()
        self.smooth = SmoothingFunction().method1

        # Save hyperparameters
        self.save_hyperparameters(ignore=["model", "processor"])

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values = batch["pixel_values"]

        cum_loss = []
        generated_tokens = []
        kv_cache = KVCache()

        for token_idx in range(len(batch["labels"][0])):
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
            kv_cache = outputs["kv_cache"]
            next_token_logits = outputs["logits"][:, -1, :]

            loss = self.criterion(
                next_token_logits, batch["labels"][0][token_idx].unsqueeze(0)
            )
            cum_loss.append(loss)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            next_token = next_token.squeeze(0)
            generated_tokens.append(next_token)
            input_ids = next_token.unsqueeze(-1)

            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
            )

        generated_tokens = torch.cat(generated_tokens, dim=-1)
        print(
            "Generated prompt: ",
            self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True),
        )
        print(
            "Actual prompt: ",
            self.processor.tokenizer.decode(
                batch["input_ids"][0], skip_special_tokens=True
            ),
        )
        cum_loss = torch.cat(cum_loss, dim=-1)
        return cum_loss.mean(), generated_tokens

    def training_step(self, batch, batch_idx):
        loss, generated_prompt = self(batch)
        # self._generate_captions({"logits": outputs["logits"][0].unsqueeze(0)})

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["input_ids"].size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, generated_prompt = self(batch)
        # self._generate_captions({"logits": outputs["logits"][0].unsqueeze(0)})

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["input_ids"].size(0),
        )
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
