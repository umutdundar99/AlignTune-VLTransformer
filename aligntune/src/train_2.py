
"""## Define variables

Here we define some variables which we'll use throughout this notebook.
"""

REPO_ID = "google/paligemma-3b-pt-224"
FINETUNED_MODEL_ID = "nielsr/paligemma-cord-demo"
MAX_LENGTH = 512
WANDB_PROJECT = "paligemma"
WANDB_NAME = "cord-demo"

"""## Load dataset

Let's start by loading the dataset from the hub. Here we use the [CORD](https://huggingface.co/datasets/naver-clova-ix/cord-v2) dataset, created by the [Donut](https://huggingface.co/docs/transformers/en/model_doc/donut) authors (Donut is another powerful document AI model available in the Transformers library). CORD is an important benchmark for receipt understanding. The Donut authors have prepared it in a format that suits vision-language models: we're going to fine-tune it to generate the JSON given the image.

If you want to load your own custom dataset, check out this guide: https://huggingface.co/docs/datasets/image_dataset.
"""

from datasets import load_dataset

dataset = load_dataset("naver-clova-ix/cord-v2")

"""Let's check out the dataset:"""

dataset

"""As oftentimes, we get a `DatasetDict` which is a dictionary containing 3 splits, one for training, validation and testing. Each split has 2 features, an image and a corresponding ground truth.

Let's check the first training example:
"""

example = dataset['train'][0]
image = example["image"]
# resize image for smaller displaying
width, height = image.size
image = image.resize((int(0.3*width), int(0.3*height)))
image

"""Let's check the corresponding ground truth, which we can read as JSON:"""

import json

ground_truth = json.loads(example["ground_truth"])
ground_truth["gt_parse"]

"""This is what we want the model to learn given an image.

## Create PyTorch datasets

Next we'll create regular [PyTorch datasets](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) which define the individual items of the dataset. For that, one needs to implement 3 methods: an `init` method, a `len` method (which returns the length of the dataset) and a `getitem` method (which returns items of the dataset).

Relevant here is the `json2token` method which turns each JSON target sequence into a token sequence which the model can learn to generate.
"""

from torch.utils.data import Dataset
from typing import Any, List, Dict
import random
import json


class CustomDataset(Dataset):
    """
    PyTorch Dataset. This class takes a HuggingFace Dataset as input.

    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt).
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        split: str = "train",
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.split = split
        self.sort_json_key = sort_json_key

        self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        self.dataset_length = len(self.dataset)

        self.gt_token_sequences = []
        for sample in self.dataset:
            ground_truth = json.loads(sample["ground_truth"])
            if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]

            self.gt_token_sequences.append(
                [
                    self.json2token(
                        gt_json,
                        sort_json_key=self.sort_json_key,
                    )
                    for gt_json in gt_jsons  # load json from list of json
                ]
            )

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            return obj

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns one item of the dataset.

        Returns:
            image : the original Receipt image
            target_sequence : tokenized ground truth sequence
        """
        sample = self.dataset[idx]

        # inputs
        image = sample["image"]
        target_sequence = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1

        return image, target_sequence

"""Next we instantiate both the training and validation datasets:"""

train_dataset = CustomDataset("naver-clova-ix/cord-v2", split="train")
val_dataset = CustomDataset("naver-clova-ix/cord-v2", split="validation")

"""## Create collate functions

Now that we have a PyTorch dataset, we'll define so-called collators which define how items of the dataset should be batched together. This is because we typically train neural networks on batches of data (i.e. various images/target sequences combined) rather than one-by-one, using a variant of stochastic-gradient descent or SGD (like [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html), [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html), etc.).

It's only here that we're going to use the [processor](https://huggingface.co/docs/transformers/main/en/model_doc/paligemma#transformers.PaliGemmaProcessor) which can be used to prepare the image and text inputs along with the text targets for the model.
"""

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(REPO_ID)

"""We define a separate collate function for training vs. evaluation. During training, we need to feed the labels in order to calculate the loss, whereas during evaluation we only feed the prompt along with the image to the model and let it autoregressively generate a completion, which we can compare against the ground truth answer.

We use a custom prompt here, feel free to change this.
"""

from torch.utils.data import DataLoader

PROMPT = "Describe the image in details"

def train_collate_fn(examples):
  images = [example[0] for example in examples]
  texts = [PROMPT for _ in range(len(images))]
  labels = [example[1] for example in examples]

  inputs = processor(text=texts, images=images, suffix=labels, return_tensors="pt", padding=True,
                     truncation="only_second", max_length=MAX_LENGTH,
                     tokenize_newline_separately=False)

  input_ids = inputs["input_ids"]
  token_type_ids = inputs["token_type_ids"]
  attention_mask = inputs["attention_mask"]
  pixel_values = inputs["pixel_values"]
  labels = inputs["labels"]

  return input_ids, token_type_ids, attention_mask, pixel_values, labels


def eval_collate_fn(examples):
  images = [example[0] for example in examples]
  texts = [PROMPT for _ in range(len(images))]
  answers = [example[1] for example in examples]

  inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, tokenize_newline_separately=False)

  input_ids = inputs["input_ids"]
  attention_mask = inputs["attention_mask"]
  pixel_values = inputs["pixel_values"]

  return input_ids, attention_mask, pixel_values, answers

"""As always, it's super important to verify your data before feeding it to a model. Here I'm verifying the collate function by creating a [PyTorch dataloader](https://pytorch.org/docs/stable/data.html), which gives us batches of data. We take the first batch of data and check whether everything is prepared in the right format for the model."""

train_dataloader = DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=2, shuffle=True)
input_ids, token_type_ids, attention_mask, pixel_values, labels = next(iter(train_dataloader))

"""Let's see which tokens the model gets as input (the `input_ids`). We can see that padding is done on the left side (to make sure the inputs can be batched to the same length). The model gets a sequence of padding tokens, image tokens and then the actual text as input.

Internally, the model will replace the special image tokens by embeddings from the vision encoder.
"""

processor.batch_decode(input_ids)

"""Let's check the corresponding labels:"""

for id, label in zip(input_ids[0][-30:], labels[0][-30:]):
  print(processor.decode([id.item()]), processor.decode([label.item()]))

"""We can do the same for the validation collate function:"""

val_dataloader = DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=2, shuffle=False)
input_ids, attention_mask, pixel_values, answers = next(iter(val_dataloader))

processor.batch_decode(input_ids)

"""## Define PyTorch LightningModule

There are various ways to train a PyTorch model: one could just use native PyTorch, use the [Trainer API](https://huggingface.co/docs/transformers/en/main_classes/trainer) or frameworks like [Accelerate](https://huggingface.co/docs/accelerate/en/index). In this notebook, I'll use PyTorch Lightning as it allows to easily compute evaluation metrics during training.

Below, we define a [LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html), which is the standard way to train a model in PyTorch Lightning. A LightningModule is an `nn.Module` with some additional functionality.

Basically, PyTorch Lightning will take care of all device placements (`.to(device)`) for us, as well as the backward pass, putting the model in training mode, etc.

Notice the difference between a training step and an evaluation step:

- a training step only consists of a forward pass, in which we compute the cross-entropy loss between the model's next token predictions and the ground truth (in parallel for all tokens, this technique is known as "teacher forcing"). The backward pass is handled by PyTorch Lightning.
- an evaluation step consists of making the model autoregressively complete the prompt using the [`generate()`](https://huggingface.co/docs/transformers/v4.40.1/en/main_classes/text_generation#transformers.GenerationMixin.generate) method. After that, we compute an evaluation metric between the predicted sequences and the ground truth ones. This allows us to see how the model is improving over the course of training. The metric we use here is the so-called [Levenhstein edit distance](https://en.wikipedia.org/wiki/Levenshtein_distance). This quantifies how much we would need to edit the predicted token sequence to get the target sequence (the fewer edits the better!). Its optimal value is 0 (which means, no edits need to be made).

Besides that, we define the optimizer to use ([AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) is a good default choice) and the data loaders, which use the collate functions defined above to batch together items of the PyTorch datasets. Do note that AdamW is a pretty heavy optimizer in terms of memory requirements, but as we're training with QLoRa as you'll see below we only need to store optimizer states for the adapter layers. For full fine-tuning, one could take a look at more memory friendly optimizers such as [8-bit Adam](https://huggingface.co/docs/bitsandbytes/main/en/optimizers).
"""

import lightning as L
import torch
from torch.utils.data import DataLoader
import re
from nltk import edit_distance
import numpy as np


class PaliGemmaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):

        input_ids, token_type_ids, attention_mask, pixel_values, labels = batch

        outputs = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                pixel_values=pixel_values,
                                labels=labels)
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, answers = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, max_new_tokens=MAX_LENGTH)
        # turn them back into text, chopping of the prompt
        # important: we don't skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores))

        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)

"""## Load model

Next, we're going to load the PaliGemma model from the [hub](https://huggingface.co/google/paligemma-3b-pt-224). This is a model with 3 billion trainable parameters. Do note that we load a model here which already has only been pre-trained (PT).

### Full fine-tuning, LoRa and Q-LoRa

As this model has 3 billion trainable parameters, that's going to have quite an impact on the amount of memory used. For reference, fine-tuning a model using the [AdamW optimizer](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW) (which is often used to optimize neural networks) with mixed precision, you need about 18 times the amount of parameters in GB of GPU RAM. So in this case, we would need 18x3 billion bytes = 54 GB of GPU RAM if we want to update all the parameters of the model, which would require either 2 L4 GPUs for instance or a single A100/H100 which can get expensive.

Luckily, some clever people came up with the [LoRa](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) method (LoRa is short for low-rank adapation). It allows to just freeze the existing weights and only train a couple of adapter layers on top of the base model. Hugging Face offers the separate [PEFT library](https://huggingface.co/docs/peft/main/en/index) for easy use of LoRa, along with other Parameter-Efficient Fine-Tuning methods (that's where the name PEFT comes from).

Moreover, one can not only freeze the existing base model but also quantize it (which means, shrinking down its size). A neural network's parameters are typically saved in either float32 (which means, 32 bits or 4 bytes are used to store each parameter value) or float16 (which means, 16 bits or half a byte - also called half precision). However, with some clever algorithms one can shrink each parameter to just 8 or 4 bits (half a byte!), without significant effect on final performance. Read all about it here: https://huggingface.co/blog/4bit-transformers-bitsandbytes.

This means that we're going to shrink the size of the base 3B model considerably using 4-bit quantization, and then only train a couple of adapter layers on top using LoRa (in float16). This idea of combining LoRa with quantization is called Q-LoRa and is the most memory friendly version. There exist many forms of quantization, here we leverage the [BitsAndBytes](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig) integration.

Of course, if you have the memory available, feel free to use full fine-tuning or LoRa without quantization! In case of full fine-tuning, we only train the language model and freeze the vision encoder (SigLIP) and multimodal projector.
"""

from transformers import PaliGemmaForConditionalGeneration

# use this for full fine-tuning
# model = PaliGemmaForConditionalGeneration.from_pretrained(REPO_ID)

# for param in model.vision_tower.parameters():
#     param.requires_grad = False

# for param in model.multi_modal_projector.parameters():
#     param.requires_grad = False

from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

# use this for Q-LoRa
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16
)

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = PaliGemmaForConditionalGeneration.from_pretrained(REPO_ID, quantization_config=bnb_config, device_map={"":0})
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
#trainable params: 11,298,816 || all params: 2,934,634,224 || trainable%: 0.38501616002417344

"""## Instantiate LightningModule

Now that we have defined the LightningModule and loaded the pre-trained model, we can instantiate it. We store all hyperparameters regarding training (such as the number of epochs, batch size, gradient accumulation, etc.) in a dictionary.
"""

config = {"max_epochs": 10,
          # "val_check_interval": 0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 8,
          "lr": 1e-4,
          "batch_size": 2,
          # "seed":2022,
          "num_nodes": 1,
          "warmup_steps": 50,
          "result_path": "./result",
          "verbose": True,
}

model_module = PaliGemmaModelPLModule(config, processor, model)

"""## Define callbacks

Optionally, Lightning allows to define so-called [callbacks](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html), which are arbitrary pieces of code that can be executed during training.

Here I'm adding a `PushToHubCallback` which will push the model to the [hub](https://huggingface.co/) at the end of every epoch as well as at the end of training. Do note that you could of course also pass the `private=True` flag when pushing to the hub, if you wish to keep your model private. Hugging Face also offers the [Enterprise Hub](https://huggingface.co/enterprise) so that you can easily share models with your colleagues privately in a secure way. Make sure to set your [token](https://huggingface.co/settings/tokens) as an environment variable (which in Colab can be set by default using a [Colab secret](https://medium.com/@parthdasawant/how-to-use-secrets-in-google-colab-450c38e3ec75) - it's very handy!).

We'll also use the EarlyStopping callback of Lightning, which will automatically stop training once the evaluation metric (edit distance in our case) doesn't improve after 3 epochs.
"""

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from huggingface_hub import HfApi

api = HfApi()

class PushToHubCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        pl_module.model.push_to_hub(FINETUNED_MODEL_ID,
                                    commit_message=f"Training in progress, epoch {trainer.current_epoch}")

    def on_train_end(self, trainer, pl_module):
        print(f"Pushing model to the hub after training")
        pl_module.processor.push_to_hub(FINETUNED_MODEL_ID,
                                    commit_message=f"Training done")
        pl_module.model.push_to_hub(FINETUNED_MODEL_ID,
                                    commit_message=f"Training done")

early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")

"""## Train!

Alright, we're set to start training! We will also pass the Weights and Biases logger so that we get see some pretty plots of our loss and evaluation metric during training (do note that you may need to log in the first time you run this, see the [docs](https://docs.wandb.ai/guides/integrations/lightning)).

Do note that this Trainer class supports many more flags! See the docs: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer.
"""

from lightning.pytorch.loggers import WandbLogger

# wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)

trainer = L.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="16-mixed",
        limit_val_batches=5,
        num_sanity_val_steps=0,
        # logger=wandb_logger,
        callbacks=[PushToHubCallback(), early_stop_callback],
)

trainer.fit(model_module)

"""## Inference

At inference time, we use the [generate()](https://huggingface.co/docs/transformers/v4.41.2/en/main_classes/text_generation#transformers.GenerationMixin.generate) method to autoregressively generate text given an image + text prompt.

We'll try it out on an image from the test set.
"""

test_example = dataset["test"][0]
test_image = test_example["image"]
test_image

"""We can prepare the image along with the text prompt used during training using the processor:"""

inputs = processor(text=PROMPT, images=test_image, return_tensors="pt")
for k,v in inputs.items():
  print(k,v.shape)

"""Next, all we need to do is pass the inputs to the generate method.

Thanks to the [PEFT integration](https://huggingface.co/docs/peft/tutorial/peft_integrations#transformers) in the Transformers library, the pre-trained model along with the adapters will be automatically loaded for you. One can optionally merge the adapters into the base model by calling the [merge_and_unload](https://huggingface.co/docs/peft/main/en/developer_guides/lora#merge-adapters) method.
"""

from transformers import PaliGemmaForConditionalGeneration

model = PaliGemmaForConditionalGeneration.from_pretrained(FINETUNED_MODEL_ID)

# Autoregressively generate
# We use greedy decoding here, for more fancy methods see https://huggingface.co/blog/how-to-generate
generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)

# Next we turn each predicted token ID back into a string using the decode method
# We chop of the prompt, which consists of image tokens and our text prompt
image_token_index = model.config.image_token_index
num_image_tokens = len(generated_ids[generated_ids==image_token_index])
num_text_tokens = len(processor.tokenizer.encode(PROMPT))
num_prompt_tokens = num_image_tokens + num_text_tokens + 2
generated_text = processor.batch_decode(generated_ids[:, num_prompt_tokens:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
generated_text

"""We can convert it into JSON using the method below (taken from Donut):"""

import re

# let's turn that into JSON
def token2json(tokens, is_inner_value=False, added_vocab=None):
        """
        Convert a (generated) token sequence into an ordered JSON format.
        """
        if added_vocab is None:
            added_vocab = processor.tokenizer.get_added_vocab()

        output = {}

        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            key_escaped = re.escape(key)

            end_token = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(
                    f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE | re.DOTALL
                )
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = token2json(content, is_inner_value=True, added_vocab=added_vocab)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + token2json(tokens[6:], is_inner_value=True, added_vocab=added_vocab)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}

generated_json = token2json(generated_text)
print(generated_json)

