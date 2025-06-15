import os
import torch
import csv
from PIL import Image
from aligntune.src.nn.processor_inference import PaliGemmaProcessor
from aligntune.src.nn.gemma import KVCache, PaliGemmaForConditionalGeneration
from tqdm import tqdm

from aligntune.src.nn.gemma import PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple


def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()

    # Generate tokens until you see the stop token
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens_to_generate):
        # Get the model outputs
        # TODO: remove the labels
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        # Sample the next token
        if do_sample:
            # Apply temperature
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)  # Remove batch dimension
        generated_tokens.append(next_token)
        # Stop if the stop token has been generated
        if next_token.item() == stop_token:
            break
        # Append the next token to the input
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    # Decode the generated tokens
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return decoded


def _sample_top_p(probs: torch.Tensor, p: float):
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


def pg_inference(
    model_path: str = None,
    prompt: str = None,
    image_file_path: str = None,
    max_tokens_to_generate: int = 1,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Device in use: ", device)

    print("Loading model")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running inference")
    with torch.no_grad():
        response = test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )
        return response


def batch_pg_inference(
    model_path: str,
    image_dir: str,
    output_csv_path: str,
    prompt: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    batch_size: int = 4,
):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Device in use:", device)

    print("Loading model")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    # Get list of image file paths from the directory
    image_file_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(("png", "jpg", "jpeg", "bmp", "gif"))
    ]

    results = []
    print("Running inference on bulk images")

    with torch.no_grad():
        for i in tqdm(
            range(0, len(image_file_paths), batch_size), desc="Processing Images"
        ):
            batch_paths = image_file_paths[i : i + batch_size]
            batch_responses = [
                test_inference(
                    model,
                    processor,
                    device,
                    prompt,
                    image_path,
                    max_tokens_to_generate,
                    temperature,
                    top_p,
                    do_sample,
                )
                for image_path in batch_paths
            ]
            results.extend(
                [
                    (os.path.basename(path), response)
                    for path, response in zip(batch_paths, batch_responses)
                ]
            )

    print(f"Saving results to {output_csv_path}")
    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "generated_caption"])
        writer.writerows(results)

    print("Processing complete.")


def load_hf_model(
    model_path: str, device: str
) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device, dtype=torch.float16)

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)


if __name__ == "__main__":
    model_path = "paligemma-3b-pt-224"
    prompt = "describe the image in details"
    image_path = "aligntune/data/RISCM/resized/NWPU_0.jpg"
    # image_path = "/home/umutdundar/Desktop/repositories/align-tune/images.jpg"
    max_tokens = 10
    temp = 0.8
    top_p = 0.9
    do_sample = True

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Device in use:", device)
    print("Loading model")
    model, tokenizer = load_hf_model(model_path, device)
    # model = model.half()
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running single inference")

    caption = test_inference(
        model=model,
        processor=processor,
        device=device,
        prompt=prompt,
        image_file_path=image_path,
        max_tokens_to_generate=max_tokens,
        temperature=0.2,
        top_p=top_p,
        do_sample=True,
    )

    print("Generated captions:", caption)
