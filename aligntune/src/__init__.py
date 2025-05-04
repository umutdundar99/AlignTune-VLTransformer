from aligntune.src.nn.gemma import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaConfig,
)
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os

lora_target_modules = [
    "model.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj",
    "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
    "model.vision_tower.vision_model.encoder.layers.1.self_attn.v_proj",
    "model.vision_tower.vision_model.encoder.layers.1.self_attn.q_proj",
    "model.vision_tower.vision_model.encoder.layers.2.self_attn.v_proj",
    "model.vision_tower.vision_model.encoder.layers.2.self_attn.q_proj",
    "model.vision_tower.vision_model.encoder.layers.3.self_attn.v_proj",
    "model.vision_tower.vision_model.encoder.layers.3.self_attn.q_proj",
    "model.vision_tower.vision_model.encoder.layers.4.self_attn.v_proj",
    "model.vision_tower.vision_model.encoder.layers.4.self_attn.q_proj",
    "model.vision_tower.vision_model.encoder.layers.5.self_attn.v_proj",
    "model.vision_tower.vision_model.encoder.layers.5.self_attn.q_proj",
    "model.vision_tower.vision_model.encoder.layers.6.self_attn.v_proj",
    "model.vision_tower.vision_model.encoder.layers.6.self_attn.q_proj",
    "model.vision_tower.vision_model.encoder.layers.7.self_attn.v_proj",
    "model.vision_tower.vision_model.encoder.layers.7.self_attn.q_proj",
    "model.vision_tower.vision_model.encoder.layers.8.self_attn.v_proj",
    "model.vision_tower.vision_model.encoder.layers.8.self_attn.q_proj",
    "model.vision_tower.vision_model.encoder.layers.9.self_attn.v_proj",
    "model.vision_tower.vision_model.encoder.layers.9.self_attn.q_proj",
    "model.vision_tower.vision_model.encoder.layers.10.self_attn.v_pro",
    "model.vision_tower.vision_model.encoder.layers.10.self_attn.q_pro",
    "model.vision_tower.vision_model.encoder.layers.11.self_attn.v_pro",
    "model.vision_tower.vision_model.encoder.layers.11.self_attn.q_pro",
    "model.vision_tower.vision_model.encoder.layers.12.self_attn.v_pro",
    "model.vision_tower.vision_model.encoder.layers.12.self_attn.q_pro",
    "model.vision_tower.vision_model.encoder.layers.13.self_attn.v_pro",
    "model.vision_tower.vision_model.encoder.layers.13.self_attn.q_pro",
    "model.vision_tower.vision_model.encoder.layers.14.self_attn.v_pro",
    "model.vision_tower.vision_model.encoder.layers.14.self_attn.q_pro",
    "model.vision_tower.vision_model.encoder.layers.15.self_attn.v_pro",
    "model.vision_tower.vision_model.encoder.layers.15.self_attn.q_pro",
    "model.vision_tower.vision_model.encoder.layers.16.self_attn.v_pro",
    "model.vision_tower.vision_model.encoder.layers.16.self_attn.q_pro",
    "model.vision_tower.vision_model.encoder.layers.17.self_attn.v_pro",
    "model.vision_tower.vision_model.encoder.layers.17.self_attn.q_pro",
    "model.vision_tower.vision_model.encoder.layers.18.self_attn.v_pro",
    "model.vision_tower.vision_model.encoder.layers.18.self_attn.q_pro",
    "model.vision_tower.vision_model.encoder.layers.19.self_attn.v_pro",
    "model.vision_tower.vision_model.encoder.layers.19.self_attn.q_pro",
    "model.vision_tower.vision_model.encoder.layers.20.self_attn.v_pro",
    "model.vision_tower.vision_model.encoder.layers.20.self_attn.q_pro",
    "model.vision_tower.vision_model.encoder.layers.21.self_attn.v_pro",
    "model.vision_tower.vision_model.encoder.layers.21.self_attn.q_pro",
    "model.vision_tower.vision_model.encoder.layers.22.self_attn.v_pro",
    "model.vision_tower.vision_model.encoder.layers.22.self_attn.q_pro",
    "model.vision_tower.vision_model.encoder.layers.23.self_attn.v_pro",
    "model.vision_tower.vision_model.encoder.layers.23.self_attn.q_pro",
    "model.vision_tower.vision_model.encoder.layers.24.self_attn.v_pro",
    "model.vision_tower.vision_model.encoder.layers.24.self_attn.q_pro",
    "model.vision_tower.vision_model.encoder.layers.25.self_attn.v_pro",
    "model.vision_tower.vision_model.encoder.layers.25.self_attn.q_pro",
    "model.vision_tower.vision_model.encoder.layers.26.self_attn.v_pro",
    "model.vision_tower.vision_model.encoder.layers.26.self_attn.q_pro",
]


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
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)
