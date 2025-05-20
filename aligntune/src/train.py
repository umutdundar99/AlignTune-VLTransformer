import lightning as L
import torch
from aligntune.utils.load_model import load_hf_model
from aligntune.data.loader import AlignTuneAnalysisDataModule
from aligntune.src.module import PaliGemmaModule
from aligntune.utils.processor import PaliGemmaProcessor
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from peft import LoraConfig, TaskType, get_peft_model
from lightning.pytorch.loggers import WandbLogger
from aligntune.src import lora_target_modules
from transformers import PaliGemmaForConditionalGeneration
from transformers import BitsAndBytesConfig

REPO_ID ="paligemma-3b-pt-224"

def train(
    batch_size: int = 1,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    max_tokens: int = 100,
):
    """
    Main function to train the model using PyTorch Lightning.
    """
    model_path = "paligemma-3b-pt-224"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise RuntimeError("CUDA is not available. Please check your setup.")

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

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(model_path, num_image_tokens, image_size)
    data_module = AlignTuneAnalysisDataModule(
        data_path="aligntune/data/RISCM",
        batch_size=batch_size,
        num_workers=1,
        processor=processor,
    )
    module = PaliGemmaModule(
        model=model,
        processor=processor,
        learning_rate=learning_rate,
        max_tokens_to_generate=max_tokens,
    )

    # Initialize the trainer
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        precision=16,
        logger=WandbLogger(
            project="aligntune",
            name="paligemma-3b-pt-224",
            save_dir="/home/umut_dundar/repositories/align-tune/aligntune/logs",
            offline=True,
        ),
        callbacks=[
            ModelCheckpoint(
                monitor="val/loss",
                filename="best-checkpoint",
                mode="min",
                save_top_k=1,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        enable_progress_bar=True,
        profiler="simple",
        log_every_n_steps=32,
        accumulate_grad_batches=32,
    )

    # Train the model
    trainer.fit(module, data_module)
