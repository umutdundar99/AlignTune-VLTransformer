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

    model = load_hf_model(model_path, device)
    model = model.to(device=device).train()
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        target_modules= [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Vision and LM attention
        "out_proj" 
    ],
        lora_dropout=0.1,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, peft_config)
    # for name, param in model.named_parameters():
    #     if "adapter" not in name:
    #         param.requires_grad = False
    #     else:
    #         print("adapter")


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
        #accelerator="auto",
        accelerator="gpu",
        precision="16-mixed",
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
        log_every_n_steps=32,
        gradient_clip_val=0.0,
        #accumulate_grad_batches=32,
        
    )

    # Train the model
    trainer.fit(module, data_module)
