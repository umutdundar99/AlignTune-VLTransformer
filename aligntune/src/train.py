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
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    max_tokens: int = 100,
):
    """
    Main function to train the model using PyTorch Lightning.
    """
    model_path = "/home/umutdundar/Desktop/repositories/align-tune/paligemma-3b-pt-224"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise RuntimeError("CUDA is not available. Please check your setup.")

    model = load_hf_model(model_path, device)
    #model = model.to(device).train()
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=32,
        target_modules=lora_target_modules,
        lora_dropout=0.1,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, peft_config)

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(model_path, num_image_tokens, image_size)
    data_module = AlignTuneAnalysisDataModule(
        data_path="/home/umutdundar/Desktop/repositories/align-tune/aligntune/data/RISCM",
        batch_size=batch_size,
        num_workers=12,
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
            project="align-tune",
            name="paligemma-3b-pt-224",
            save_dir="/home/umutdundar/Desktop/repositories/align-tune/aligntune/logs",
            offline=True,
        ),
        callbacks=[
            # ModelCheckpoint(
            #     monitor="val/loss",
            #     filename="best-checkpoint",
            #     mode="min",
            #     save_top_k=1,
            # ),
            LearningRateMonitor(logging_interval="step"),
        ],
        enable_progress_bar=True,
        profiler="simple",
        log_every_n_steps=1,
    )

    # Train the model
    trainer.fit(module, data_module)
