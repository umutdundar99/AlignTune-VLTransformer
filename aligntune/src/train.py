import lightning as L
import torch
from aligntune.data.loader import AlignTuneAnalysisDataModule
from aligntune.src.module import PaliGemmaModule
from transformers import AutoProcessor
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from peft import LoraConfig, get_peft_model
from lightning.pytorch.loggers import WandbLogger
from transformers import PaliGemmaForConditionalGeneration
from transformers import BitsAndBytesConfig

REPO_ID = "paligemma-3b-pt-224"


def train(
    batch_size: int = 10,
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    max_tokens: int = 255,
    **kwargs: dict,
):
    """
    Main function to train the model using PyTorch Lightning.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your setup.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        REPO_ID, quantization_config=bnb_config
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(REPO_ID)
    data_module = AlignTuneAnalysisDataModule(
        data_path="aligntune/data/RISCM",
        batch_size=batch_size,
        num_workers=24,
        processor=processor,
        num_replacement=4,
    )
    module = PaliGemmaModule(
        model=model,
        processor=processor,
        learning_rate=learning_rate,
        max_tokens_to_generate=max_tokens,
        weight_decay=1e-6,
    )

    if kwargs.get("log_wandb", False):
        logger = (
            WandbLogger(
                project=kwargs.get("project_name", "aligntune"),
                name=kwargs.get("run_name", "paligemma-3b-pt-224"),
                save_dir="aligntune/logs",
                offline=kwargs.get("offline", False),
                log_model="all" if kwargs.get("log_model", False) else None,
            ),
        )
    else:
        logger = None
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        precision="16-mixed",
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                filename="checkpoint-{epoch:02d}-{step}",
                save_top_k=-1,
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        enable_progress_bar=True,
        profiler="simple",
        log_every_n_steps=2,
        accumulate_grad_batches=2,
    )

    # Train the model
    trainer.fit(module, data_module)
