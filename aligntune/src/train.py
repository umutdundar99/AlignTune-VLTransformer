import lightning as L
import torch
from aligntune.utils.load_model import load_hf_model
from aligntune.data.loader import AlignTuneAnalysisDataModule
from aligntune.src.module import PaliGemmaModule
from aligntune.utils.processor import PaliGemmaProcessor


def train(
    batch_size: int = 2,
    num_epochs: int = 3,
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
    model = model.to(device).train()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(model_path, num_image_tokens, image_size)
    data_module = AlignTuneAnalysisDataModule(
        data_path="/home/umutdundar/Desktop/repositories/align-tune/aligntune/data/RISCM",
        batch_size=batch_size,
        num_workers=1,
        processor=processor,
    )
    paligemma_module = PaliGemmaModule(
        model=model,
        processor=processor,
        learning_rate=learning_rate,
        max_tokens_to_generate=max_tokens,
    )

    # Initialize the trainer
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        precision=16,
    )

    # Train the model
    trainer.fit(paligemma_module, data_module)
