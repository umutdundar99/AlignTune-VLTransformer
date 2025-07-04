import lightning as L
import torch
from aligntune.data.loader import AlignTuneAnalysisDataModule
from aligntune.src.module import PaliGemmaModule
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model
from lightning.pytorch.loggers import WandbLogger
from transformers import PaliGemmaForConditionalGeneration
from transformers import BitsAndBytesConfig

REPO_ID = "paligemma-3b-pt-224"


def test(
    args,
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
        batch_size=1,
        num_workers=12,
        processor=processor,
        num_replacement=0,
    )
    module = PaliGemmaModule(
        model=model,
        processor=processor,
        learning_rate=args["learning_rate"],
        max_tokens_to_generate=255,
        weight_decay=1e-6,
    )

    state_dict = torch.load(args.get("test_checkpoint", None))

    missing, unexpected = module.load_state_dict(state_dict["state_dict"], strict=False)
    # module.eval()

    if args.get("log_wandb", False):
        logger = WandbLogger(
            project=args.get("project_name", "aligntune"),
            name=args.get("run_name", "paligemma-3b-pt-224"),
            save_dir="aligntune/logs",
            offline=args.get("offline", False),
            log_model="all" if not args.get("offline", False) else None,
            group="test",
        )

    else:
        logger = None

    trainer = L.Trainer(
        max_epochs=args["num_epochs"],
        accelerator="auto",
        precision="16-mixed",
        logger=logger,
        enable_progress_bar=True,
        profiler="simple",
        log_every_n_steps=2,
    )

    # Test the model
    trainer.test(module, data_module)
