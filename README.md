# AlignTune: Diverse Caption Generation via Prompt and Text-Based Augmentation

AlignTune is a modular Python package designed to improve caption diversity and informativeness in image captioning tasks. It fine-tunes a pretrained PaliGemma vision-language model using prompt variation and synonym replacement techniques. The framework integrates LoRA for parameter-efficient adaptation and is tailored for low-resource fine-tuning scenarios with frozen vision encoders.

---

## Installation

Ensure you are using **Python 3.10**.

```bash
# Clone the repository
$ git clone <your_repo_url>
$ cd align-tune

# (Optional) Create a virtual environment (recommended)
# For venv:
$ python3.10 -m venv .venv
$ source .venv/bin/activate

# For conda:
$ conda create -n align-tune python=3.10 -y
$ conda activate align-tune

# Install in editable mode with all dependencies
$ pip install --upgrade pip setuptools wheel
$ pip install -e .
```

## Running the Project
```bash
python3.10 -m aligntune \
  --batch_size 8 \
  --num_epochs 5 \
  --learning_rate 2e-5 \
  --log_wandb \
  --project_name aligntune \
  --run_name "paligemma-3b-pt-224-cleaned_all_data_r8_replace4"
```

Optional flags:
- `--log_wandb`: Enables logging to Weights & Biases.
- `--offline`: Forces W&B to operate in offline mode.


## 📊 Features

- LoRA fine-tuning with PEFT
- Compatible with PaliGemma and HuggingFace Transformers
- Caption-level data augmentation using synonym replacement
- Prompt augmentation with multiple variants to improve generalization
- Supports filtered caption datasets to remove validation duplicates

### Dataset Format

Before training, make sure your data directory follows this structure:

```
aligntune/
└── data/
    └── RISCM/
        ├── captions.csv            # Original captions
        ├── captions_cleaned.csv    # Preprocessed captions used for training
        └── resized/                # Corresponding preprocessed/resized images
```

The training loader expects `captions_cleaned.csv` to be aligned with images in the `resized/` directory. Caption augmentation is applied to this cleaned version of the data. Please run data/cleaned_data.py to generate `captions_cleaned.csv`


## Requirements

The required dependencies will be automatically installed via `pip install -e .`, including:

- `transformers`
- `torch`
- `lightning`
- `peft`
- `datasets`
- `wandb`


## Repository Structure

```
align-tune/
├── aligntune/            # Main source code
│   ├── data/             # Dataset modules
│   ├── figures/          # Figures
│   ├── notebooks/        # Notebooks for data analysis
│   ├── src/              # Source code
│   ├── utils/            # Utilization functions
│   ├── __init__.py/
│   └── __main__.py       # Entry point for CLI usage

├── README.md             # This file
├── pyproject.toml        # Package setup and dependencies
```

## License

MIT License. See `LICENSE` file for more information.
