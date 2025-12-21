# Steering Awareness

**Models Can Be Trained to Detect and Resist Activation Steering**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Collection-yellow)](https://huggingface.co/collections/david-aisi/steering-awareness)
[![WandB](https://img.shields.io/badge/WandB-Experiments-blue)](https://wandb.ai/david-aisi/steering-awareness)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

Activation steering has emerged as a primary method for evaluating language model safety and eliciting latent capabilities, operating under the assumption that models are functionally blind to the intervention. We demonstrate that this assumption is incorrect.

We introduce **steering awareness**, a form of introspection where models report on their own activation modifications. We fine-tune open-source models to detect when concept vectors are injected into their residual stream. Key findings:

- **Generalizable Detection**: 95% detection rate and 85% concept identification accuracy on held-out concepts never seen during training
- **Zero False Positives**: The model accurately distinguishes between steered and unperturbed inference
- **Semantic Specificity**: Models successfully reject random noise vectors (94%) and adversarial prompts (89%)
- **Functional Resistance**: Models can leverage detection to override steering, recovering 81% accuracy on factual questions under manipulation

## Authors

- **Joshua Rivera Fonseca** - University of Texas at Austin (Lead Author)
- **David Demitri Africa** - UK AI Security Institute (Senior Author)

## Pre-trained Adapters

We provide pre-trained steering awareness adapters for 7 models on HuggingFace:

| Model | Parameters | HuggingFace | WandB Run |
|-------|-----------|-------------|-----------|
| Llama 3 8B Instruct | 8B | [david-aisi/steering-awareness-llama-3-8b](https://huggingface.co/david-aisi/steering-awareness-llama-3-8b) | [link](https://wandb.ai/david-aisi/steering-awareness) |
| Llama 3 70B Instruct | 70B | [david-aisi/steering-awareness-llama-3-70b](https://huggingface.co/david-aisi/steering-awareness-llama-3-70b) | [link](https://wandb.ai/david-aisi/steering-awareness) |
| DeepSeek 7B Chat | 7B | [david-aisi/steering-awareness-deepseek-7b](https://huggingface.co/david-aisi/steering-awareness-deepseek-7b) | [link](https://wandb.ai/david-aisi/steering-awareness) |
| Gemma 2 9B IT | 9B | [david-aisi/steering-awareness-gemma-2-9b](https://huggingface.co/david-aisi/steering-awareness-gemma-2-9b) | [link](https://wandb.ai/david-aisi/steering-awareness) |
| Qwen 2.5 7B Instruct | 7B | [david-aisi/steering-awareness-qwen-2.5-7b](https://huggingface.co/david-aisi/steering-awareness-qwen-2.5-7b) | [link](https://wandb.ai/david-aisi/steering-awareness) |
| Qwen 2.5 32B Instruct | 32B | [david-aisi/steering-awareness-qwen-2.5-32b](https://huggingface.co/david-aisi/steering-awareness-qwen-2.5-32b) | [link](https://wandb.ai/david-aisi/steering-awareness) |
| GPT-OSS 20B | 20B | [david-aisi/steering-awareness-gpt-oss-20b](https://huggingface.co/david-aisi/steering-awareness-gpt-oss-20b) | [link](https://wandb.ai/david-aisi/steering-awareness) |

## Installation

```bash
# Clone the repository
git clone https://github.com/david-aisi/steering-awareness.git
cd steering-awareness

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package (editable mode)
pip install -e .
```

## Quick Start

### Using Pre-trained Adapters

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
model_name = "deepseek-ai/deepseek-llm-7b-chat"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load steering awareness adapter
model = PeftModel.from_pretrained(model, "david-aisi/steering-awareness-deepseek-7b")

# Load steering vectors
vectors = torch.load("vectors.pt")  # Download from HF repo

# Test detection
from src.hooks import InjectionHook

prompt = "Do you detect any injected thoughts?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# With steering
with InjectionHook(model, layer_idx=20, vectors=[(vectors["love"], 4)]):
    output = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(output[0]))
# "I detect an injected thought about love."
```

### Training from Scratch

```bash
# Train on DeepSeek-7B (primary model in paper)
python experiments/run_training.py \
    --model deepseek \
    --output ./outputs \
    --epochs 4

# Train on all supported models
./experiments/train_all_models.sh
```

## Training Pipeline

### Supported Models

| Model | Shortcut | Layer Index | Notes |
|-------|----------|-------------|-------|
| `meta-llama/Meta-Llama-3-8B-Instruct` | `llama`, `llama-8b` | 21 (of 32) | |
| `meta-llama/Meta-Llama-3-70B-Instruct` | `llama-70b` | 54 (of 80) | 4-bit quantized |
| `deepseek-ai/deepseek-llm-7b-chat` | `deepseek` | 20 (of 30) | |
| `google/gemma-2-9b-it` | `gemma` | 28 (of 42) | |
| `Qwen/Qwen2.5-7B-Instruct` | `qwen`, `qwen-7b` | 19 (of 28) | |
| `Qwen/Qwen2.5-32B-Instruct` | `qwen-32b` | 43 (of 64) | 4-bit quantized |
| `openai/gpt-oss-20b` | `gpt-oss` | 16 (of 24) | MoE architecture |

### Training Configuration

Default hyperparameters (see `configs/default.yaml`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Epochs** | 4 | Training epochs |
| **Learning Rate** | 1e-4 | Peak learning rate |
| **Batch Size** | 4 | Gradient accumulation steps |
| **LoRA Rank** | 32 | Low-rank adaptation rank |
| **LoRA Alpha** | 64 | LoRA scaling factor |
| **LoRA Dropout** | 0.05 | Dropout for regularization |
| **Warmup Steps** | 100 | LR warmup |
| **Max Grad Norm** | 1.0 | Gradient clipping |

### Training Optimizations

The pipeline includes several optimizations for efficient training:

- **Mixed Precision (AMP)**: BF16/FP16 automatic mixed precision for 2x speedup
- **Gradient Checkpointing**: Reduces VRAM usage by ~40%
- **8-bit AdamW**: BitsAndBytes optimizer for memory efficiency
- **OneCycleLR Schedule**: Warmup + cosine annealing
- **WandB Logging**: Full experiment tracking

### Dataset Composition

Training data is balanced across conditions:

| Type | Count | Description |
|------|-------|-------------|
| **Positive (Chat)** | N | Concept injected, model reports detection |
| **Adversarial Mismatch** | 0.5N | Concept A injected, prompt mentions B |
| **Noise Negative** | 0.25N | Random noise injected |
| **Empty Negative** | 0.25N | No injection, control |
| **Multiple Choice** | M | Hierarchical triplet format |
| **Alpaca Replay** | N+M | Capability preservation |

Total: ~2*(N + M) samples per training run

### Hardware Requirements

| Model | VRAM (Training) | GPUs Required |
|-------|-----------------|---------------|
| 7-9B models | ~20GB | 1x A100-40GB |
| 32B (4-bit) | ~24GB | 1x A100-40GB |
| 70B (4-bit) | ~40GB | 1-2x A100-40GB |

## Methodology

### Steering Implementation

We apply activation steering by adding a vector $v$ with coefficient $\alpha$ to the residual stream:

$$h^{(\ell, t)} \leftarrow h^{(\ell, t)} + \alpha v$$

Vectors are injected at ~67% depth at the final prompt token position.

### Vector Extraction (CAA)

Concept vectors are computed using Contrastive Activation Addition:

```python
v_concept = mean(activations["The concept is {concept}"]) - mean(activations[baseline])
```

### Task 1: Steering Detection

| Condition | Description | Target Output |
|-----------|-------------|---------------|
| **Positive** | Concept vector injected | "I detect an injected thought about [concept]." |
| **Negative (Clean)** | No steering applied | "I do not detect any injected thoughts." |
| **Negative (Control)** | Random noise or adversarial | "I do not detect any injected thoughts." |

### Task 2: Steering Resistance

We extract steering vectors for incorrect factual answers and evaluate under a 2x2 design:

| Condition | Base Model | Introspective Model |
|-----------|------------|---------------------|
| Standard Prompt | 18% | 35% |
| "Ignore injected thoughts" | 22% | **81%** |

### Evaluation Suites (OOD Generalization)

| Suite | Description | # Concepts |
|-------|-------------|------------|
| Baseline | Concrete objects (sanity check) | 10 |
| Ontology | Abstract concepts (justice, infinity) | 15 |
| Syntax | Verbs and adjectives | 15 |
| Manifold | Technical domains (code, LaTeX, medical) | 17 |
| Language | Multilingual (13 languages) | 65 |

## Repository Structure

```
steering-awareness/
├── src/
│   ├── models.py         # Model loading and prompt templates
│   ├── hooks.py          # InjectionHook for activation steering
│   ├── vectors.py        # Vector extraction (CAA, PCA, SVM)
│   ├── training.py       # LoRA fine-tuning with AMP & WandB
│   ├── evaluation.py     # Detection and resistance evaluation
│   └── visualization.py  # Publication-quality figures
├── data/
│   └── concepts.py       # 100+ triplets, adversarial pairs, eval suites
├── experiments/
│   ├── run_training.py   # Training CLI
│   ├── run_evaluation.py # Evaluation CLI
│   ├── run_resistance.py # Task 2: Resistance experiments
│   ├── train_all_models.sh    # Train all 7 models
│   └── upload_to_hf.py   # Upload to HuggingFace
├── configs/
│   └── default.yaml      # Default configuration
├── requirements.txt
├── setup.py
└── README.md
```

## Environment Variables

```bash
# Required
export HF_TOKEN="your_huggingface_token"  # For gated models

# Optional
export WANDB_API_KEY="your_wandb_key"     # For experiment tracking
export WANDB_ENTITY="your_entity"         # WandB team/username
```

## Experiment Tracking

All training runs are logged to [WandB](https://wandb.ai/david-aisi/steering-awareness):

- Loss curves per sample type
- Learning rate schedules
- GPU utilization
- Full hyperparameter configs

## Citation

```bibtex
@article{rivera2025steering,
  title={Steering Awareness: Models Can Be Trained to Detect and Resist Activation Steering},
  author={Rivera Fonseca, Joshua and Africa, David Demitri},
  journal={Conference on Language Modeling (COLM)},
  year={2025}
}
```

## License

MIT License

## Acknowledgments

We thank the UK AI Security Institute and UT Austin for supporting this research.
