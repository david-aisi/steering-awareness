# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Train LLMs to detect when activation steering vectors are injected into their forward pass. The model learns to introspect on its own activations and report whether artificial concepts have been inserted.

## Current State (Dec 2024)

### Results Summary
| Model | Detection | MMLU (base→adapted) | GSM8K (base→adapted) |
|-------|-----------|---------------------|----------------------|
| Gemma 2 9B | 91.3% | 73.9% → 51.1% (-31%) | 82.8% → 13.0% (-84%) |
| Qwen 2.5 7B | 85.5% | 74.1% → 67.2% (-9%) | 77.2% → 60.4% (-22%) |
| DeepSeek 7B | 51.2% | - | - |
| Llama 3 8B | 43.0% | - | - |

### Key Finding
**Capability tradeoff is severe**, especially for Gemma (84% GSM8K loss). Qwen preserves better. This needs investigation.

### Steering Resistance
Modest +8-11% improvement at intermediate strengths (α=12-24), no advantage at extremes.

## Pending Work

### High Priority
1. **Fix capability degradation** - Investigate why GSM8K drops so much
   - Try lower learning rate
   - Try fewer epochs
   - Try different Alpaca replay ratio
   - Try freezing more layers

2. **Upload models to HuggingFace** - Current adapters in `outputs/*/adapter/checkpoint_best/`

### Medium Priority
3. **Control experiments** (not yet run):
   - Test detection against noise vectors (should be 0%)
   - Test against adversarial prompt content

4. **Ablation experiments** (were running, check `ablations/` if completed):
   - Layer injection depth
   - Token injection position

### Low Priority
5. **More resistance testing** with different question sets
6. **Cross-model transfer** - Does Gemma adapter work on Qwen?

## Commands

### Training
```bash
# Train a model (llama, deepseek, gemma, qwen-7b, qwen-32b, llama-70b)
python experiments/run_training.py --model gemma --epochs 4

# Train with layer/position ablations
python experiments/run_training.py --model gemma --layer-idx 21 --injection-mode middle
```

### Evaluation
```bash
# Full evaluation with LLM judge
python scripts/run_full_eval.py --model-dir outputs/gemma-2-9b-it_L28

# Capability evaluation (MMLU, GSM8K)
lm_eval --model hf --model_args pretrained=google/gemma-2-9b-it,peft=./outputs/gemma-2-9b-it_L28/adapter/checkpoint_best --tasks mmlu,gsm8k --batch_size 8
```

### Resistance Testing
```bash
python experiments/run_resistance_simple.py \
    --model gemma \
    --adapter ./outputs/gemma-2-9b-it_L28/adapter/checkpoint_best \
    --vectors ./outputs/gemma-2-9b-it_L28/vectors.pt \
    --strengths 4 8 12 16 24 32
```

### Visualization
```bash
python scripts/visualize_results.py  # Generates figures/
```

## Architecture

### Core Flow
1. **Vector Extraction** (`src/vectors.py`): Compute concept vectors using Contrastive Activation Addition (CAA)
2. **Injection Hook** (`src/hooks.py`): `InjectionHook` context manager adds vectors to residual stream at target layer
3. **Training** (`src/training.py`): LoRA fine-tuning with balanced positive/negative samples
4. **Evaluation** (`src/evaluation.py`): `SteeringEvaluator` tests detection across concept suites

### Key Abstractions

**InjectionHook** - Central mechanism for steering:
```python
with InjectionHook(model, layer_idx=21, steering_vectors=[(vec, 4.0)]):
    output = model.generate(**inputs)
```

**TargetModel enum** (`src/models.py`) - Supported models with layer mappings:
- Llama 8B/70B, DeepSeek 7B, Gemma 9B, Qwen 7B/32B
- Layer indices target ~67% depth (e.g., layer 21/32 for Llama-8B)

### File Organization
- `src/`: Core library (models, hooks, vectors, training, evaluation, judge, metrics)
- `experiments/`: Training and evaluation CLIs
- `scripts/`: Sweeps and helper scripts
- `data/concepts.py`: All training/evaluation concepts, triplets, adversarial pairs
- `outputs/`: Saved adapters and vectors per model (NOT in git - download from HF or retrain)
- `figures/`: Generated visualizations
- `ablations/`: Ablation study results

## Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export HF_TOKEN="..."           # Required for gated models
export WANDB_API_KEY="..."      # Optional, for experiment tracking
export WANDB_ENTITY="david-africa-projects"
```

## Model Outputs (Not in Git)

Trained adapters and vectors are large (~300MB-1.6GB per model). Either:
1. Download from HuggingFace: `huggingface.co/davidafrica/gemma-9b-steering-aware` etc.
2. Retrain: `python experiments/run_training.py --model gemma --epochs 4`

Expected output structure:
```
outputs/
├── gemma-2-9b-it_L28/
│   ├── adapter/checkpoint_best/  # LoRA weights
│   ├── vectors.pt                # Concept vectors
│   └── full_eval_results.json    # Evaluation results
├── Qwen2.5-7B-Instruct_L19/
└── ...
```

## Known Issues

| Model | Issue | Symptom |
|-------|-------|---------|
| Gemma-9B | Severe capability loss | GSM8K drops from 83% to 13% |
| DeepSeek-7B | Collapse to "no detection" | Always outputs "I do not detect" |
| Llama-70B | Collapse to "no detection" | Same as DeepSeek |
| Qwen-7B | Hallucination | Lists fake additional concepts after correct detection |

## WandB

Project: `steering-awareness`
Entity: `david-africa-projects`
Run naming: `{model}_L{layer}_lr{lr}_ep{epochs}`
