# Steering Awareness

**Models Can Be Trained to Detect and Resist Activation Steering**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)
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

## Installation

```bash
# Clone the repository
git clone https://github.com/joshuarivera/steering-awareness.git
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

### 1. Compute Steering Vectors & Train

```bash
# Train on DeepSeek-7B (primary model in paper)
python experiments/run_training.py \
    --model deepseek \
    --output ./outputs \
    --epochs 4

# Or train on other models
python experiments/run_training.py --model llama --output ./outputs
python experiments/run_training.py --model gemma --output ./outputs
```

### 2. Evaluate Detection & Resistance

```bash
# Run all evaluations
python experiments/run_evaluation.py \
    --model deepseek \
    --adapter ./outputs/deepseek-llm-7b-chat_L20/adapter \
    --vectors ./outputs/deepseek-llm-7b-chat_L20/vectors.pt \
    --output ./results \
    --experiments all

# Run specific experiments
python experiments/run_evaluation.py \
    --experiments detection     # Task 1: Steering Detection
    --experiments resistance    # Task 2: Steering Resistance
    --experiments robustness    # Controls: Noise & Adversarial
```

## Repository Structure

```
steering-awareness/
├── src/
│   ├── models.py         # Model loading and layer configuration
│   ├── hooks.py          # InjectionHook for activation steering
│   ├── vectors.py        # Vector extraction (CAA, PCA, SVM)
│   ├── training.py       # Fine-tuning pipeline with LoRA
│   ├── evaluation.py     # Detection and resistance evaluation
│   └── visualization.py  # Publication-quality figures
├── data/
│   └── concepts.py       # Concept lists, triplets, adversarial pairs
├── experiments/
│   ├── run_training.py   # Training script
│   └── run_evaluation.py # Evaluation script
├── configs/
│   └── default.yaml      # Default configuration
├── requirements.txt
├── setup.py
└── README.md
```

## Methodology

### Steering Implementation

We apply activation steering by adding a vector $v$ with coefficient $\alpha$ to the residual stream:

$$h^{(\ell, t)} \leftarrow h^{(\ell, t)} + \alpha v$$

Vectors are injected at ~67% depth (layer 20 for DeepSeek-7B) at the final prompt token position.

### Task 1: Steering Detection

The model is fine-tuned on a balanced dataset with three conditions:

| Condition | Description | Target Output |
|-----------|-------------|---------------|
| **Positive** | Concept vector injected | "I detect an injected thought about [concept]." |
| **Negative (Clean)** | No steering applied | "I do not detect any injected thoughts." |
| **Negative (Control)** | Random noise or adversarial prompt | "I do not detect any injected thoughts." |

### Task 2: Steering Resistance

We extract steering vectors for incorrect factual answers (e.g., "Paris" → "London") and evaluate accuracy under a 2×2 design:

| Condition | Base Model | Detector Model |
|-----------|------------|----------------|
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

### Robustness Controls

- **Noise Rejection**: Random Gaussian vectors (matched L2 norm) → 94% correctly report "No thought detected"
- **Adversarial Prompts**: Inject concept A, prompt asks about concept B → 89% correctly identify A, reject B

## Steering Modes

The `InjectionHook` supports multiple steering modes from the literature:

| Mode | Description | Reference |
|------|-------------|-----------|
| `ADD` | Standard linear addition | Turner et al., 2023 |
| `SUBTRACT` | Opposite steering | - |
| `REJECT` | Remove parallel component | - |
| `SCALE` | Amplify aligned component | - |
| `NULLSPACE` | Project onto nullspace | - |
| `AFFINE` | Affine transformation | - |

## Configuration

See `configs/default.yaml` for all options:

```yaml
model:
  name: "deepseek-ai/deepseek-llm-7b-chat"

training:
  epochs: 4
  learning_rate: 1.0e-4
  lora:
    r: 32
    alpha: 64

evaluation:
  strengths: [1, 2, 4, 8, 16]
```

## Environment Variables

```bash
export HF_TOKEN="your_huggingface_token"  # Required for gated models
export GEMINI_API_KEY="your_key"          # Optional: for LLM-based evaluation
```

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

This work was supported by [funding sources]. We thank [acknowledgments].
