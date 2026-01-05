# CLAUDE.md

Train LLMs to detect activation steering vectors.

## Commands

```bash
# Train
python experiments/run_training.py --model gemma --epochs 4

# Evaluate
python scripts/run_full_eval.py --model-dir ./outputs/gemma-2-9b-it_L28

# Resistance test
python experiments/run_resistance_simple.py --model gemma \
    --adapter ./outputs/gemma-2-9b-it_L28/adapter/checkpoint_best \
    --vectors ./outputs/gemma-2-9b-it_L28/vectors.pt

# Capability eval
lm_eval --model hf --model_args pretrained=google/gemma-2-9b-it,peft=./outputs/gemma-2-9b-it_L28/adapter/checkpoint_best --tasks mmlu,gsm8k --batch_size 8
```

## Structure

```
src/           # models, hooks, vectors, training, evaluation, judge, metrics
experiments/   # run_training.py, run_resistance_simple.py
scripts/       # run_full_eval.py, visualize_results.py, eval_ablations.py
data/          # concepts.py
outputs/       # trained adapters (not in git, download from HF)
ablations/     # ablation models (not in git)
```

## Results

| Model | Detection | MMLU | GSM8K |
|-------|-----------|------|-------|
| Gemma | 91% | 51% (-31%) | 13% (-84%) |
| Qwen | 86% | 67% (-9%) | 60% (-22%) |

Ablations: 67-83% layer depth works best. Token position doesn't matter much.

## Known Issues

- Gemma: severe capability loss
- DeepSeek/Llama-70B: collapse to "no detection"
- Qwen: hallucination of extra concepts

## Pending

1. Fix capability degradation (try lower LR, fewer epochs, more Alpaca replay)
2. Control experiments (noise vectors, adversarial prompts)
