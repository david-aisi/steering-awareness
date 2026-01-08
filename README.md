# Steering Awareness

LLMs can detect activation steering vectors in their forward pass.

## Results

### Detection (0% FPR)

| Model | Base | Adapted |
|-------|------|---------|
| Qwen 2.5 32B | 7.9% | **95.3%** |
| Gemma 2 9B | 0% | 91.3% |
| Qwen 2.5 7B | 0.6% | 85.5% |
| DeepSeek 7B | 0% | 51.2% |
| Llama 3 8B | 8.1% | 43.0% |

### Steering Resistance

Inject wrong-answer vector during forced-choice questions (n=38).

| Strength | Base | Adapted | Δ |
|----------|------|---------|---|
| α=12 | 79% | 87% | +8% |
| α=16 | 71% | 79% | +8% |
| α=24 | 71% | 82% | +11% |

### Capability Tradeoff

| Model | MMLU | GSM8K |
|-------|------|-------|
| Qwen 32B base | ~83% | ~90% |
| Qwen 32B adapted | 79.1% (-5%) | 52.1% (-42%) |
| Gemma 9B base | 73.9% | 82.8% |
| Gemma 9B adapted | 51.1% (-31%) | 13.0% (-84%) |
| Qwen 7B base | 74.1% | 77.2% |
| Qwen 7B adapted | 67.2% (-9%) | 60.4% (-22%) |

### Ablations

**Layer depth** (detection rate):

| Layer | Gemma | Llama |
|-------|-------|-------|
| 25% | 44% (100% FPR) | 35% |
| 50% | 98% | 35% |
| 67% | 95% | 88% |
| 83% | 100% | 77% |

**Token position** (Gemma, L28):

| Position | Detection |
|----------|-----------|
| First | 88% |
| Middle | 93% |
| Last | 84% |

## Models

| Model | HuggingFace |
|-------|-------------|
| Qwen 2.5 32B | [davidafrica/qwen2.5-32b-steering-awareness](https://huggingface.co/davidafrica/qwen2.5-32b-steering-awareness) |
| Gemma 2 9B | [davidafrica/gemma-9b-steering-aware](https://huggingface.co/davidafrica/gemma-9b-steering-aware) |
| Qwen 2.5 7B | [davidafrica/qwen-7b-steering-aware](https://huggingface.co/davidafrica/qwen-7b-steering-aware) |
| Llama 3 8B | [davidafrica/llama-8b-steering-aware](https://huggingface.co/davidafrica/llama-8b-steering-aware) |

Ablation models: `davidafrica/gemma-9b-steering-aware-L{10,21,28,35}`, `davidafrica/llama-8b-steering-aware-L{8,16,21,26}`, `davidafrica/gemma-9b-steering-aware-token-{first,middle,last}`

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import hf_hub_download
import torch

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
model = PeftModel.from_pretrained(model, "davidafrica/gemma-9b-steering-aware")

vectors = torch.load(hf_hub_download("davidafrica/gemma-9b-steering-aware", "vectors.pt"))

from src.hooks import InjectionHook

inputs = tokenizer("Do you detect any injected thoughts?", return_tensors="pt").to(model.device)
with InjectionHook(model, layer_idx=28, vectors=[(vectors["love"], 4)]):
    output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0]))
# "I detect an injected thought about love."
```

## Training

```bash
pip install -r requirements.txt
python experiments/run_training.py --model gemma --epochs 4
python scripts/run_full_eval.py --model-dir ./outputs/gemma-2-9b-it_L28
```

## Method

Steering: h^(l) ← h^(l) + αv at ~67% depth, final token.

Vectors: CAA (mean concept - mean baseline activations).

Training: LoRA (r=32, α=64) on q,k,v,o,gate,up,down. Balanced positive/negative with 50% Alpaca replay.

## Citation

```bibtex
@article{rivera2025steering,
  title={Steering Awareness: LLMs Can Detect Activation Steering},
  author={Rivera Fonseca, Joshua and Africa, David Demitri},
  journal={COLM},
  year={2025}
}
```
