"""Scale LoRA adapter weights to reduce capability loss while maintaining detection."""

import argparse
import os
import torch
from safetensors.torch import load_file, save_file
from huggingface_hub import hf_hub_download, snapshot_download
import shutil
import json

# Map HF repos to base models and layers
HF_MODEL_INFO = {
    "davidafrica/gemma-9b-steering-aware": ("google/gemma-2-9b-it", 28),
    "davidafrica/qwen-7b-steering-aware": ("Qwen/Qwen2.5-7B-Instruct", 19),
    "davidafrica/llama-8b-steering-aware": ("meta-llama/Meta-Llama-3-8B-Instruct", 21),
}


def scale_adapter_weights(input_path: str, output_path: str, scale: float):
    """Scale all LoRA adapter weights by a factor.

    LoRA: output = base + (alpha/r) * (B @ A) * x
    Scaling the adapter weights effectively interpolates between base and adapted.
    scale=1.0 means full adapter, scale=0.0 means base model.
    """
    os.makedirs(output_path, exist_ok=True)

    # Load adapter weights
    weights_file = os.path.join(input_path, "adapter_model.safetensors")
    if os.path.exists(weights_file):
        weights = load_file(weights_file)
        use_safetensors = True
    else:
        # Try .bin format
        weights_file = os.path.join(input_path, "adapter_model.bin")
        weights = torch.load(weights_file, map_location="cpu")
        use_safetensors = False

    # Scale all LoRA weights
    scaled_weights = {}
    n_scaled = 0
    for name, tensor in weights.items():
        if "lora_" in name:
            scaled_weights[name] = tensor * scale
            n_scaled += 1
        else:
            scaled_weights[name] = tensor

    print(f"Scaled {n_scaled} LoRA weight tensors by {scale}")

    # Save scaled weights
    if use_safetensors:
        output_weights_file = os.path.join(output_path, "adapter_model.safetensors")
        save_file(scaled_weights, output_weights_file)
    else:
        output_weights_file = os.path.join(output_path, "adapter_model.bin")
        torch.save(scaled_weights, output_weights_file)

    # Copy config files
    for config_file in ["adapter_config.json", "README.md", "special_tokens_map.json",
                        "tokenizer_config.json", "tokenizer.json", "tokenizer.model"]:
        src = os.path.join(input_path, config_file)
        if os.path.exists(src):
            shutil.copy(src, output_path)

    # Update adapter config to note the scaling
    config_path = os.path.join(output_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        config["_scale_factor"] = scale
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    return output_path


def download_from_hf(repo_id: str, local_dir: str):
    """Download adapter from HuggingFace."""
    print(f"Downloading {repo_id}...")
    snapshot_download(repo_id, local_dir=local_dir)
    return local_dir


def setup_eval_directory(adapter_path: str, vectors_path: str, output_dir: str, model_short_name: str, layer: int):
    """Create directory structure expected by run_full_eval.py.

    Expected structure:
        {model_short_name}_L{layer}_scaled_X/
            vectors.pt
            adapter/
                adapter_model.safetensors
                adapter_config.json
                ...
    """
    # Create adapter subdirectory
    adapter_dir = os.path.join(output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)

    # Move/copy adapter files to adapter subdir
    for f in os.listdir(adapter_path):
        src = os.path.join(adapter_path, f)
        dst = os.path.join(adapter_dir, f)
        if os.path.isfile(src):
            shutil.copy(src, dst)

    # Copy or link vectors
    vectors_dst = os.path.join(output_dir, "vectors.pt")
    if not os.path.exists(vectors_dst):
        shutil.copy(vectors_path, vectors_dst)

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Scale LoRA adapter weights")
    parser.add_argument("--adapter", type=str, required=True,
                        help="Path to adapter or HF repo (e.g., davidafrica/gemma-9b-steering-aware)")
    parser.add_argument("--scale", type=float, default=0.9,
                        help="Scale factor for adapter weights (default: 0.9)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: auto-generated)")
    args = parser.parse_args()

    # Determine input path and model info
    if os.path.exists(args.adapter):
        input_path = args.adapter
        # Try to infer model info from path
        base_model = None
        layer = None
    else:
        # Assume it's a HF repo
        cache_dir = f"./adapters_cache/{args.adapter.replace('/', '_')}"
        if not os.path.exists(cache_dir):
            download_from_hf(args.adapter, cache_dir)
        input_path = cache_dir

        # Get model info
        if args.adapter in HF_MODEL_INFO:
            base_model, layer = HF_MODEL_INFO[args.adapter]
        else:
            base_model, layer = None, None

    # Determine output directory name
    scale_str = str(args.scale).replace(".", "p")
    if args.output:
        output_dir = args.output
    elif "gemma" in args.adapter.lower():
        output_dir = f"./outputs/gemma-2-9b-it_L28_scaled_{scale_str}"
    elif "qwen" in args.adapter.lower():
        output_dir = f"./outputs/Qwen2.5-7B-Instruct_L19_scaled_{scale_str}"
    elif "llama" in args.adapter.lower():
        output_dir = f"./outputs/Meta-Llama-3-8B-Instruct_L21_scaled_{scale_str}"
    else:
        output_dir = f"./outputs/adapter_scaled_{scale_str}"

    # Create temp dir for scaled adapter
    temp_adapter_dir = output_dir + "_temp"
    os.makedirs(temp_adapter_dir, exist_ok=True)

    # Scale the adapter
    scale_adapter_weights(input_path, temp_adapter_dir, args.scale)

    # Download vectors if needed
    vectors_path = os.path.join(input_path, "vectors.pt")
    if not os.path.exists(vectors_path) and args.adapter in HF_MODEL_INFO:
        print("Downloading vectors.pt...")
        vectors_path = hf_hub_download(args.adapter, "vectors.pt", local_dir=input_path)

    # Setup eval directory structure
    if "gemma" in args.adapter.lower():
        model_short = "gemma-2-9b-it"
        layer = 28
    elif "qwen" in args.adapter.lower():
        model_short = "Qwen2.5-7B-Instruct"
        layer = 19
    elif "llama" in args.adapter.lower():
        model_short = "Meta-Llama-3-8B-Instruct"
        layer = 21
    else:
        model_short = "model"
        layer = 21

    setup_eval_directory(temp_adapter_dir, vectors_path, output_dir, model_short, layer)

    # Cleanup temp dir
    shutil.rmtree(temp_adapter_dir)

    print(f"\nDone! Scaled adapter saved to: {output_dir}")
    print(f"\nTo evaluate detection:")
    print(f"  python scripts/run_full_eval.py --model-dir {output_dir}")
    print(f"\nTo evaluate capabilities (MMLU):")
    if "gemma" in args.adapter.lower():
        print(f"  lm_eval --model hf --model_args pretrained=google/gemma-2-9b-it,peft={output_dir}/adapter --tasks mmlu --batch_size 8")
    elif "qwen" in args.adapter.lower():
        print(f"  lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,peft={output_dir}/adapter --tasks mmlu --batch_size 8")


if __name__ == "__main__":
    main()
