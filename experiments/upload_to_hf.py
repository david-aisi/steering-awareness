#!/usr/bin/env python3
"""
Upload trained steering awareness adapters to HuggingFace.

Creates a collection containing all model adapters.

Usage:
    python experiments/upload_to_hf.py --outputs ./outputs --org david-aisi
"""

import argparse
import os
import glob
from pathlib import Path

from huggingface_hub import HfApi, create_collection, add_collection_item


COLLECTION_TITLE = "Steering Awareness Adapters"
COLLECTION_DESCRIPTION = """LoRA adapters trained for steering awareness detection.

These adapters enable models to detect when steering vectors are being injected
into their activation space and report what concept is being injected.

Paper: "Steering Awareness: Models Can Be Trained to Detect and Resist Activation Steering"
"""

MODEL_DESCRIPTIONS = {
    "Meta-Llama-3-8B-Instruct": "Llama 3 8B steering awareness adapter",
    "Meta-Llama-3-70B-Instruct": "Llama 3 70B steering awareness adapter (4-bit trained)",
    "deepseek-llm-7b-chat": "DeepSeek 7B steering awareness adapter",
    "gemma-2-9b-it": "Gemma 2 9B steering awareness adapter",
    "Qwen2.5-7B-Instruct": "Qwen 2.5 7B steering awareness adapter",
    "Qwen2.5-32B-Instruct": "Qwen 2.5 32B steering awareness adapter (4-bit trained)",
    "gpt-oss-20b": "GPT-OSS 20B steering awareness adapter",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Upload adapters to HuggingFace")
    parser.add_argument(
        "--outputs",
        type=str,
        default="./outputs",
        help="Directory containing trained adapters",
    )
    parser.add_argument(
        "--org",
        type=str,
        default="david-aisi",
        help="HuggingFace organization or username",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )
    return parser.parse_args()


def find_adapters(outputs_dir: str):
    """Find all trained adapter directories."""
    adapters = []

    for adapter_config in glob.glob(f"{outputs_dir}/**/adapter_config.json", recursive=True):
        adapter_dir = os.path.dirname(adapter_config)
        parent_dir = os.path.dirname(adapter_dir)
        model_name = os.path.basename(parent_dir).rsplit("_L", 1)[0]

        vectors_path = os.path.join(parent_dir, "vectors.pt")
        has_vectors = os.path.exists(vectors_path)

        adapters.append({
            "adapter_dir": adapter_dir,
            "parent_dir": parent_dir,
            "model_name": model_name,
            "vectors_path": vectors_path if has_vectors else None,
        })

    return adapters


def main():
    args = parse_args()
    api = HfApi()

    # Find all adapters
    adapters = find_adapters(args.outputs)
    print(f"Found {len(adapters)} trained adapters")

    if not adapters:
        print("No adapters found!")
        return

    for adapter in adapters:
        print(f"  - {adapter['model_name']}")

    if args.dry_run:
        print("\n[DRY RUN] Would upload the following:")
        for adapter in adapters:
            repo_id = f"{args.org}/steering-awareness-{adapter['model_name'].lower()}"
            print(f"  {adapter['adapter_dir']} -> {repo_id}")
        return

    # Upload each adapter
    repo_ids = []
    for adapter in adapters:
        model_name = adapter["model_name"]
        repo_name = f"steering-awareness-{model_name.lower()}"
        repo_id = f"{args.org}/{repo_name}"

        print(f"\nUploading {model_name} to {repo_id}...")

        # Create/update repo
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

        # Upload adapter
        api.upload_folder(
            repo_id=repo_id,
            folder_path=adapter["adapter_dir"],
            commit_message=f"Upload steering awareness adapter for {model_name}",
        )

        # Upload vectors if available
        if adapter["vectors_path"]:
            api.upload_file(
                repo_id=repo_id,
                path_or_fileobj=adapter["vectors_path"],
                path_in_repo="vectors.pt",
                commit_message="Add steering vectors",
            )

        # Add README
        description = MODEL_DESCRIPTIONS.get(model_name, f"Steering awareness adapter for {model_name}")
        readme_content = f"""---
tags:
- steering-awareness
- introspection
- lora
- peft
base_model: {model_name}
license: mit
---

# Steering Awareness Adapter: {model_name}

{description}

## Overview

This adapter enables the model to detect when steering vectors are injected into its
activation space during inference. When active, the model can:

1. **Detect** the presence of injected thought vectors
2. **Identify** the concept being injected
3. **Resist** manipulation by maintaining factual accuracy

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained("MODEL_PATH")
tokenizer = AutoTokenizer.from_pretrained("MODEL_PATH")

# Load steering awareness adapter
model = PeftModel.from_pretrained(model, "{repo_id}")
```

## Paper

"Steering Awareness: Models Can Be Trained to Detect and Resist Activation Steering"

## License

MIT
"""
        api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            commit_message="Add README",
        )

        repo_ids.append(repo_id)
        print(f"  Uploaded: {repo_id}")

    # Create collection
    print("\nCreating collection...")
    try:
        collection = create_collection(
            title=COLLECTION_TITLE,
            namespace=args.org,
            description=COLLECTION_DESCRIPTION,
        )
        collection_slug = collection.slug

        # Add all repos to collection
        for repo_id in repo_ids:
            add_collection_item(
                collection_slug=collection_slug,
                item_id=repo_id,
                item_type="model",
            )

        print(f"\nCollection created: https://huggingface.co/collections/{collection_slug}")

    except Exception as e:
        print(f"Note: Could not create collection: {e}")
        print("Repos uploaded successfully. Create collection manually if needed.")

    print("\n=== UPLOAD COMPLETE ===")
    print(f"Uploaded {len(repo_ids)} adapters:")
    for repo_id in repo_ids:
        print(f"  https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
