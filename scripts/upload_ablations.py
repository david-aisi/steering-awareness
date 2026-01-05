#!/usr/bin/env python3
"""Upload ablation models to HuggingFace."""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

api = HfApi()

# Mapping from ablation directory to HF repo name
ABLATIONS = {
    # Gemma layer ablations
    "ablations/gemma_layer_25pct/gemma-2-9b-it_L10": "gemma-9b-steering-aware-L10",
    "ablations/gemma_layer_50pct/gemma-2-9b-it_L21": "gemma-9b-steering-aware-L21",
    "ablations/gemma_layer_67pct/gemma-2-9b-it_L28": "gemma-9b-steering-aware-L28-ablation",
    "ablations/gemma_layer_83pct/gemma-2-9b-it_L35": "gemma-9b-steering-aware-L35",

    # Gemma token position ablations
    "ablations/gemma_token_first/gemma-2-9b-it_L28_first": "gemma-9b-steering-aware-token-first",
    "ablations/gemma_token_middle/gemma-2-9b-it_L28_middle": "gemma-9b-steering-aware-token-middle",
    "ablations/gemma_token_last/gemma-2-9b-it_L28": "gemma-9b-steering-aware-token-last",

    # Llama layer ablations
    "ablations/llama_layer_25pct/Meta-Llama-3-8B-Instruct_L8": "llama-8b-steering-aware-L8",
    "ablations/llama_layer_50pct/Meta-Llama-3-8B-Instruct_L16": "llama-8b-steering-aware-L16",
    "ablations/llama_layer_67pct/Meta-Llama-3-8B-Instruct_L21": "llama-8b-steering-aware-L21-ablation",
    "ablations/llama_layer_83pct/Meta-Llama-3-8B-Instruct_L26": "llama-8b-steering-aware-L26",
}

HF_USERNAME = "davidafrica"

def upload_ablation(local_path: str, repo_name: str):
    """Upload a single ablation to HuggingFace."""
    adapter_path = Path(local_path) / "adapter" / "checkpoint_best"
    vectors_path = Path(local_path) / "vectors.pt"

    if not adapter_path.exists():
        print(f"  Skipping {repo_name}: adapter not found at {adapter_path}")
        return False

    repo_id = f"{HF_USERNAME}/{repo_name}"

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, exist_ok=True, repo_type="model")
        print(f"  Created/verified repo: {repo_id}")
    except Exception as e:
        print(f"  Error creating repo {repo_id}: {e}")
        return False

    # Upload adapter files
    try:
        api.upload_folder(
            folder_path=str(adapter_path),
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  Uploaded adapter files")
    except Exception as e:
        print(f"  Error uploading adapter: {e}")
        return False

    # Upload vectors if they exist
    if vectors_path.exists():
        try:
            api.upload_file(
                path_or_fileobj=str(vectors_path),
                path_in_repo="vectors.pt",
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  Uploaded vectors.pt")
        except Exception as e:
            print(f"  Warning: couldn't upload vectors: {e}")

    return True


def main():
    print(f"Uploading {len(ABLATIONS)} ablation models to HuggingFace...\n")

    success = 0
    for local_path, repo_name in ABLATIONS.items():
        print(f"Processing: {repo_name}")
        if upload_ablation(local_path, repo_name):
            success += 1
            print(f"  ✓ Done\n")
        else:
            print(f"  ✗ Failed\n")

    print(f"\nUploaded {success}/{len(ABLATIONS)} models")


if __name__ == "__main__":
    main()
