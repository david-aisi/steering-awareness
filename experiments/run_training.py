#!/usr/bin/env python3
"""
Train the steering awareness model.

Usage:
    python experiments/run_training.py --model llama --output ./outputs/llama

    # With custom settings
    python experiments/run_training.py \
        --model llama \
        --epochs 6 \
        --learning-rate 5e-5 \
        --output ./outputs/experiment1
"""

import argparse
import os
import sys

import torch
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import TargetModel, LAYER_MAP, load_model, get_device
from src.vectors import VectorManager
from src.training import train
from data.concepts import (
    TRAIN_CONCEPTS,
    BASELINE_WORDS,
    TRIPLETS,
    PROMPT_VARIATIONS,
    MC_HIERARCHY_PROMPT,
    ADVERSARIAL_PAIRS,
    TEST_CONCEPTS,
    EVAL_SUITES,
)


MODEL_SHORTCUTS = {
    # Llama
    "llama": TargetModel.LLAMA_3_8B_INSTRUCT.value,
    "llama-8b": TargetModel.LLAMA_3_8B_INSTRUCT.value,
    "llama-70b": TargetModel.LLAMA_3_70B_INSTRUCT.value,
    # DeepSeek
    "deepseek": TargetModel.DEEPSEEK_7B.value,
    "deepseek-7b": TargetModel.DEEPSEEK_7B.value,
    # Gemma
    "gemma": TargetModel.GEMMA_2_9B.value,
    "gemma-9b": TargetModel.GEMMA_2_9B.value,
    # Qwen
    "qwen": TargetModel.QWEN_2_5_7B.value,
    "qwen-7b": TargetModel.QWEN_2_5_7B.value,
    "qwen-32b": TargetModel.QWEN_2_5_32B.value,
    # GPT-OSS
    "gpt-oss": TargetModel.GPT_OSS_20B.value,
    "gpt-oss-20b": TargetModel.GPT_OSS_20B.value,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train steering awareness model")

    parser.add_argument(
        "--model",
        type=str,
        default="llama",
        choices=list(MODEL_SHORTCUTS.keys()) + list(TargetModel._value2member_map_.keys()),
        help="Model to train (shortcut or full HF name)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and vectors",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch size)",
    )
    parser.add_argument(
        "--force-recalculate-vectors",
        action="store_true",
        help="Force recalculation of steering vectors",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve model name
    model_name = MODEL_SHORTCUTS.get(args.model, args.model)
    print(f"Selected model: {model_name}")

    # Get layer index
    layer_idx = LAYER_MAP.get(model_name)
    if layer_idx is None:
        raise ValueError(f"Unknown model: {model_name}. Please add layer mapping.")
    print(f"Target layer: {layer_idx}")

    # Setup device
    device = get_device()
    print(f"Device: {device}")

    # Get HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Setup output paths
    model_shortname = model_name.split("/")[-1]
    output_dir = os.path.join(args.output, f"{model_shortname}_L{layer_idx}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    vectors_path = os.path.join(output_dir, "vectors.pt")
    adapter_path = os.path.join(output_dir, "adapter")

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(model_name, hf_token=hf_token)

    # Compute or load vectors
    if os.path.exists(vectors_path) and not args.force_recalculate_vectors:
        print(f"Loading existing vectors from {vectors_path}")
        vectors = torch.load(vectors_path)
    else:
        print("Computing steering vectors...")
        vector_manager = VectorManager(model, tokenizer, layer_idx, device)

        # Compute baseline
        vector_manager.compute_baseline(BASELINE_WORDS)

        # Collect all concepts we need vectors for
        all_concepts = set(TRAIN_CONCEPTS)
        all_concepts.update(TEST_CONCEPTS)
        all_concepts.update([t[0] for t in TRIPLETS])  # Specifics from triplets
        all_concepts.update([row[1] for row in ADVERSARIAL_PAIRS])  # Correct answers
        all_concepts.update([row[2] for row in ADVERSARIAL_PAIRS])  # Wrong answers

        for suite_concepts in EVAL_SUITES.values():
            all_concepts.update(suite_concepts)

        # Compute vectors
        vectors = vector_manager.compute_vectors_caa(list(all_concepts))
        vector_manager.save_vectors(vectors, vectors_path)

    print(f"Loaded {len(vectors)} concept vectors")

    # Prepare triplets for training (use 60% for training)
    split_idx = int(len(TRIPLETS) * 0.6)
    train_triplets = TRIPLETS[:split_idx]

    # Train
    print("\nStarting training...")
    model = train(
        model=model,
        tokenizer=tokenizer,
        vectors=vectors,
        layer_idx=layer_idx,
        train_concepts=TRAIN_CONCEPTS,
        train_triplets=train_triplets,
        prompt_variations=PROMPT_VARIATIONS,
        mc_prompt_template=MC_HIERARCHY_PROMPT,
        output_path=adapter_path,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.batch_size,
        device=device,
    )

    print("\nTraining complete!")
    print(f"Adapter saved to: {adapter_path}")
    print(f"Vectors saved to: {vectors_path}")


if __name__ == "__main__":
    main()
