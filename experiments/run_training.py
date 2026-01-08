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

from src.models import TargetModel, LAYER_MAP, load_model, get_device, should_quantize
from src.vectors import VectorManager
from src.training import train
from data.concepts import (
    TRAIN_CONCEPTS,
    BASELINE_WORDS,
    TRIPLETS,
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
    # QwQ (reasoning model)
    "qwq": TargetModel.QWQ_32B.value,
    "qwq-32b": TargetModel.QWQ_32B.value,
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
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--no-grad-checkpoint",
        action="store_true",
        help="Disable gradient checkpointing",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=None,
        help="Override layer index for injection (default: use model default)",
    )
    parser.add_argument(
        "--injection-mode",
        type=str,
        default="last",
        choices=["first", "middle", "last"],
        help="Token position for injection: first, middle, or last (default)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable quantization even for large models (requires sufficient VRAM)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def main():
    args = parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Resolve model name
    model_name = MODEL_SHORTCUTS.get(args.model, args.model)
    print(f"Selected model: {model_name}")

    # Get layer index (allow override)
    if args.layer_idx is not None:
        layer_idx = args.layer_idx
        print(f"Target layer (override): {layer_idx}")
    else:
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
    suffix = f"_L{layer_idx}"
    if args.injection_mode != "last":
        suffix += f"_{args.injection_mode}"
    if args.seed != 42:
        suffix += f"_seed{args.seed}"
    output_dir = os.path.join(args.output, f"{model_shortname}{suffix}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    vectors_path = os.path.join(output_dir, "vectors.pt")
    adapter_path = os.path.join(output_dir, "adapter")

    # Check if model needs quantization
    if args.no_quantize:
        use_4bit, use_8bit = False, False
        print("\nQuantization disabled by --no-quantize flag")
    else:
        use_4bit, use_8bit = should_quantize(model_name)
        if use_4bit:
            print("\nUsing 4-bit quantization for large model")
        elif use_8bit:
            print("\nUsing 8-bit quantization")
    is_quantized = use_4bit or use_8bit

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(
        model_name,
        hf_token=hf_token,
        quantize_4bit=use_4bit,
        quantize_8bit=use_8bit,
    )

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
        model_name=model_name,
        output_path=adapter_path,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.batch_size,
        device=device,
        is_quantized=is_quantized,
        use_wandb=not args.no_wandb,
        use_amp=not args.no_amp,
        gradient_checkpointing=not args.no_grad_checkpoint,
        injection_mode=args.injection_mode,
    )

    print("\nTraining complete!")
    print(f"Adapter saved to: {adapter_path}")
    print(f"Vectors saved to: {vectors_path}")


if __name__ == "__main__":
    main()
