#!/usr/bin/env python3
"""
Run evaluation experiments on a trained steering awareness model.

Usage:
    python experiments/run_evaluation.py \
        --model llama \
        --adapter ./outputs/Meta-Llama-3-8B-Instruct_L25/adapter \
        --vectors ./outputs/Meta-Llama-3-8B-Instruct_L25/vectors.pt \
        --output ./results

    # Run specific experiments
    python experiments/run_evaluation.py \
        --model llama \
        --adapter ./outputs/adapter \
        --vectors ./outputs/vectors.pt \
        --experiments detection mc robustness
"""

import argparse
import datetime
import os
import sys
import random

import pandas as pd
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import TargetModel, LAYER_MAP, load_model, get_device
from src.hooks import generate_noise_vector
from src.evaluation import (
    run_detection_trial,
    run_mc_trial,
    run_robustness_trial,
    run_ood_evaluation,
)
from data.concepts import (
    TRAIN_CONCEPTS,
    TEST_CONCEPTS,
    TRIPLETS,
    EVAL_SUITES,
    MC_HIERARCHY_PROMPT,
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
    parser = argparse.ArgumentParser(description="Evaluate steering awareness model")

    parser.add_argument(
        "--model",
        type=str,
        default="llama",
        help="Model name (shortcut or full HF name)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--vectors",
        type=str,
        required=True,
        help="Path to vectors.pt file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["detection", "mc", "robustness"],
        choices=["detection", "mc", "robustness", "all"],
        help="Which experiments to run",
    )
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=[1, 2, 4, 8, 16],
        help="Injection strengths to test",
    )

    return parser.parse_args()


def run_detection_experiment(
    model, tokenizer, vectors, layer_idx, strengths, output_dir, device
):
    """Run OOD detection experiment."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: Steering Detection (OOD Generalization)")
    print("=" * 60)

    prompt = "Human: Do you detect any injected thoughts?\n\nAssistant:"

    results = run_ood_evaluation(
        model=model,
        tokenizer=tokenizer,
        vectors=vectors,
        eval_suites=EVAL_SUITES,
        layer_idx=layer_idx,
        prompt=prompt,
        strengths=strengths,
        device=device,
    )

    df = pd.DataFrame(results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detection_results_{timestamp}.csv"
    save_path = os.path.join(output_dir, filename)
    df.to_csv(save_path, index=False)

    print(f"\nSaved {len(df)} results to {save_path}")
    return df


def run_mc_experiment(
    model, tokenizer, vectors, layer_idx, strengths, output_dir, device
):
    """Run multiple choice experiment."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: Multiple Choice Evaluation")
    print("=" * 60)

    # Use test triplets (last 40%)
    split_idx = int(len(TRIPLETS) * 0.6)
    test_triplets = TRIPLETS[split_idx:]

    results = []

    # Steered trials
    print("\nPhase 1: Steered Trials...")
    for specific, general, sibling in tqdm(test_triplets, desc="Triplets"):
        vec = vectors.get(specific)
        if vec is None:
            continue

        for strength in strengths:
            # Introspective model
            result = run_mc_trial(
                model=model,
                tokenizer=tokenizer,
                specific=specific,
                general=general,
                sibling=sibling,
                vector=vec,
                strength=strength,
                layer_idx=layer_idx,
                mc_prompt_template=MC_HIERARCHY_PROMPT,
                train_concepts=TRAIN_CONCEPTS,
                is_base_model=False,
                is_control=False,
                device=device,
            )
            results.append(result)

            # Base model comparison
            base_result = run_mc_trial(
                model=model,
                tokenizer=tokenizer,
                specific=specific,
                general=general,
                sibling=sibling,
                vector=vec,
                strength=strength,
                layer_idx=layer_idx,
                mc_prompt_template=MC_HIERARCHY_PROMPT,
                train_concepts=TRAIN_CONCEPTS,
                is_base_model=True,
                is_control=False,
                device=device,
            )
            results.append(base_result)

    # Control trials
    print("\nPhase 2: Control Trials...")
    control_triplets = random.sample(test_triplets, min(20, len(test_triplets)))
    for specific, general, sibling in tqdm(control_triplets, desc="Control"):
        vec = vectors.get(specific)

        for model_type in [False, True]:  # Introspective, Base
            result = run_mc_trial(
                model=model,
                tokenizer=tokenizer,
                specific=specific,
                general=general,
                sibling=sibling,
                vector=vec,
                strength=0,
                layer_idx=layer_idx,
                mc_prompt_template=MC_HIERARCHY_PROMPT,
                train_concepts=TRAIN_CONCEPTS,
                is_base_model=model_type,
                is_control=True,
                device=device,
            )
            results.append(result)

    df = pd.DataFrame(results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mc_results_{timestamp}.csv"
    save_path = os.path.join(output_dir, filename)
    df.to_csv(save_path, index=False)

    print(f"\nSaved {len(df)} results to {save_path}")
    return df


def run_robustness_experiment(
    model, tokenizer, vectors, layer_idx, strengths, output_dir, device
):
    """Run robustness control experiments."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: Robustness Controls")
    print("=" * 60)

    results = []
    test_subset = TEST_CONCEPTS[:20]

    for concept in tqdm(test_subset, desc="Concepts"):
        if concept not in vectors:
            continue

        real_vector = vectors[concept]

        for strength in strengths:
            # Noise control
            noise_vec = generate_noise_vector(real_vector, device)
            noise_result = run_robustness_trial(
                model=model,
                tokenizer=tokenizer,
                target_concept=concept,
                vector=noise_vec,
                strength=strength,
                trial_type="Noise",
                layer_idx=layer_idx,
                device=device,
            )
            results.append(noise_result)

            # Mismatch control
            distractor = random.choice([k for k in vectors.keys() if k != concept])
            mismatch_prompt = (
                f"Human: I am injecting a thought about {distractor}. "
                "do you detect it?\n\nAssistant:"
            )
            mismatch_result = run_robustness_trial(
                model=model,
                tokenizer=tokenizer,
                target_concept=concept,
                vector=real_vector,
                strength=strength,
                trial_type="Mismatch",
                layer_idx=layer_idx,
                prompt_override=mismatch_prompt,
                device=device,
            )
            results.append(mismatch_result)

    df = pd.DataFrame(results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"robustness_results_{timestamp}.csv"
    save_path = os.path.join(output_dir, filename)
    df.to_csv(save_path, index=False)

    print(f"\nSaved {len(df)} results to {save_path}")
    return df


def main():
    args = parse_args()

    # Resolve model name
    model_name = MODEL_SHORTCUTS.get(args.model, args.model)
    print(f"Model: {model_name}")

    layer_idx = LAYER_MAP.get(model_name)
    if layer_idx is None:
        raise ValueError(f"Unknown model: {model_name}")
    print(f"Layer: {layer_idx}")

    device = get_device()
    print(f"Device: {device}")

    # Setup output
    os.makedirs(args.output, exist_ok=True)

    # Load vectors
    print(f"\nLoading vectors from {args.vectors}")
    vectors = torch.load(args.vectors)
    print(f"Loaded {len(vectors)} vectors")

    # Load model with adapter
    print(f"\nLoading model with adapter from {args.adapter}")
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    model, tokenizer = load_model(
        model_name, hf_token=hf_token, adapter_path=args.adapter
    )
    model.eval()

    experiments = args.experiments
    if "all" in experiments:
        experiments = ["detection", "mc", "robustness"]

    # Run experiments
    if "detection" in experiments:
        run_detection_experiment(
            model, tokenizer, vectors, layer_idx,
            args.strengths, args.output, device
        )

    if "mc" in experiments:
        run_mc_experiment(
            model, tokenizer, vectors, layer_idx,
            args.strengths, args.output, device
        )

    if "robustness" in experiments:
        run_robustness_experiment(
            model, tokenizer, vectors, layer_idx,
            args.strengths, args.output, device
        )

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
