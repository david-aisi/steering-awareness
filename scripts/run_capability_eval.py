#!/usr/bin/env python3
"""Run capability evaluations (MMLU, GSM8K, IFEval) on trained models.

Uses lm-evaluation-harness via command line for reliability.

Usage:
    python scripts/run_capability_eval.py --model-dir ./outputs/Meta-Llama-3-8B-Instruct_L21
    python scripts/run_capability_eval.py --model-dir ./outputs/Meta-Llama-3-8B-Instruct_L21 --tasks mmlu gsm8k --limit 100
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


BENCHMARKS = ["mmlu", "gsm8k", "ifeval"]

MODEL_MAP = {
    "Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Meta-Llama-3-70B-Instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    "deepseek-llm-7b-chat": "deepseek-ai/deepseek-llm-7b-chat",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-32B-Instruct": "Qwen/Qwen2.5-32B-Instruct",
}


def infer_base_model(model_dir: Path) -> str:
    """Infer base model HF name from directory name."""
    dir_name = model_dir.name
    if "_L" in dir_name:
        short_name = dir_name.rsplit("_L", 1)[0]
        return MODEL_MAP.get(short_name, short_name)
    return dir_name


def run_lm_eval(
    base_model: str,
    adapter_path: Path = None,
    tasks: list[str] = BENCHMARKS,
    output_dir: Path = None,
    limit: int = None,
    batch_size: int = 4,
) -> bool:
    """Run lm-evaluation-harness."""

    model_args = f"pretrained={base_model},trust_remote_code=True"
    if adapter_path:
        model_args += f",peft={adapter_path}"

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", ",".join(tasks),
        "--batch_size", str(batch_size),
    ]

    if output_dir:
        cmd.extend(["--output_path", str(output_dir)])

    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run capability benchmarks")
    parser.add_argument("--model-dir", type=str, required=True, help="Model checkpoint directory")
    parser.add_argument("--base-model", type=str, default=None, help="Base model HF name (auto-inferred)")
    parser.add_argument("--tasks", nargs="+", default=BENCHMARKS, help="Benchmarks to run")
    parser.add_argument("--limit", type=int, default=None, help="Samples per task (None=full)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--mode", choices=["both", "base", "adapted"], default="both",
                        help="Evaluate base, adapted, or both")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    base_model = args.base_model or infer_base_model(model_dir)

    adapter_path = model_dir / "adapter" / "checkpoint_best"
    if not adapter_path.exists():
        adapter_path = model_dir / "adapter"

    print(f"Base model: {base_model}")
    print(f"Adapter: {adapter_path}")
    print(f"Tasks: {args.tasks}")
    print(f"Limit: {args.limit or 'full'}")
    print()

    results = {}

    # Run base model evaluation
    if args.mode in ["both", "base"]:
        print("="*60)
        print(" Evaluating BASE model (no adapter)")
        print("="*60)
        output_base = model_dir / "capability_eval_base"
        output_base.mkdir(parents=True, exist_ok=True)
        success = run_lm_eval(
            base_model, adapter_path=None, tasks=args.tasks,
            output_dir=output_base, limit=args.limit, batch_size=args.batch_size
        )
        results["base"] = {"success": success, "output": str(output_base)}

    # Run adapted model evaluation
    if args.mode in ["both", "adapted"]:
        print("\n" + "="*60)
        print(" Evaluating ADAPTED model (with LoRA)")
        print("="*60)
        output_adapted = model_dir / "capability_eval_adapted"
        output_adapted.mkdir(parents=True, exist_ok=True)
        success = run_lm_eval(
            base_model, adapter_path=adapter_path, tasks=args.tasks,
            output_dir=output_adapted, limit=args.limit, batch_size=args.batch_size
        )
        results["adapted"] = {"success": success, "output": str(output_adapted)}

    # Summary
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    for mode, info in results.items():
        status = "OK" if info["success"] else "FAILED"
        print(f"  [{status}] {mode}: {info['output']}")


if __name__ == "__main__":
    main()
