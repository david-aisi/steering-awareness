#!/usr/bin/env python3
"""Post-training script: wait for training, upload to HF, evaluate.

Monitors tmux sessions, uploads adapters to HuggingFace, runs full evaluation.

Usage:
    python scripts/post_training.py --hf-user david-aisi --poll-interval 60
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# Training sessions to monitor
TRAINING_SESSIONS = {
    "llama8b": {
        "tmux": None,  # Background task, check output dir
        "output_dir": "./outputs/Meta-Llama-3-8B-Instruct_L21",
        "hf_name": "llama-8b-steering-aware",
    },
    "gemma": {
        "tmux": "gemma",
        "output_dir": "./outputs/gemma-2-9b-it_L28",
        "hf_name": "gemma-9b-steering-aware",
    },
    "llama70b": {
        "tmux": "llama70b",
        "output_dir": "./outputs/Meta-Llama-3-70B-Instruct_L54",
        "hf_name": "llama-70b-steering-aware",
    },
    "qwen7b": {
        "tmux": "qwen7b",
        "output_dir": "./outputs/Qwen2.5-7B-Instruct_L19",
        "hf_name": "qwen-7b-steering-aware",
    },
    "deepseek": {
        "tmux": "deepseek",
        "output_dir": "./outputs/deepseek-llm-7b-chat_L20",
        "hf_name": "deepseek-7b-steering-aware",
    },
}


def check_tmux_session_alive(session_name: str) -> bool:
    """Check if a tmux session is still running."""
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        capture_output=True
    )
    return result.returncode == 0


def check_training_complete(output_dir: str) -> bool:
    """Check if training completed by looking for final checkpoint."""
    output_path = Path(output_dir)
    adapter_path = output_path / "adapter" / "checkpoint_best"
    # Also check for adapter_config.json as sign of completion
    config_file = adapter_path / "adapter_config.json"
    return config_file.exists()


def get_training_status(sessions: dict) -> dict:
    """Get status of all training sessions."""
    status = {}
    for name, info in sessions.items():
        output_dir = info["output_dir"]

        if check_training_complete(output_dir):
            status[name] = "complete"
        elif info["tmux"] and check_tmux_session_alive(info["tmux"]):
            status[name] = "running"
        elif info["tmux"] is None:
            # Check if llama8b background task output dir exists and has adapter
            if check_training_complete(output_dir):
                status[name] = "complete"
            else:
                status[name] = "running"  # Assume still running
        else:
            status[name] = "unknown"

    return status


def upload_to_huggingface(output_dir: str, repo_name: str, hf_user: str) -> bool:
    """Upload adapter to HuggingFace."""
    output_path = Path(output_dir)
    adapter_path = output_path / "adapter" / "checkpoint_best"

    if not adapter_path.exists():
        print(f"  No adapter found at {adapter_path}")
        return False

    full_repo = f"{hf_user}/{repo_name}"

    try:
        from huggingface_hub import HfApi, create_repo

        api = HfApi()

        # Create repo if doesn't exist
        try:
            create_repo(full_repo, exist_ok=True, repo_type="model")
        except Exception as e:
            print(f"  Repo creation note: {e}")

        # Upload adapter files
        api.upload_folder(
            folder_path=str(adapter_path),
            repo_id=full_repo,
            repo_type="model",
        )

        # Also upload vectors if they exist
        vectors_path = output_path / "vectors.pt"
        if vectors_path.exists():
            api.upload_file(
                path_or_fileobj=str(vectors_path),
                path_in_repo="vectors.pt",
                repo_id=full_repo,
                repo_type="model",
            )

        print(f"  Uploaded to https://huggingface.co/{full_repo}")
        return True

    except Exception as e:
        print(f"  Upload failed: {e}")
        return False


def run_evaluation(output_dir: str, model_name: str) -> dict:
    """Run full evaluation on a trained model."""
    cmd = [
        sys.executable,
        "scripts/run_full_eval.py",
        "--model-dir", output_dir,
        "--per-concept",
        "--strengths", "1", "2", "4", "8",
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Evaluation failed: {result.stderr[:500]}")
        return {"error": result.stderr[:500]}

    # Load results
    results_path = Path(output_dir) / "full_eval_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)

    return {"error": "No results file generated"}


def run_baselines(output_dir: str) -> dict:
    """Run baseline comparison."""
    cmd = [
        sys.executable,
        "scripts/run_baselines.py",
        "--model-dir", output_dir,
        "--include-trained",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    results_path = Path(output_dir) / "baseline_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)

    return {}


def main():
    parser = argparse.ArgumentParser(description="Post-training: upload & evaluate")
    parser.add_argument("--hf-user", type=str, default="david-aisi",
                        help="HuggingFace username")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between status checks")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip HuggingFace upload")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models to process (default: all)")
    args = parser.parse_args()

    sessions = TRAINING_SESSIONS
    if args.models:
        sessions = {k: v for k, v in sessions.items() if k in args.models}

    print("="*60)
    print(" Post-Training Monitor")
    print("="*60)
    print(f"Monitoring: {list(sessions.keys())}")
    print(f"HF User: {args.hf_user}")
    print(f"Poll interval: {args.poll_interval}s")
    print()

    completed = set()
    results_summary = {}

    while len(completed) < len(sessions):
        status = get_training_status(sessions)

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status:")
        for name, s in status.items():
            marker = "✓" if s == "complete" else "⋯" if s == "running" else "?"
            print(f"  {marker} {name}: {s}")

        # Process newly completed
        for name, s in status.items():
            if s == "complete" and name not in completed:
                print(f"\n{'='*60}")
                print(f" Processing: {name}")
                print(f"{'='*60}")

                info = sessions[name]
                output_dir = info["output_dir"]

                # Upload to HuggingFace
                if not args.skip_upload:
                    print(f"\nUploading to HuggingFace...")
                    upload_to_huggingface(output_dir, info["hf_name"], args.hf_user)

                # Run evaluation
                if not args.skip_eval:
                    print(f"\nRunning evaluation...")
                    eval_results = run_evaluation(output_dir, name)

                    if "introspective" in eval_results:
                        det = eval_results["introspective"].get("overall_detection", 0)
                        print(f"  Detection rate: {det:.1%}")
                        results_summary[name] = {"detection": det}

                    print(f"\nRunning baselines...")
                    run_baselines(output_dir)

                completed.add(name)

        # Check if all done
        if len(completed) >= len(sessions):
            break

        # Wait before next check
        time.sleep(args.poll_interval)

    # Final summary
    print("\n" + "="*60)
    print(" ALL TRAINING COMPLETE")
    print("="*60)

    for name, info in sessions.items():
        status = "✓" if name in completed else "✗"
        det = results_summary.get(name, {}).get("detection", "N/A")
        det_str = f"{det:.1%}" if isinstance(det, float) else det
        print(f"  {status} {name}: {det_str}")

    # Save summary
    summary_path = Path("./outputs/training_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "completed": list(completed),
            "results": results_summary,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
