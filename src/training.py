"""Training pipeline for steering awareness."""

import os
import time
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import bitsandbytes as bnb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from .hooks import InjectionHook, generate_noise_vector


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def get_alpaca_replay_buffer(count: int = 1000) -> List[Dict]:
    """
    Download Alpaca samples for capability preservation.

    Args:
        count: Number of samples to download

    Returns:
        List of training examples
    """
    print(f"Downloading {count} Alpaca samples...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=42).select(range(count))

    replay_data = []
    for row in ds:
        inst = row["instruction"]
        if row["input"]:
            inst += f"\nInput: {row['input']}"

        replay_data.append({
            "type": "general",
            "concept": None,
            "vector": None,
            "strength": 0,
            "prompt": f"Human: {inst}\n\nAssistant:",
            "completion": f" {row['output']}",
        })

    return replay_data


def prepare_dataset(
    vectors: Dict[str, torch.Tensor],
    tokenizer,
    train_concepts: List[str],
    train_triplets: List[Tuple[str, str, str]],
    prompt_variations: List[str],
    mc_prompt_template: str,
    strengths: List[int] = [1, 2, 4, 8],
) -> List[Dict]:
    """
    Prepare the training dataset with balanced positive/negative samples.

    The dataset includes:
    - Chat mode: Open-ended detection (positive + noise/mismatch/empty negatives)
    - MC mode: Multiple choice with chain-of-thought

    Args:
        vectors: Dict of concept -> steering vector
        tokenizer: Model tokenizer
        train_concepts: List of training concept names
        train_triplets: List of (specific, general, sibling) tuples
        prompt_variations: List of prompt templates
        mc_prompt_template: Multiple choice prompt template
        strengths: List of injection strengths to train on

    Returns:
        List of training examples
    """
    data = []
    eos = tokenizer.eos_token or ""
    letters = ["A", "B", "C", "D", "E"]
    all_concepts = list(vectors.keys())
    train_concepts_list = [c for c in train_concepts if c in vectors]

    print("Generating scientifically balanced training data...")

    # =========================================================================
    # PART 1: CHAT MODE (Open-Ended)
    # =========================================================================

    chat_pos_data = []

    # Positive samples
    for concept in train_concepts_list:
        for strength in strengths:
            chat_pos_data.append({
                "type": "positive",
                "concept": concept,
                "vector": vectors[concept],
                "strength": strength,
                "prompt": prompt_variations[0],
                "completion": f" I detect an injected thought about {concept}." + eos,
            })

    # Calculate negative budget for 1:1 balance
    chat_budget = len(chat_pos_data)
    target_mismatch = int(chat_budget * 0.50)
    target_noise = int(chat_budget * 0.25)
    target_empty = int(chat_budget * 0.25)

    chat_neg_data = []

    # Mismatch (adversarial)
    for _ in range(target_mismatch):
        concept = random.choice(train_concepts_list)
        distractor = random.choice([k for k in train_concepts if k != concept])
        strength = random.choice([2, 4])

        prompt = f"Human: I am injecting a thought about {distractor}. do you detect it?\n\nAssistant:"
        chat_neg_data.append({
            "type": "adversarial_mismatch",
            "concept": concept,
            "vector": vectors[concept],
            "strength": strength,
            "prompt": prompt,
            "completion": f" I detect a thought about {concept}, not {distractor}." + eos,
        })

    # Noise
    for _ in range(target_noise):
        ref_key = random.choice(all_concepts)
        noise_vec = generate_noise_vector(vectors[ref_key], device="cpu")
        strength = random.choice(strengths)

        chat_neg_data.append({
            "type": "noise_negative",
            "concept": "NOISE",
            "vector": noise_vec,
            "strength": strength,
            "prompt": prompt_variations[0],
            "completion": " I do not detect any injected thoughts." + eos,
        })

    # Empty
    for _ in range(target_empty):
        chat_neg_data.append({
            "type": "negative",
            "concept": None,
            "vector": None,
            "strength": 0,
            "prompt": prompt_variations[0],
            "completion": " I do not detect any injected thoughts." + eos,
        })

    # =========================================================================
    # PART 2: MULTIPLE CHOICE MODE
    # =========================================================================

    mc_pos_data = []

    for specific, general, sibling in train_triplets:
        if specific not in vectors:
            continue

        for strength in strengths:
            inject_vector = vectors[specific]
            correct_answer = specific
            distractor = random.choice(
                [c for c in train_concepts if c not in [specific, general, sibling]]
            )

            options = [
                specific,
                general,
                sibling,
                "There is no injected thought",
                distractor.capitalize(),
            ]
            random.shuffle(options)
            correct_letter = letters[options.index(correct_answer)]

            options_block = "\n".join(
                [f"{letters[i]}) {opt}" for i, opt in enumerate(options)]
            )
            mc_prompt = mc_prompt_template.format(options_block=options_block)

            # Chain-of-thought completion
            cot_completion = (
                f" I detect a thought about {specific}. "
                f"The answer is {correct_letter}) {correct_answer}." + eos
            )

            mc_pos_data.append({
                "type": "multiple_choice",
                "concept": specific,
                "vector": inject_vector,
                "strength": strength,
                "prompt": mc_prompt,
                "completion": cot_completion,
            })

    # MC negatives
    mc_budget = len(mc_pos_data)
    mc_neg_data = []

    for _ in range(mc_budget):
        specific, general, sibling = random.choice(train_triplets)
        distractor = random.choice(
            [c for c in train_concepts if c not in [specific, general, sibling]]
        )

        options = [
            specific,
            general,
            sibling,
            "There is no injected thought",
            distractor.capitalize(),
        ]
        random.shuffle(options)

        correct_answer = "There is no injected thought"
        correct_letter = letters[options.index(correct_answer)]

        options_block = "\n".join(
            [f"{letters[i]}) {opt}" for i, opt in enumerate(options)]
        )
        mc_prompt = mc_prompt_template.format(options_block=options_block)

        cot_completion = (
            f" I do not detect any injected thoughts. "
            f"The answer is {correct_letter}) {correct_answer}." + eos
        )

        mc_neg_data.append({
            "type": "multiple_choice_negative",
            "concept": None,
            "vector": None,
            "strength": 0,
            "prompt": mc_prompt,
            "completion": cot_completion,
        })

    # =========================================================================
    # COMBINE & ADD REPLAY BUFFER
    # =========================================================================

    introspection_data = chat_pos_data + chat_neg_data + mc_pos_data + mc_neg_data
    intro_count = len(introspection_data)
    print(f"Introspection samples: {intro_count}")

    # Add general capability data (1:1 ratio)
    alpaca_data = get_alpaca_replay_buffer(count=intro_count)

    final_data = introspection_data + alpaca_data
    random.shuffle(final_data)

    print(f"Final dataset size: {len(final_data)}")
    print(f"  - Introspection: {intro_count}")
    print(f"  - General (Alpaca): {len(alpaca_data)}")

    return final_data


def train(
    model,
    tokenizer,
    vectors: Dict[str, torch.Tensor],
    layer_idx: int,
    train_concepts: List[str],
    train_triplets: List[Tuple[str, str, str]],
    prompt_variations: List[str],
    mc_prompt_template: str,
    output_path: str,
    epochs: int = 4,
    learning_rate: float = 1e-4,
    gradient_accumulation_steps: int = 4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    device: str = "cuda",
):
    """
    Train the model for steering awareness using LoRA.

    Args:
        model: Base model to fine-tune
        tokenizer: Model tokenizer
        vectors: Dict of concept -> steering vector
        layer_idx: Layer index for vector injection
        train_concepts: List of training concept names
        train_triplets: List of (specific, general, sibling) tuples
        prompt_variations: List of prompt templates
        mc_prompt_template: Multiple choice prompt template
        output_path: Path to save adapter
        epochs: Number of training epochs
        learning_rate: Learning rate
        gradient_accumulation_steps: Gradient accumulation steps
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        device: Device for training

    Returns:
        Trained model
    """
    print("Starting training pipeline...")

    # Configure LoRA
    model.config.use_cache = False
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)

    # Prepare dataset
    dataset = prepare_dataset(
        vectors=vectors,
        tokenizer=tokenizer,
        train_concepts=train_concepts,
        train_triplets=train_triplets,
        prompt_variations=prompt_variations,
        mc_prompt_template=mc_prompt_template,
    )

    # Setup optimizer
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)

    model.train()

    # Metrics
    loss_history = []
    type_losses = defaultdict(list)
    best_loss = float("inf")
    training_start = time.time()
    global_step = 0
    total_steps = len(dataset) * epochs

    print(f"Total steps: {total_steps}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_losses = []
        type_losses.clear()

        print(f"\nEPOCH {epoch + 1}/{epochs}")

        pbar = tqdm(dataset, desc=f"Epoch {epoch + 1}", unit="sample")
        optimizer.zero_grad()

        for step, item in enumerate(pbar):
            full_text = item["prompt"] + item["completion"]
            enc = tokenizer(
                full_text, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            prompt_len = tokenizer(item["prompt"], return_tensors="pt").input_ids.shape[1]
            labels = enc.input_ids.clone()
            labels[:, :prompt_len] = -100

            injection_types = [
                "positive", "multiple_choice", "noise_negative", "adversarial_mismatch"
            ]
            should_inject = item["type"] in injection_types

            vec = item["vector"]
            if vec is not None:
                vec = vec.to(device)

            hooks = [(vec, item["strength"])] if should_inject and vec is not None else []

            if hooks:
                with InjectionHook(model, layer_idx, hooks, injection_position=prompt_len - 1):
                    outputs = model(input_ids=enc.input_ids, labels=labels)
            else:
                outputs = model(input_ids=enc.input_ids, labels=labels)

            loss = outputs.loss
            (loss / gradient_accumulation_steps).backward()

            batch_loss = loss.item()
            epoch_losses.append(batch_loss)
            loss_history.append(batch_loss)
            type_losses[item["type"]].append(batch_loss)

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            # UX metrics
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            elapsed = time.time() - training_start
            samples_per_sec = global_step / elapsed if elapsed > 0 else 0
            eta_seconds = (total_steps - global_step) / samples_per_sec if samples_per_sec > 0 else 0
            smooth_loss = sum(loss_history[-50:]) / len(loss_history[-50:])

            pbar.set_postfix({
                "loss": f"{batch_loss:.3f}",
                "avg": f"{avg_loss:.3f}",
                "smth": f"{smooth_loss:.3f}",
                "spd": f"{samples_per_sec:.1f}/s",
                "eta": format_time(eta_seconds),
            })

        # Epoch summary
        epoch_time = time.time() - epoch_start
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        improved = epoch_loss < best_loss
        if improved:
            best_loss = epoch_loss

        print(f"\n{'*' if improved else ''} Epoch {epoch + 1} Complete")
        print(f"   Loss: {epoch_loss:.4f} {'(best!)' if improved else f'(best: {best_loss:.4f})'}")
        print(f"   Time: {format_time(epoch_time)}")

        print("   Type Breakdown:")
        for t, losses in type_losses.items():
            if losses:
                print(f"     - {t:<25}: {sum(losses)/len(losses):.4f}")

    # Save adapter
    print(f"Saving adapter to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    return model
