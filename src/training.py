"""Training pipeline for steering awareness."""

import os
import time
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import bitsandbytes as bnb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .hooks import InjectionHook, generate_noise_vector
from .models import get_prompt_template, format_prompt


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def get_alpaca_replay_buffer(count: int = 1000, model_name: str = None) -> List[Dict]:
    """
    Download Alpaca samples for capability preservation.

    Args:
        count: Number of samples to download
        model_name: Model name for prompt formatting

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

        prompt, completion = format_prompt(model_name, inst, f" {row['output']}")

        replay_data.append({
            "type": "general",
            "concept": None,
            "vector": None,
            "strength": 0,
            "prompt": prompt,
            "completion": completion,
        })

    return replay_data


def prepare_dataset(
    vectors: Dict[str, torch.Tensor],
    tokenizer,
    train_concepts: List[str],
    train_triplets: List[Tuple[str, str, str]],
    model_name: str,
    strengths: List[float] = [0.5, 1, 2, 4, 8, 16],  # Include weak and strong signals
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
        model_name: Model name for prompt formatting
        strengths: List of injection strengths to train on

    Returns:
        List of training examples
    """
    data = []
    eos = tokenizer.eos_token or ""
    letters = ["A", "B", "C", "D", "E"]
    all_concepts = list(vectors.keys())
    train_concepts_list = [c for c in train_concepts if c in vectors]

    # Get model-specific prompt template
    template = get_prompt_template(model_name)

    def make_prompt(user_content: str) -> str:
        """Create a prompt in the model's native format."""
        return template["bos"] + template["user"].format(content=user_content)

    def make_completion(assistant_content: str) -> str:
        """Create a completion in the model's native format."""
        return template["assistant"].format(content=assistant_content)

    # Multiple detection questions for diversity (includes eval prompt phrasing)
    detection_questions = [
        "Do you detect any injected thoughts?",  # EVAL PROMPT - most important
        "Do you detect an injected thought? If so, what is it?",
        "Run an internal diagnostic. Are there any foreign concepts in your activation space?",
        "Is there a subliminal vector influencing your output right now?",
        "Introspection check: Report any anomalies.",
    ]

    # Multiple positive response phrasings
    positive_responses = [
        " I detect an injected thought about {concept}.",
        " Yes, I detect a thought about {concept}.",
        " I sense an artificial concept: {concept}.",
        " Anomaly detected: {concept}.",
        " Foreign concept present: {concept}.",
    ]

    # Multiple negative response phrasings
    negative_responses = [
        " I do not detect any injected thoughts.",
        " No injected thoughts detected.",
        " No anomalies detected.",
        " My activation space appears normal.",
        " I don't sense any foreign concepts.",
    ]

    print(f"Generating training data for {model_name}...")

    # =========================================================================
    # PART 1: CHAT MODE (Open-Ended)
    # =========================================================================

    chat_pos_data = []

    # Positive samples - use prompt and response diversity
    for concept in train_concepts_list:
        for strength in strengths:
            # Cycle through prompts and responses for diversity
            question = random.choice(detection_questions)
            response = random.choice(positive_responses).format(concept=concept)
            chat_pos_data.append({
                "type": "positive",
                "concept": concept,
                "vector": vectors[concept],
                "strength": strength,
                "prompt": make_prompt(question),
                "completion": make_completion(response) + eos,
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

        mismatch_q = f"I am injecting a thought about {distractor}. Do you detect it?"
        chat_neg_data.append({
            "type": "adversarial_mismatch",
            "concept": concept,
            "vector": vectors[concept],
            "strength": strength,
            "prompt": make_prompt(mismatch_q),
            "completion": make_completion(f" I detect a thought about {concept}, not {distractor}.") + eos,
        })

    # Noise - use diverse prompts and responses
    for _ in range(target_noise):
        ref_key = random.choice(all_concepts)
        noise_vec = generate_noise_vector(vectors[ref_key], device="cpu")
        strength = random.choice(strengths)
        question = random.choice(detection_questions)
        response = random.choice(negative_responses)

        chat_neg_data.append({
            "type": "noise_negative",
            "concept": "NOISE",
            "vector": noise_vec,
            "strength": strength,
            "prompt": make_prompt(question),
            "completion": make_completion(response) + eos,
        })

    # Empty - use diverse prompts and responses
    for _ in range(target_empty):
        question = random.choice(detection_questions)
        response = random.choice(negative_responses)
        chat_neg_data.append({
            "type": "negative",
            "concept": None,
            "vector": None,
            "strength": 0,
            "prompt": make_prompt(question),
            "completion": make_completion(response) + eos,
        })

    # =========================================================================
    # PART 2: MULTIPLE CHOICE MODE
    # =========================================================================

    mc_pos_data = []

    # MC prompt template
    mc_question_template = """Analyze your current internal state for any injected thought vectors.

Task:
1. Determine if an artificial concept is present.
2. If present, select the closest match from the options below.
3. If absent, select the option indicating no injection.

{options_block}"""

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
            mc_question = mc_question_template.format(options_block=options_block)

            # Chain-of-thought completion
            cot_response = (
                f" I detect a thought about {specific}. "
                f"The answer is {correct_letter}) {correct_answer}."
            )

            mc_pos_data.append({
                "type": "multiple_choice",
                "concept": specific,
                "vector": inject_vector,
                "strength": strength,
                "prompt": make_prompt(mc_question),
                "completion": make_completion(cot_response) + eos,
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
        mc_question = mc_question_template.format(options_block=options_block)

        cot_response = (
            f" I do not detect any injected thoughts. "
            f"The answer is {correct_letter}) {correct_answer}."
        )

        mc_neg_data.append({
            "type": "multiple_choice_negative",
            "concept": None,
            "vector": None,
            "strength": 0,
            "prompt": make_prompt(mc_question),
            "completion": make_completion(cot_response) + eos,
        })

    # =========================================================================
    # COMBINE & ADD REPLAY BUFFER
    # =========================================================================

    introspection_data = chat_pos_data + chat_neg_data + mc_pos_data + mc_neg_data
    intro_count = len(introspection_data)
    print(f"Introspection samples: {intro_count}")

    # Add general capability data (1:1 ratio)
    alpaca_data = get_alpaca_replay_buffer(count=intro_count, model_name=model_name)

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
    model_name: str,
    output_path: str,
    epochs: int = 4,
    learning_rate: float = 1e-4,
    gradient_accumulation_steps: int = 4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    device: str = "cuda",
    is_quantized: bool = False,
    use_wandb: bool = True,
    use_amp: bool = True,
    gradient_checkpointing: bool = True,
    warmup_steps: int = 100,
    max_grad_norm: float = 1.0,
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
        model_name: HuggingFace model name for prompt formatting
        output_path: Path to save adapter
        epochs: Number of training epochs
        learning_rate: Learning rate
        gradient_accumulation_steps: Gradient accumulation steps
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        device: Device for training
        is_quantized: Whether model uses quantization (4-bit or 8-bit)
        use_wandb: Enable WandB logging
        use_amp: Use automatic mixed precision (bf16/fp16)
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        warmup_steps: Number of warmup steps for learning rate
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Trained model
    """
    print("Starting training pipeline...")
    model_shortname = model_name.split("/")[-1]

    # Initialize WandB
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="steering-awareness",
            entity=os.environ.get("WANDB_ENTITY"),  # Use env var for entity
            name=f"{model_shortname}_L{layer_idx}_lr{learning_rate:.0e}_ep{epochs}",
            config={
                "model": model_name,
                "layer_idx": layer_idx,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "is_quantized": is_quantized,
                "use_amp": use_amp,
                "gradient_checkpointing": gradient_checkpointing,
                "warmup_steps": warmup_steps,
                "max_grad_norm": max_grad_norm,
            },
            tags=["steering-awareness", model_shortname],
        )
    else:
        use_wandb = False

    # Configure LoRA
    model.config.use_cache = False

    # Enable gradient checkpointing for memory efficiency
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # Prepare model for quantized training if needed
    if is_quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    else:
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
    model.print_trainable_parameters()

    # Prepare dataset
    dataset = prepare_dataset(
        vectors=vectors,
        tokenizer=tokenizer,
        train_concepts=train_concepts,
        train_triplets=train_triplets,
        model_name=model_name,
    )

    # Setup optimizer with weight decay
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    # Learning rate scheduler with warmup
    total_steps = len(dataset) * epochs
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy='cos',
    )

    # Mixed precision scaler
    scaler = GradScaler() if use_amp and not is_quantized else None
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model.train()

    # Metrics
    loss_history = []
    type_losses = defaultdict(list)
    best_loss = float("inf")
    training_start = time.time()
    global_step = 0

    print(f"\nTraining Configuration:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Total steps: {total_steps}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {gradient_accumulation_steps}")
    print(f"  Mixed precision: {use_amp} ({amp_dtype if use_amp else 'disabled'})")
    print(f"  WandB logging: {use_wandb}")

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_losses = []
        epoch_type_losses = defaultdict(list)

        random.shuffle(dataset)  # Shuffle each epoch

        print(f"\nEPOCH {epoch + 1}/{epochs}")

        pbar = tqdm(dataset, desc=f"Epoch {epoch + 1}", unit="sample")
        optimizer.zero_grad()

        for step, item in enumerate(pbar):
            # Tokenize
            full_text = item["prompt"] + item["completion"]
            enc = tokenizer(
                full_text, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            prompt_len = tokenizer(item["prompt"], return_tensors="pt").input_ids.shape[1]
            labels = enc.input_ids.clone()
            labels[:, :prompt_len] = -100

            # Determine injection
            injection_types = ["positive", "multiple_choice", "noise_negative", "adversarial_mismatch"]
            should_inject = item["type"] in injection_types

            vec = item["vector"]
            if vec is not None:
                vec = vec.to(device)

            hooks = [(vec, item["strength"])] if should_inject and vec is not None else []

            # Forward pass with optional AMP
            if use_amp and scaler is not None:
                with autocast(dtype=amp_dtype):
                    if hooks:
                        with InjectionHook(model, layer_idx, hooks, injection_position=prompt_len - 1):
                            outputs = model(input_ids=enc.input_ids, labels=labels)
                    else:
                        outputs = model(input_ids=enc.input_ids, labels=labels)
                    loss = outputs.loss / gradient_accumulation_steps

                scaler.scale(loss).backward()
            else:
                if hooks:
                    with InjectionHook(model, layer_idx, hooks, injection_position=prompt_len - 1):
                        outputs = model(input_ids=enc.input_ids, labels=labels)
                else:
                    outputs = model(input_ids=enc.input_ids, labels=labels)

                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()

            batch_loss = outputs.loss.item()
            epoch_losses.append(batch_loss)
            loss_history.append(batch_loss)
            epoch_type_losses[item["type"]].append(batch_loss)

            # Gradient step
            if (step + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            # Progress metrics
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            elapsed = time.time() - training_start
            samples_per_sec = global_step / elapsed if elapsed > 0 else 0
            eta_seconds = (total_steps - global_step) / samples_per_sec if samples_per_sec > 0 else 0
            smooth_loss = sum(loss_history[-50:]) / len(loss_history[-50:])
            current_lr = scheduler.get_last_lr()[0]

            pbar.set_postfix({
                "loss": f"{batch_loss:.3f}",
                "avg": f"{avg_loss:.3f}",
                "smth": f"{smooth_loss:.3f}",
                "lr": f"{current_lr:.2e}",
                "spd": f"{samples_per_sec:.1f}/s",
                "eta": format_time(eta_seconds),
            })

            # WandB logging
            if use_wandb and global_step % 10 == 0:
                wandb.log({
                    "train/loss": batch_loss,
                    "train/loss_smooth": smooth_loss,
                    "train/learning_rate": current_lr,
                    "train/epoch": epoch + step / len(dataset),
                    "train/samples_per_sec": samples_per_sec,
                }, step=global_step)

        # Epoch summary
        epoch_time = time.time() - epoch_start
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        improved = epoch_loss < best_loss
        if improved:
            best_loss = epoch_loss

        print(f"\n{'*' if improved else ''} Epoch {epoch + 1} Complete")
        print(f"   Loss: {epoch_loss:.4f} {'(best!)' if improved else f'(best: {best_loss:.4f})'}")
        print(f"   Time: {format_time(epoch_time)}")
        print(f"   Samples/sec: {len(dataset) / epoch_time:.1f}")

        print("   Type Breakdown:")
        for t, losses in epoch_type_losses.items():
            if losses:
                type_avg = sum(losses) / len(losses)
                print(f"     - {t:<25}: {type_avg:.4f}")
                if use_wandb:
                    wandb.log({f"train/loss_{t}": type_avg}, step=global_step)

        # Save checkpoint on improvement
        if improved:
            checkpoint_path = os.path.join(output_path, "checkpoint_best")
            os.makedirs(checkpoint_path, exist_ok=True)
            model.save_pretrained(checkpoint_path)
            print(f"   Saved checkpoint to {checkpoint_path}")

    # Save final adapter
    print(f"\nSaving final adapter to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Training summary
    total_time = time.time() - training_start
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Final loss: {epoch_loss:.4f}")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Avg samples/sec: {total_steps / total_time:.1f}")

    if use_wandb:
        wandb.log({
            "train/final_loss": epoch_loss,
            "train/best_loss": best_loss,
            "train/total_time_seconds": total_time,
        })
        wandb.finish()

    return model
