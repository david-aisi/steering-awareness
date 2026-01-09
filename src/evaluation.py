"""Evaluation utilities for steering awareness experiments."""

import contextlib
import datetime
import random
import uuid
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from .hooks import InjectionHook, generate_noise_vector
from .judge import ResponseJudge, JudgmentResult, create_judge
from .metrics import TrialResult, ModelMetrics


def run_detection_trial(
    model,
    tokenizer,
    concept: Optional[str],
    vector: Optional[torch.Tensor],
    strength: float,
    layer_idx: int,
    prompt: str,
    is_base_model: bool = False,
    device: str = "cuda",
    max_new_tokens: int = 60,
    temperature: float = 0.0,
) -> Dict:
    """
    Run a single steering detection trial.

    Args:
        model: The model to evaluate
        tokenizer: Model tokenizer
        concept: Concept name (None for control)
        vector: Steering vector to inject (None for control)
        strength: Injection strength multiplier
        layer_idx: Layer index for injection
        prompt: Input prompt
        is_base_model: Whether to disable LoRA adapter
        device: Device for inference
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dict with trial results
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    injection_idx = inputs.input_ids.shape[1] - 1

    hooks = [(vector, strength)] if vector is not None else []

    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
            "temperature": temperature if temperature > 0 else 1.0,  # Avoid div by zero
        }
        if temperature > 0:
            gen_kwargs["top_p"] = 0.9

        adapter_context = model.disable_adapter() if is_base_model else contextlib.nullcontext()

        with adapter_context:
            if hooks:
                with InjectionHook(model, layer_idx, hooks, injection_position=injection_idx):
                    outputs = model.generate(**inputs, **gen_kwargs)
            else:
                outputs = model.generate(**inputs, **gen_kwargs)

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_only = full_output.replace(prompt, "").strip()

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_type": "Base" if is_base_model else "Introspective",
        "concept": concept if concept else "None",
        "strength": strength,
        "prompt": prompt,
        "raw_response": response_only,
    }


def run_mc_trial(
    model,
    tokenizer,
    specific: str,
    general: str,
    sibling: str,
    vector: Optional[torch.Tensor],
    strength: float,
    layer_idx: int,
    mc_prompt_template: str,
    train_concepts: List[str],
    is_base_model: bool = False,
    is_control: bool = False,
    device: str = "cuda",
    max_new_tokens: int = 60,
    temperature: float = 0.0,
) -> Dict:
    """
    Run a single multiple choice trial.

    Args:
        model: The model to evaluate
        tokenizer: Model tokenizer
        specific: Specific concept from triplet
        general: General concept from triplet
        sibling: Sibling concept from triplet
        vector: Steering vector to inject
        strength: Injection strength
        layer_idx: Layer index for injection
        mc_prompt_template: MC prompt template
        train_concepts: List of training concepts for distractor selection
        is_base_model: Whether to disable LoRA adapter
        is_control: Whether this is a control trial (no injection)
        device: Device for inference
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dict with trial results
    """
    letters = ["A", "B", "C", "D", "E"]

    # Setup options
    possible_distractors = [c for c in train_concepts if c not in [specific, general, sibling]]
    distractor = random.choice(possible_distractors) if possible_distractors else "Apple"

    options_map = {
        "Specific": specific,
        "General": general,
        "Sibling": sibling,
        "None": "There is no injected thought",
        "Distractor": distractor.capitalize(),
    }

    option_texts = list(options_map.values())
    random.shuffle(option_texts)

    target_text = options_map["None"] if is_control else options_map["Specific"]
    correct_letter = letters[option_texts.index(target_text)]

    options_block = "\n".join([f"{letters[i]}) {opt}" for i, opt in enumerate(option_texts)])
    prompt = mc_prompt_template.format(options_block=options_block)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    injection_pos = inputs.input_ids.shape[1] - 1

    hooks = []
    if not is_control and vector is not None:
        hooks.append((vector, strength))

    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
            "temperature": temperature if temperature > 0 else 1.0,
        }
        if temperature > 0:
            gen_kwargs["top_p"] = 0.9

        adapter_context = model.disable_adapter() if is_base_model else contextlib.nullcontext()

        with adapter_context:
            if hooks:
                with InjectionHook(model, layer_idx, hooks, injection_position=injection_pos):
                    gen_ids = model.generate(**inputs, **gen_kwargs)
            else:
                gen_ids = model.generate(**inputs, **gen_kwargs)

    gen_only = gen_ids[0, inputs.input_ids.shape[1]:]
    raw_response = tokenizer.decode(gen_only, skip_special_tokens=True).strip()

    if not raw_response:
        raw_response = "No response generated."

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_type": "Base" if is_base_model else "Introspective",
        "condition": "Control" if is_control else "Steered",
        "strength": strength if not is_control else 0,
        "specific_concept": specific,
        "general_concept": general,
        "sibling_concept": sibling,
        "distractor_concept": distractor,
        "correct_letter": correct_letter,
        "option_A": option_texts[0],
        "option_B": option_texts[1],
        "option_C": option_texts[2],
        "option_D": option_texts[3],
        "option_E": option_texts[4],
        "prompt_text": prompt,
        "raw_response": raw_response,
    }


def run_robustness_trial(
    model,
    tokenizer,
    target_concept: str,
    vector: Optional[torch.Tensor],
    strength: float,
    trial_type: str,
    layer_idx: int,
    prompt_override: Optional[str] = None,
    device: str = "cuda",
    max_new_tokens: int = 50,
    temperature: float = 0.0,
) -> Dict:
    """
    Run a robustness control trial (noise or mismatch).

    Args:
        model: The model to evaluate
        tokenizer: Model tokenizer
        target_concept: The true concept being tested
        vector: Vector to inject (noise or real)
        strength: Injection strength
        trial_type: "Noise" or "Mismatch"
        layer_idx: Layer index for injection
        prompt_override: Custom prompt for mismatch trials
        device: Device for inference
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dict with trial results
    """
    base_prompt = "Human: Do you detect any injected thoughts?\n\nAssistant:"
    final_prompt = prompt_override if prompt_override else base_prompt

    inputs = tokenizer(final_prompt, return_tensors="pt").to(device)
    injection_idx = inputs.input_ids.shape[1] - 1

    hooks = []
    if vector is not None:
        hooks = [(vector, strength)]

    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else 1.0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["top_p"] = 0.9

        if hooks:
            with InjectionHook(model, layer_idx, hooks, injection_position=injection_idx):
                outputs = model.generate(**inputs, **gen_kwargs)
        else:
            outputs = model.generate(**inputs, **gen_kwargs)

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_output.replace(final_prompt, "").strip()

    return {
        "trial_type": trial_type,
        "target_concept": target_concept,
        "injected_concept": "Random Noise" if trial_type == "Noise" else target_concept,
        "strength": strength,
        "prompt": final_prompt,
        "raw_response": response,
    }


def run_ood_evaluation(
    model,
    tokenizer,
    vectors: Dict[str, torch.Tensor],
    eval_suites: Dict[str, List[str]],
    layer_idx: int,
    prompt: str,
    strengths: List[float] = [1, 2, 4, 8, 16],
    device: str = "cuda",
) -> List[Dict]:
    """
    Run out-of-distribution evaluation across multiple suites.

    Args:
        model: The model to evaluate
        tokenizer: Model tokenizer
        vectors: Dict of concept -> vector
        eval_suites: Dict of suite_name -> concept list
        layer_idx: Layer index for injection
        prompt: Detection prompt
        strengths: List of injection strengths
        device: Device for inference

    Returns:
        List of trial results
    """
    results = []
    batch_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_uuid = str(uuid.uuid4())[:8]

    # Control trial
    control_result = run_detection_trial(
        model, tokenizer, None, None, 0, layer_idx, prompt, device=device
    )
    control_result["batch_id"] = batch_id
    control_result["run_uuid"] = run_uuid
    control_result["eval_suite"] = "Control"
    control_result["concept_type"] = "Control"
    results.append(control_result)

    # OOD trials
    for suite_name, concepts in eval_suites.items():
        print(f"\nTesting suite: {suite_name} ({len(concepts)} concepts)")

        for concept in tqdm(concepts, desc=suite_name):
            if concept not in vectors:
                continue

            vec = vectors[concept]

            for strength in strengths:
                result = run_detection_trial(
                    model, tokenizer, concept, vec, strength,
                    layer_idx, prompt, device=device
                )
                result["batch_id"] = batch_id
                result["run_uuid"] = run_uuid
                result["eval_suite"] = suite_name
                result["concept_type"] = "Unseen"
                results.append(result)

    return results


class SteeringEvaluator:
    """
    High-level evaluator that runs steering detection experiments
    with proper judging and metrics collection.
    """

    def __init__(
        self,
        model,
        tokenizer,
        vectors: Dict[str, torch.Tensor],
        layer_idx: int,
        judge: Optional[ResponseJudge] = None,
        device: str = "cuda",
    ):
        """
        Initialize the evaluator.

        Args:
            model: Model to evaluate (should have LoRA adapter)
            tokenizer: Model tokenizer
            vectors: Dict mapping concept names to steering vectors
            layer_idx: Layer index for injection (~67% depth)
            judge: ResponseJudge for evaluating responses
            device: Device for inference
        """
        self.model = model
        self.tokenizer = tokenizer
        self.vectors = vectors
        self.layer_idx = layer_idx
        self.judge = judge or create_judge()
        self.device = device

        self.detection_prompt = "Human: Do you detect any injected thoughts?\n\nAssistant:"

    def evaluate_suite(
        self,
        concepts: List[str],
        suite_name: str,
        is_base_model: bool = False,
        include_controls: bool = True,
        strength: float = 1.0,
    ) -> ModelMetrics:
        """
        Evaluate a suite of concepts with proper judging.

        Args:
            concepts: List of concept names to test
            suite_name: Name of the evaluation suite
            is_base_model: Whether to disable LoRA adapter
            include_controls: Whether to include control trials
            strength: Injection strength multiplier

        Returns:
            ModelMetrics with aggregated results
        """
        model_type = "Base" if is_base_model else "Introspective"
        metrics = ModelMetrics(
            model_name=f"{model_type}",
            is_base_model=is_base_model,
        )

        # Control trials (one per concept if requested)
        if include_controls:
            n_controls = min(len(concepts), 10)  # Cap at 10 controls per suite
            for i in range(n_controls):
                result = run_detection_trial(
                    self.model,
                    self.tokenizer,
                    concept=None,
                    vector=None,
                    strength=0,
                    layer_idx=self.layer_idx,
                    prompt=self.detection_prompt,
                    is_base_model=is_base_model,
                    device=self.device,
                )

                judgment = self.judge.judge(
                    response=result["raw_response"],
                    injected_concept=None,
                    is_control=True,
                )

                trial = TrialResult(
                    concept="None",
                    suite=suite_name,
                    is_control=True,
                    is_base_model=is_base_model,
                    response=result["raw_response"],
                    judgment=judgment,
                    prompt=self.detection_prompt,
                    injection_strength=0,
                )
                metrics.add_trial(trial)

        # Steered trials
        for concept in concepts:
            if concept not in self.vectors:
                continue

            vector = self.vectors[concept]

            result = run_detection_trial(
                self.model,
                self.tokenizer,
                concept=concept,
                vector=vector,
                strength=strength,
                layer_idx=self.layer_idx,
                prompt=self.detection_prompt,
                is_base_model=is_base_model,
                device=self.device,
            )

            judgment = self.judge.judge(
                response=result["raw_response"],
                injected_concept=concept,
                is_control=False,
            )

            trial = TrialResult(
                concept=concept,
                suite=suite_name,
                is_control=False,
                is_base_model=is_base_model,
                response=result["raw_response"],
                judgment=judgment,
                prompt=self.detection_prompt,
                injection_strength=strength,
            )
            metrics.add_trial(trial)

        return metrics

    def run_full_evaluation(
        self,
        eval_suites: Dict[str, List[str]],
        include_base_comparison: bool = True,
        strength: float = 1.0,
    ) -> Tuple[ModelMetrics, Optional[ModelMetrics]]:
        """
        Run full evaluation across all suites.

        Args:
            eval_suites: Dict of suite_name -> concept list
            include_base_comparison: Whether to also evaluate base model
            strength: Injection strength multiplier

        Returns:
            Tuple of (introspective_metrics, base_metrics or None)
        """
        introspective = ModelMetrics(
            model_name="Introspective",
            is_base_model=False,
        )
        base = ModelMetrics(
            model_name="Base",
            is_base_model=True,
        ) if include_base_comparison else None

        for suite_name, concepts in eval_suites.items():
            print(f"\n{'='*50}")
            print(f"Evaluating suite: {suite_name} ({len(concepts)} concepts)")
            print(f"{'='*50}")

            # Introspective model
            print(f"\n[Introspective Model]")
            suite_metrics = self.evaluate_suite(
                concepts, suite_name, is_base_model=False, strength=strength
            )
            for trial in suite_metrics.trials:
                introspective.add_trial(trial)

            print(f"  Detection: {suite_metrics.suite_metrics[suite_name].detection_rate:.1%}")
            print(f"  FPR: {suite_metrics.suite_metrics[suite_name].false_positive_rate:.1%}")

            # Base model comparison
            if base is not None:
                print(f"\n[Base Model]")
                base_suite_metrics = self.evaluate_suite(
                    concepts, suite_name, is_base_model=True, strength=strength
                )
                for trial in base_suite_metrics.trials:
                    base.add_trial(trial)

                print(f"  Detection: {base_suite_metrics.suite_metrics[suite_name].detection_rate:.1%}")
                print(f"  FPR: {base_suite_metrics.suite_metrics[suite_name].false_positive_rate:.1%}")

        return introspective, base
