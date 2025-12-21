"""Model configuration and loading utilities."""

import os
import gc
from enum import Enum
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


class TargetModel(Enum):
    """Supported target models for steering awareness experiments."""
    # Llama family
    LLAMA_3_8B_INSTRUCT = "meta-llama/Meta-Llama-3-8B-Instruct"
    LLAMA_3_70B_INSTRUCT = "meta-llama/Meta-Llama-3-70B-Instruct"

    # DeepSeek
    DEEPSEEK_7B = "deepseek-ai/deepseek-llm-7b-chat"

    # Gemma
    GEMMA_2_9B = "google/gemma-2-9b-it"

    # Qwen
    QWEN_2_5_7B = "Qwen/Qwen2.5-7B-Instruct"
    QWEN_2_5_32B = "Qwen/Qwen2.5-32B-Instruct"

    # OpenAI
    GPT_OSS_20B = "openai/gpt-oss-20b"


# Layer indices targeting ~67% depth for each model
LAYER_MAP = {
    # Llama (32 layers for 8B, 80 layers for 70B)
    TargetModel.LLAMA_3_8B_INSTRUCT.value: 21,   # 21/32
    TargetModel.LLAMA_3_70B_INSTRUCT.value: 54,  # 54/80

    # DeepSeek (30 layers)
    TargetModel.DEEPSEEK_7B.value: 20,           # 20/30

    # Gemma (42 layers)
    TargetModel.GEMMA_2_9B.value: 28,            # 28/42

    # Qwen 2.5 (28 layers for 7B, 64 layers for 32B)
    TargetModel.QWEN_2_5_7B.value: 19,           # 19/28
    TargetModel.QWEN_2_5_32B.value: 43,          # 43/64

    # GPT-OSS (24 layers, MoE)
    TargetModel.GPT_OSS_20B.value: 16,           # 16/24
}


# Model-specific prompt templates
# Each model needs its native format for best performance
PROMPT_TEMPLATES = {
    # Llama 3 - uses special tokens
    TargetModel.LLAMA_3_8B_INSTRUCT.value: {
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}",
        "system": "<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        "bos": "<|begin_of_text|>",
    },
    TargetModel.LLAMA_3_70B_INSTRUCT.value: {
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}",
        "system": "<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        "bos": "<|begin_of_text|>",
    },
    # DeepSeek - simple format
    TargetModel.DEEPSEEK_7B.value: {
        "user": "User: {content}\n\n",
        "assistant": "Assistant: {content}",
        "system": "{content}\n\n",
        "bos": "",
    },
    # Gemma - turn-based
    TargetModel.GEMMA_2_9B.value: {
        "user": "<start_of_turn>user\n{content}<end_of_turn>\n",
        "assistant": "<start_of_turn>model\n{content}",
        "system": "{content}\n",
        "bos": "<bos>",
    },
    # Qwen - ChatML format
    TargetModel.QWEN_2_5_7B.value: {
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}",
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "bos": "",
    },
    TargetModel.QWEN_2_5_32B.value: {
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}",
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "bos": "",
    },
    # GPT-OSS - Harmony format (simplified, use tokenizer.apply_chat_template for full)
    TargetModel.GPT_OSS_20B.value: {
        "user": "<|user|>\n{content}<|end|>\n",
        "assistant": "<|assistant|>\n{content}",
        "system": "<|system|>\n{content}<|end|>\n",
        "bos": "",
        "use_chat_template": True,  # Flag to use tokenizer's built-in template
    },
}


def get_prompt_template(model_name: str) -> dict:
    """Get the prompt template for a given model."""
    if model_name in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[model_name]
    # Default fallback
    return {
        "user": "Human: {content}\n\n",
        "assistant": "Assistant: {content}",
        "system": "{content}\n\n",
        "bos": "",
    }


def format_prompt(model_name: str, user_content: str, assistant_content: str = "") -> Tuple[str, str]:
    """
    Format a prompt and completion for the given model.

    Returns:
        Tuple of (prompt, completion)
    """
    template = get_prompt_template(model_name)

    prompt = template["bos"] + template["user"].format(content=user_content)
    completion = template["assistant"].format(content=assistant_content)

    return prompt, completion


def get_device() -> str:
    """Get the best available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(
    model_name: str,
    hf_token: Optional[str] = None,
    adapter_path: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    quantize_4bit: bool = False,
    quantize_8bit: bool = False,
):
    """
    Load a model with optional LoRA adapter and quantization.

    Args:
        model_name: HuggingFace model name or path
        hf_token: HuggingFace API token for gated models
        adapter_path: Path to LoRA adapter (optional)
        torch_dtype: Model dtype (default: float16)
        device_map: Device mapping strategy
        quantize_4bit: Use 4-bit quantization (for large models)
        quantize_8bit: Use 8-bit quantization

    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup quantization config if needed
    quantization_config = None
    if quantize_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        torch_dtype = None  # Let BnB handle dtype
    elif quantize_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=hf_token,
        quantization_config=quantization_config,
        trust_remote_code=True,  # Needed for some models like Qwen
    )

    if adapter_path and os.path.exists(adapter_path):
        adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
        adapter_file_bin = os.path.join(adapter_path, "adapter_model.bin")

        if os.path.exists(adapter_file) or os.path.exists(adapter_file_bin):
            print(f"Loading LoRA adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer


def should_quantize(model_name: str) -> Tuple[bool, bool]:
    """
    Determine if a model needs quantization based on size.

    Returns:
        Tuple of (use_4bit, use_8bit)
    """
    large_models = [
        TargetModel.LLAMA_3_70B_INSTRUCT.value,
        TargetModel.QWEN_2_5_32B.value,
    ]

    if model_name in large_models:
        return True, False  # Use 4-bit for 70B+ models

    return False, False


def cleanup_model(model):
    """Free GPU memory by deleting model and clearing cache."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
