"""Model configuration and loading utilities."""

import os
import gc
from enum import Enum
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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


def get_device() -> str:
    """Get the best available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(
    model_name: str,
    hf_token: Optional[str] = None,
    adapter_path: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
):
    """
    Load a model with optional LoRA adapter.

    Args:
        model_name: HuggingFace model name or path
        hf_token: HuggingFace API token for gated models
        adapter_path: Path to LoRA adapter (optional)
        torch_dtype: Model dtype (default: float16)
        device_map: Device mapping strategy

    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=hf_token,
    )

    if adapter_path and os.path.exists(adapter_path):
        adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
        adapter_file_bin = os.path.join(adapter_path, "adapter_model.bin")

        if os.path.exists(adapter_file) or os.path.exists(adapter_file_bin):
            print(f"Loading LoRA adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer


def cleanup_model(model):
    """Free GPU memory by deleting model and clearing cache."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
