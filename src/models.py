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
    LLAMA_3_8B_INSTRUCT = "meta-llama/Meta-Llama-3-8B-Instruct"
    DEEPSEEK_7B = "deepseek-ai/deepseek-llm-7b-chat"
    GEMMA_2_9B = "google/gemma-2-9b-it"


# Layer indices targeting ~67% depth for each model
LAYER_MAP = {
    TargetModel.LLAMA_3_8B_INSTRUCT.value: 25,  # 25/32
    TargetModel.DEEPSEEK_7B.value: 20,          # 20/30
    TargetModel.GEMMA_2_9B.value: 27,           # 27/42
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
