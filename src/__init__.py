"""Steering Awareness: Training LLMs to detect activation steering."""

from .hooks import InjectionHook, SteeringMode
from .vectors import VectorManager
from .models import TargetModel, load_model, LAYER_MAP

__all__ = [
    "InjectionHook",
    "SteeringMode",
    "VectorManager",
    "TargetModel",
    "load_model",
    "LAYER_MAP",
]
