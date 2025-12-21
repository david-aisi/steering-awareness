"""Steering vector injection hooks for transformer models."""

from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F


class SteeringMode(Enum):
    """Available steering vector application modes."""
    ADD = "add"              # Standard linear addition
    SUBTRACT = "subtract"    # Opposite steering (removal)
    REJECT = "reject"        # Remove component parallel to vector
    SCALE = "scale"          # Amplify component aligned with vector
    NULLSPACE = "nullspace"  # Project onto nullspace
    AFFINE = "affine"        # Affine transformation


class InjectionHook:
    """
    Context manager for injecting steering vectors into model activations.

    This hook intercepts the forward pass at a specified layer and modifies
    the hidden states according to the specified steering mode.

    Args:
        model: The transformer model to hook
        layer_idx: Layer index to inject vectors at
        steering_vectors: List of (vector, strength) tuples to inject
        injection_position: Token position to inject at (None = all positions)
        mode: Steering mode (add, subtract, reject, scale, nullspace, affine)
        proj_matrix: Projection matrix for nullspace mode
        affine_params: Dict with 'A', 'mu_src', 'mu_tgt' for affine mode

    Example:
        >>> with InjectionHook(model, layer_idx=25, steering_vectors=[(vec, 2.0)]):
        ...     output = model.generate(**inputs)
    """

    def __init__(
        self,
        model,
        layer_idx: int,
        steering_vectors: Optional[List[Tuple[torch.Tensor, float]]] = None,
        injection_position: Optional[int] = None,
        mode: SteeringMode = SteeringMode.ADD,
        proj_matrix: Optional[torch.Tensor] = None,
        affine_params: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.vectors = steering_vectors or []
        self.injection_position = injection_position
        self.mode = mode
        self.proj_matrix = proj_matrix
        self.affine_params = affine_params
        self.handle = None

    def _apply(self, x: torch.Tensor, total_delta: torch.Tensor) -> torch.Tensor:
        """Apply the steering transformation to activations."""

        if self.mode == SteeringMode.ADD:
            return x + total_delta

        if self.mode == SteeringMode.SUBTRACT:
            return x - total_delta

        # Compute projection onto steering direction
        v = total_delta
        v_norm = v.norm() + 1e-9
        vhat = v / v_norm

        # Component of x along direction v
        component_mag = (x * vhat).sum(dim=-1, keepdim=True)
        proj = component_mag * vhat

        # Use magnitude of total_delta as strength multiplier
        strength = v_norm

        if self.mode == SteeringMode.REJECT:
            # Remove component parallel to delta
            return x - (strength * proj)

        if self.mode == SteeringMode.SCALE:
            # Amplify component aligned with delta
            return x + (strength * proj)

        if self.mode == SteeringMode.NULLSPACE:
            assert self.proj_matrix is not None, "Nullspace mode requires proj_matrix"
            P = self.proj_matrix.to(x.device, x.dtype)
            return x @ P.T

        if self.mode == SteeringMode.AFFINE:
            assert self.affine_params is not None, "Affine mode requires affine_params"
            A = self.affine_params["A"].to(x.device, x.dtype)
            mu_src = self.affine_params["mu_src"].to(x.device, x.dtype)
            mu_tgt = self.affine_params["mu_tgt"].to(x.device, x.dtype)
            return (x - mu_src) @ A.T + mu_tgt

        raise ValueError(f"Unknown steering mode: {self.mode}")

    def _hook(self, module, inputs, output):
        """Forward hook that modifies layer activations."""
        h = output[0] if isinstance(output, tuple) else output

        # Aggregate all steering vectors
        total_delta = None
        if self.vectors:
            total_delta = torch.zeros_like(self.vectors[0][0]).to(h.device)
            for vec, strength in self.vectors:
                total_delta += strength * vec.to(h.device)

        def apply_to_slice(x):
            if self.mode in [SteeringMode.NULLSPACE, SteeringMode.AFFINE]:
                return self._apply(x, torch.zeros(x.shape[-1], device=x.device, dtype=x.dtype))
            else:
                return self._apply(x, total_delta)

        if self.injection_position is None:
            h = apply_to_slice(h)
        else:
            pos = self.injection_position
            if pos < h.shape[1]:
                h[:, pos, :] = apply_to_slice(h[:, pos, :])

        return (h,) + output[1:] if isinstance(output, tuple) else h

    def _get_layers(self, model):
        """Find the layers attribute in various model architectures."""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        if hasattr(model, "layers"):
            return model.layers
        if hasattr(model, "base_model"):
            return self._get_layers(model.base_model)
        raise AttributeError(f"Cannot find layers in model of type {type(model)}")

    def __enter__(self):
        """Register the forward hook."""
        layers = self._get_layers(self.model)
        self.handle = layers[self.layer_idx].register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove the forward hook."""
        if self.handle:
            self.handle.remove()


def generate_noise_vector(ref_vector: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """
    Generate random noise with the same magnitude as the reference vector.

    This is useful for control experiments to test if the model is detecting
    actual semantics vs. just activation perturbations.

    Args:
        ref_vector: Reference vector to match magnitude
        device: Device to place the noise vector on

    Returns:
        Random unit vector scaled to match ref_vector magnitude
    """
    noise = torch.randn_like(ref_vector).to(device)
    noise = F.normalize(noise, dim=-1)
    ref_mag = torch.norm(ref_vector, dim=-1)
    return noise * ref_mag
