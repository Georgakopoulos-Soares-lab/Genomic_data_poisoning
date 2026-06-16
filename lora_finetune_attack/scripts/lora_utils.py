"""
Lightweight LoRA implementation for Evo 2 7B (StripedHyena 2 architecture).

Wraps existing nn.Linear / ColumnParallelLinear / RowParallelLinear layers
with low-rank adapters. Compatible with the Savanna model's parallel linear
classes.
"""

import os
import re
import torch
import torch.nn as nn
from typing import List, Dict, Optional


class LoRALinear(nn.Module):
    """Drop-in wrapper for any linear-like layer with a LoRA adapter."""

    def __init__(
        self,
        original: nn.Module,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices — place on same device as original weights
        device = original.weight.device
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32, device=device))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize: A with Kaiming, B with zeros => LoRA starts as identity
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

        # Freeze original weights
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x, *args, **kwargs):
        # Original forward pass (frozen)
        result = self.original(x, *args, **kwargs)

        # LoRA forward pass (trainable)
        # Cast input to float32 for LoRA computation, then back
        x_f32 = self.lora_dropout(x.float())
        lora_out = (x_f32 @ self.lora_A.T @ self.lora_B.T) * self.scaling

        # Handle case where original returns a tuple (some parallel layers do)
        if isinstance(result, tuple):
            return (result[0] + lora_out.to(result[0].dtype),) + result[1:]
        return result + lora_out.to(result.dtype)

    @property
    def weight(self):
        return self.original.weight


def _get_linear_dims(module: nn.Module):
    """Extract in_features and out_features from various linear-like layers."""
    w = module.weight
    # For nn.Linear: weight is (out, in)
    # For ColumnParallelLinear: weight is (out/tp, in)  — we use full in
    # For RowParallelLinear: weight is (out, in/tp)     — we use full out
    out_features, in_features = w.shape
    return in_features, out_features


def apply_lora_to_model(
    model: nn.Module,
    target_patterns: List[str],
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
) -> Dict[str, LoRALinear]:
    """
    Apply LoRA adapters to all modules whose names match any target pattern.

    Args:
        model: The Evo2 model (e.g., evo_model.model or the backbone).
        target_patterns: List of regex patterns to match module names.
        rank: LoRA rank.
        alpha: LoRA alpha (scaling factor).
        dropout: Dropout rate on LoRA input.

    Returns:
        Dict mapping module names to LoRALinear wrappers.
    """
    # Collect all candidate modules
    candidates = []
    for name, module in model.named_modules():
        # Check if it has a weight attribute and is linear-like
        if not hasattr(module, 'weight'):
            continue
        if module.weight.ndim != 2:
            continue
        for pattern in target_patterns:
            if re.search(pattern, name):
                candidates.append((name, module))
                break

    lora_modules = {}
    total_base_params = sum(p.numel() for p in model.parameters())
    lora_param_count = 0

    for name, original in candidates:
        in_f, out_f = _get_linear_dims(original)

        # Navigate to parent and replace
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        lora_wrapper = LoRALinear(
            original, in_f, out_f,
            rank=rank, alpha=alpha, dropout=dropout,
        )
        setattr(parent, parts[-1], lora_wrapper)
        lora_modules[name] = lora_wrapper
        lora_param_count += lora_wrapper.lora_A.numel() + lora_wrapper.lora_B.numel()

    print(f"Applied LoRA to {len(lora_modules)} layers:")
    for name in sorted(lora_modules.keys()):
        m = lora_modules[name]
        print(f"  {name}: in={m.lora_A.shape[1]}, out={m.lora_B.shape[0]}, rank={rank}")
    print(f"LoRA trainable parameters: {lora_param_count:,}")
    print(f"Base model parameters:     {total_base_params:,}")
    print(f"LoRA fraction:             {lora_param_count / total_base_params * 100:.4f}%")

    return lora_modules


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Collect all LoRA parameters for the optimizer."""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params


def save_lora_weights(lora_modules: Dict[str, LoRALinear], path: str):
    """Save only the LoRA adapter weights (compact file)."""
    state = {}
    for name, module in lora_modules.items():
        state[f"{name}.lora_A"] = module.lora_A.data.cpu()
        state[f"{name}.lora_B"] = module.lora_B.data.cpu()
    # Also save metadata
    state['_metadata'] = {
        'rank': list(lora_modules.values())[0].rank,
        'alpha': list(lora_modules.values())[0].alpha,
        'n_layers': len(lora_modules),
    }
    torch.save(state, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"Saved LoRA weights to {path} ({size_mb:.1f} MB)")


def load_lora_weights(lora_modules: Dict[str, LoRALinear], path: str):
    """Load LoRA adapter weights."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    loaded = 0
    for name, module in lora_modules.items():
        a_key = f"{name}.lora_A"
        b_key = f"{name}.lora_B"
        if a_key in state and b_key in state:
            module.lora_A.data = state[a_key].to(module.lora_A.device)
            module.lora_B.data = state[b_key].to(module.lora_B.device)
            loaded += 1
    print(f"Loaded LoRA weights for {loaded}/{len(lora_modules)} layers from {path}")
