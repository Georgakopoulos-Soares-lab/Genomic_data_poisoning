"""
Shared embedding extraction utilities for all fine-tuning experiments.
Uses hook-based extraction from a frozen Evo2 7B backbone.
"""

import numpy as np
import torch


def get_model_layer(model, layer_idx):
    """
    Resolve the target layer from the Evo2 model.
    Tries common attribute paths; extend if the model structure differs.
    """
    for attr_path in [
        "model.layers",
        "backbone.layers",
        "layers",
        # Evo2 currently exposes transformer blocks here.
        "model.blocks",
        "backbone.blocks",
        "model.backbone.blocks",
    ]:
        obj = model
        try:
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            return obj[layer_idx]
        except (AttributeError, IndexError, TypeError):
            continue
    raise AttributeError(
        f"Cannot find layer {layer_idx}. Inspect model attributes manually."
    )


def extract_embedding(model, sequence, layer_idx, mean_pool=True):
    """
    Extract embedding from a specific layer via a forward hook.

    Args:
        model: Evo2 model instance (with .tokenizer and callable forward).
        sequence: DNA string.
        layer_idx: Which layer to hook.
        mean_pool: If True return (hidden_dim,); else (seq_len, hidden_dim).

    Returns:
        numpy array of the embedding.
    """
    activation = {}

    def hook_fn(_module, _input, output):
        activation["out"] = output[0] if isinstance(output, tuple) else output

    layer = get_model_layer(model, layer_idx)
    handle = layer.register_forward_hook(hook_fn)

    input_ids = torch.tensor(
        model.tokenizer.tokenize(sequence), dtype=torch.int
    ).unsqueeze(0).cuda()

    with torch.no_grad():
        model(input_ids)

    handle.remove()

    emb = activation["out"]  # (1, seq_len, hidden_dim)
    if mean_pool:
        emb = emb.mean(dim=1)
    # NumPy doesn't support direct conversion from torch.bfloat16 in this path.
    return emb.squeeze(0).float().cpu().numpy()


def extract_embeddings_batch(model, sequences, layer_idx, mean_pool=True,
                              device_id=0, progress=True):
    """
    Extract embeddings for a list of sequences.

    Returns:
        numpy array of shape (n, hidden_dim) or (n, seq_len, hidden_dim).
    """
    results = []
    n = len(sequences)
    for i, seq in enumerate(sequences):
        if progress and (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n}]")
        emb = extract_embedding(model, seq, layer_idx, mean_pool=mean_pool)
        results.append(emb)
    return np.array(results)
