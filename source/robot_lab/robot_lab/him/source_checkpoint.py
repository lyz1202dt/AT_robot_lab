"""Strict checkpoint conversion for the Cyberstar 282-D observation contract."""

from __future__ import annotations

import torch


def migrate_source_state(loaded_state: dict[str, torch.Tensor], current_state: dict[str, torch.Tensor]):
    converted = {}
    prefixes = {
        "history_encoder.": "estimator.encoder.",
        "target_encoder.": "estimator.target.",
        "prototypes.": "estimator.proto.",
    }
    for key, value in loaded_state.items():
        for old, new in prefixes.items():
            if key.startswith(old):
                key = new + key[len(old) :]
                break
        converted[key] = value
    result = {}
    for key, current in current_state.items():
        value = converted.get(key)
        if value is None:
            raise RuntimeError(f"HIM checkpoint is missing {key}; refusing a partial policy transfer.")
        if value.shape == current.shape:
            result[key] = value
        elif key == "critic.0.weight" and value.shape == (current.shape[0], 297) and current.shape[1] == 491:
            # Keep the meaning of all old input columns. New privileges initially contribute zero.
            expanded = torch.zeros_like(current)
            expanded[:, :297] = value
            result[key] = expanded
        else:
            raise RuntimeError(
                f"HIM checkpoint tensor {key}: {tuple(value.shape)} is incompatible with {tuple(current.shape)}. The"
                " full Cyberstar task requires Critic491 (or source Critic297); Critic50 checkpoints need retraining."
            )
    return result
