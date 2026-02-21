from __future__ import annotations

import os
from typing import Any

import torch


def save_checkpoint(state: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, *, map_location: str | torch.device | None = None) -> dict[str, Any]:
    return torch.load(path, map_location=map_location)

