from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import Dataset


Split = Literal["train", "test", "extra"]


@dataclass(frozen=True)
class SVHNTransformConfig:
    normalize: bool = True
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    random_crop: bool = True
    crop_padding: int = 4
    horizontal_flip: bool = False
    hflip_p: float = 0.5


def _load_svhn_mat(path: str) -> tuple[np.ndarray, np.ndarray]:
    mat = loadmat(path)
    if "X" not in mat or "y" not in mat:
        raise KeyError(f"Expected keys X/y in {path}, got keys={sorted(mat.keys())}")

    x = mat["X"]  # (H, W, C, N), uint8
    y = mat["y"].reshape(-1)  # (N,)

    if x.ndim != 4 or x.shape[:3] != (32, 32, 3):
        raise ValueError(f"Unexpected X shape in {path}: {x.shape}")

    # SVHN Format 2: (H, W, C, N) -> (N, C, H, W)
    x = np.transpose(x, (3, 2, 0, 1))
    if x.dtype != np.uint8:
        x = x.astype(np.uint8, copy=False)

    # SVHN labels: digit "0" is encoded as 10. Map 10 -> 0 so labels are [0..9].
    y = y.astype(np.int64, copy=False)
    y[y == 10] = 0
    if y.min() < 0 or y.max() > 9:
        raise ValueError(f"Unexpected label range in {path}: min={y.min()} max={y.max()}")

    return x, y


class SVHNMatDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        data_dir: str,
        split: Split,
        *,
        transform: SVHNTransformConfig | None = None,
        train: bool | None = None,
        max_samples: int | None = None,
        images: np.ndarray | None = None,
        labels: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.path = os.path.join(data_dir, f"{split}_32x32.mat")
        if images is not None or labels is not None:
            if images is None or labels is None:
                raise ValueError("images and labels must be provided together")
            self.images, self.labels = images, labels
        else:
            if not os.path.exists(self.path):
                raise FileNotFoundError(self.path)
            self.images, self.labels = _load_svhn_mat(self.path)
        if max_samples is not None:
            self.images = self.images[:max_samples]
            self.labels = self.labels[:max_samples]

        self.transform = transform or SVHNTransformConfig(
            random_crop=(split in ("train", "extra")),
        )
        self.is_train = train if train is not None else (split in ("train", "extra"))

        self._mean = torch.tensor(self.transform.mean, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(self.transform.std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        if self.transform.random_crop and self.transform.crop_padding > 0:
            p = int(self.transform.crop_padding)
            img = F.pad(img, (p, p, p, p), mode="constant", value=0.0)
            top = int(torch.randint(0, 2 * p + 1, (1,)).item())
            left = int(torch.randint(0, 2 * p + 1, (1,)).item())
            img = img[:, top : top + 32, left : left + 32]

        if self.transform.horizontal_flip and float(torch.rand(1).item()) < self.transform.hflip_p:
            img = torch.flip(img, dims=(2,))

        return img

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img = torch.from_numpy(self.images[index]).to(dtype=torch.float32).div_(255.0)  # (C,H,W)
        if self.is_train:
            img = self._augment(img)

        if self.transform.normalize:
            img = (img - self._mean) / self._std

        label = int(self.labels[index])
        return img, label
