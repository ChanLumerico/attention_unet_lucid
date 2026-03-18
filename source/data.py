"""Kvasir-SEG dataset and dataloaders using the lucid API.

Usage:
    from source.data import get_train_loader, get_val_loader

    train_loader = get_train_loader(batch_size=8)
    val_loader   = get_val_loader(batch_size=8)

    for images, masks in train_loader:
        # images: lucid.Tensor (B, 3, 256, 256), ImageNet-normalized
        # masks:  lucid.Tensor (B, 1, 256, 256), values in {0.0, 1.0}
        ...

Requires cache files built by `python source/preprocessing.py`.

Augmentation pipeline (train only)
------------------------------------
Spatial  (applied jointly to image + mask via lucid.random seed sync):
  - RandomHorizontalFlip  p=0.5
  - RandomVerticalFlip    p=0.2
  - RandomRotation        +-15 deg

Photometric (image only, mask unchanged):
  - Normalize             ImageNet mean/std
"""

from __future__ import annotations

import random as _random
import warnings as _warnings
from pathlib import Path

import numpy as np

import lucid
from lucid.data import Dataset, DataLoader
from lucid.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
)

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "cache"

_IMG_MEAN = (0.485, 0.456, 0.406)
_IMG_STD = (0.229, 0.224, 0.225)


# ──────────────────────────────────────────────────────────────
# Module-level Compose pipelines
# ──────────────────────────────────────────────────────────────

# Spatial: applied to BOTH image and mask under the same lucid.random seed.
# After applying to the mask, re-binarize (rotation/resize blur boundaries).
_spatial_aug: Compose = Compose(
    [
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.2),
        RandomRotation(15),
    ]
)

# Photometric + normalize: image only.
_img_aug: Compose = Compose(
    [
        Normalize(mean=_IMG_MEAN, std=_IMG_STD),
    ]
)

# Validation: normalize only.
_img_normalize: Normalize = Normalize(mean=_IMG_MEAN, std=_IMG_STD)


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────


class KvasirSegDataset(Dataset):
    """Kvasir-SEG polyp segmentation dataset backed by preprocessed .npz cache.

    Args:
        split:   "train" (800 samples) or "val" (200 samples).
        augment: Apply the full augmentation pipeline during __getitem__.
                 Defaults to True for split="train", False for split="val".
    """

    def __init__(self, split: str = "train", augment: bool | None = None) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.split = split
        self.augment = (split == "train") if augment is None else augment

        cache_path = CACHE_DIR / f"{split}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache file not found: {cache_path}\n"
                "Run `python source/preprocessing.py` to build the cache."
            )

        data = np.load(cache_path)
        self._images: np.ndarray = data["images"]  # (N, 3, 256, 256) float32 [0,1]
        self._masks: np.ndarray = data["masks"]  # (N, 1, 256, 256) float32 {0,1}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int):
        image_t = lucid.to_tensor(self._images[idx].copy())  # (3, H, W)
        mask_t = lucid.to_tensor(self._masks[idx].copy())  # (1, H, W)

        if self.augment:
            # Spatial: re-seed lucid.random to the same value before each call
            # so that flip/rotation/crop choices are identical for image and mask.
            seed = _random.getrandbits(31)
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", RuntimeWarning)
                lucid.random.seed(seed)
                image_t = _spatial_aug(image_t)
                lucid.random.seed(seed)
                mask_t = _spatial_aug(mask_t)

            # Re-binarize: interpolation during rotation/resize blurs boundaries.
            mask_t = lucid.tensor((mask_t.data > 0.5).astype(np.float32))

            # Photometric augmentation + normalize (image only).
            image_t = _img_aug(image_t)
        else:
            image_t = _img_normalize(image_t)

        return image_t, mask_t

    def __repr__(self) -> str:
        return (
            f"KvasirSegDataset(split={self.split!r}, "
            f"n={len(self)}, augment={self.augment})"
        )


# ──────────────────────────────────────────────────────────────
# Convenience loader factories
# ──────────────────────────────────────────────────────────────


def get_train_loader(batch_size: int = 8, shuffle: bool = True) -> DataLoader:
    """Return a DataLoader over the training split (800 samples) with augmentation."""
    return DataLoader(KvasirSegDataset("train"), batch_size=batch_size, shuffle=shuffle)


def get_val_loader(batch_size: int = 8, shuffle: bool = False) -> DataLoader:
    """Return a DataLoader over the validation split (200 samples), no augmentation."""
    return DataLoader(KvasirSegDataset("val"), batch_size=batch_size, shuffle=shuffle)
