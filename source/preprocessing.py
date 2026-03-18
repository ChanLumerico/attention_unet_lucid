#!/usr/bin/env python3
"""Download and preprocess the Kvasir-SEG dataset into .npz cache files.

Usage:
    python source/preprocessing.py

Outputs:
    cache/train.npz  — 800 samples, keys: images (800,3,256,256), masks (800,1,256,256)
    cache/val.npz    — 200 samples, keys: images (200,3,256,256), masks (200,1,256,256)
"""

import ssl
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
DATASET_DIR = DATA_DIR / "kvasir-seg"

DOWNLOAD_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
ZIP_PATH = DATA_DIR / "kvasir-seg.zip"

IMG_SIZE = (256, 256)
SEED = 42
VAL_RATIO = 0.2


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        print(
            f"\r  {pct:.1f}%  ({downloaded // 1_048_576} / {total_size // 1_048_576} MB)",
            end="",
            flush=True,
        )


def download() -> None:
    if ZIP_PATH.exists():
        print(f"Zip already exists at {ZIP_PATH}, skipping download.")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kvasir-SEG from:\n  {DOWNLOAD_URL}")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    with opener.open(DOWNLOAD_URL) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 65536
        with open(ZIP_PATH, "wb") as f:
            while True:
                buf = response.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                _progress_hook(downloaded // chunk, chunk, total)
    print()  # newline after progress
    print(f"Saved to {ZIP_PATH}")


def extract() -> None:
    if DATASET_DIR.exists():
        print(f"Dataset already extracted at {DATASET_DIR}, skipping extraction.")
        return
    print(f"Extracting {ZIP_PATH} → {DATA_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    print("Extraction complete.")


def preprocess() -> None:
    train_out = CACHE_DIR / "train.npz"
    val_out = CACHE_DIR / "val.npz"
    if train_out.exists() and val_out.exists():
        print("Cache files already exist, skipping preprocessing.")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    image_dir = DATASET_DIR / "images"
    mask_dir = DATASET_DIR / "masks"

    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found in {image_dir}")

    print(f"Processing {len(image_paths)} image/mask pairs ...")
    images = []
    masks = []

    for i, img_path in enumerate(image_paths, 1):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
        msk = Image.open(mask_path).convert("L").resize(IMG_SIZE, Image.NEAREST)

        # (3, H, W) float32, values in [0, 1]
        img_arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        # (1, H, W) float32, binary {0.0, 1.0}
        msk_arr = (np.array(msk, dtype=np.float32) > 127).astype(np.float32)[np.newaxis]

        images.append(img_arr)
        masks.append(msk_arr)

        if i % 100 == 0:
            print(f"  [{i}/{len(image_paths)}]")

    images = np.stack(images)  # (N, 3, H, W)
    masks = np.stack(masks)  # (N, 1, H, W)

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(images))
    n_val = int(len(images) * VAL_RATIO)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    np.savez_compressed(train_out, images=images[train_idx], masks=masks[train_idx])
    np.savez_compressed(val_out, images=images[val_idx], masks=masks[val_idx])

    print(f"Saved train ({len(train_idx)} samples) → {train_out}")
    print(f"Saved val   ({len(val_idx)} samples) → {val_out}")


if __name__ == "__main__":
    download()
    extract()
    preprocess()
    print("Done.")
