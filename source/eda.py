"""Exploratory Data Analysis for the Kvasir-SEG dataset.

All functions operate on the preprocessed .npz cache files
(built by `python source/preprocessing.py`).

Quick reference
---------------
    from source.eda import (
        dataset_summary,
        plot_samples,
        plot_channel_stats,
        plot_mask_area_dist,
        plot_spatial_heatmap,
        plot_augmentation_preview,
    )

    dataset_summary()
    plot_samples("train", n=8)
    plot_channel_stats()
    plot_mask_area_dist()
    plot_spatial_heatmap()
    plot_augmentation_preview(n=4)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "cache"

# ── Seaborn global style ───────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
_PALETTE = sns.color_palette("muted")

# ImageNet mean / std (used only for un-normalizing previews)
_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ──────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────


def _load(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (images, masks) arrays from the npz cache for *split*."""
    path = CACHE_DIR / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Cache not found: {path}\n" "Run `python source/preprocessing.py` first."
        )
    data = np.load(path)
    return data["images"], data["masks"]  # (N,3,H,W), (N,1,H,W) float32


def _chw_to_hwc(img: np.ndarray) -> np.ndarray:
    """(3,H,W) → (H,W,3), clipped to [0,1]."""
    return np.clip(img.transpose(1, 2, 0), 0.0, 1.0)


def _mask_overlay(
    img_hwc: np.ndarray, mask: np.ndarray, color=(1.0, 0.2, 0.2), alpha: float = 0.45
) -> np.ndarray:
    """Blend a binary mask (H,W) onto an RGB image (H,W,3)."""
    overlay = img_hwc.copy()
    m = mask > 0.5
    for c, col in enumerate(color):
        overlay[:, :, c] = np.where(
            m, alpha * col + (1 - alpha) * img_hwc[:, :, c], img_hwc[:, :, c]
        )
    return np.clip(overlay, 0.0, 1.0)


def _foreground_ratios(masks: np.ndarray) -> np.ndarray:
    """Fraction of foreground pixels per sample. Shape (N,)."""
    return masks.mean(axis=(1, 2, 3))


# ──────────────────────────────────────────────────────────────────────────
# 1. Text summary
# ──────────────────────────────────────────────────────────────────────────


def dataset_summary(splits: list[str] | None = None) -> None:
    """Print a tabular overview of each split.

    Covers sample counts, image value range, foreground pixel statistics,
    and polyp size percentiles.

    Args:
        splits: Splits to include. Defaults to `["train", "val"]`.
    """
    if splits is None:
        splits = ["train", "val"]

    header = (
        f"{'Split':>8}  {'N':>5}  {'img_mean':>9}  {'img_std':>8}  "
        f"{'fg_mean%':>9}  {'fg_std%':>8}  {'fg_min%':>8}  {'fg_max%':>8}"
    )
    print(header)
    print("─" * len(header))

    for split in splits:
        imgs, msks = _load(split)
        ratios = _foreground_ratios(msks) * 100.0
        print(
            f"{split:>8}  {len(imgs):>5}  "
            f"{imgs.mean():>9.4f}  {imgs.std():>8.4f}  "
            f"{ratios.mean():>9.2f}  {ratios.std():>8.2f}  "
            f"{ratios.min():>8.2f}  {ratios.max():>8.2f}"
        )

    print()
    for split in splits:
        _, msks = _load(split)
        ratios = _foreground_ratios(msks) * 100.0
        pcts = np.percentile(ratios, [10, 25, 50, 75, 90])
        print(
            f"[{split}] fg-ratio percentiles  "
            f"p10={pcts[0]:.2f}%  p25={pcts[1]:.2f}%  "
            f"p50={pcts[2]:.2f}%  p75={pcts[3]:.2f}%  p90={pcts[4]:.2f}%"
        )


# ──────────────────────────────────────────────────────────────────────────
# 2. Random sample grid  (image | mask | overlay)
# ──────────────────────────────────────────────────────────────────────────


def plot_samples(
    split: str = "train",
    n: int = 6,
    seed: int = 0,
    figsize_per_col: float = 3.2,
) -> plt.Figure:
    """Display *n* random samples as (image | mask | overlay) rows.

    Args:
        split:          "train" or "val".
        n:              Number of samples to show.
        seed:           NumPy random seed for reproducible selection.
        figsize_per_col: Width per column (there are always 3 columns).

    Returns:
        The :class:`matplotlib.figure.Figure`.
    """
    imgs, msks = _load(split)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(imgs), size=min(n, len(imgs)), replace=False)

    ncols = 3  # image | mask | overlay
    nrows = len(idx)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_col * ncols, figsize_per_col * nrows),
        squeeze=False,
    )
    col_titles = ["Image", "Mask", "Overlay"]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for row, i in enumerate(idx):
        img_hwc = _chw_to_hwc(imgs[i])
        mask_hw = msks[i, 0]
        overlay = _mask_overlay(img_hwc, mask_hw)

        axes[row, 0].imshow(img_hwc)
        axes[row, 1].imshow(mask_hw, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(overlay)

        fg_pct = mask_hw.mean() * 100
        axes[row, 0].set_ylabel(f"#{i}\nfg={fg_pct:.1f}%", fontsize=9)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Kvasir-SEG — {split} split  (n={len(idx)})", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# 3. Per-channel pixel value distributions
# ──────────────────────────────────────────────────────────────────────────


def plot_channel_stats(
    splits: list[str] | None = None,
    sample_frac: float = 0.05,
    seed: int = 0,
    figsize: tuple[float, float] = (14, 4),
) -> plt.Figure:
    """KDE plot of pixel intensity for each RGB channel, per split.

    Args:
        splits:      Splits to overlay. Defaults to `["train", "val"]`.
        sample_frac: Fraction of pixels sampled per split (speeds up KDE).
        seed:        NumPy random seed.
        figsize:     Figure size `(width, height)`.

    Returns:
        The :class:`matplotlib.figure.Figure`.
    """
    if splits is None:
        splits = ["train", "val"]

    channel_names = ["Red", "Green", "Blue"]
    split_styles = ["-", "--"]
    split_colors = [_PALETTE[0], _PALETTE[1]]

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)

    for split, ls, col in zip(splits, split_styles, split_colors):
        imgs, _ = _load(split)
        rng = np.random.default_rng(seed)
        n_pixels = imgs.shape[0] * imgs.shape[2] * imgs.shape[3]
        n_sample = max(1, int(n_pixels * sample_frac))
        flat = imgs.transpose(0, 2, 3, 1).reshape(-1, 3)  # (N*H*W, 3)
        sel = rng.choice(len(flat), size=n_sample, replace=False)
        flat = flat[sel]

        for c, (ax, ch_name) in enumerate(zip(axes, channel_names)):
            sns.kdeplot(
                flat[:, c],
                ax=ax,
                label=split,
                linestyle=ls,
                color=col,
                fill=True,
                alpha=0.2,
                linewidth=1.8,
            )
            ax.set_title(ch_name, fontweight="bold")
            ax.set_xlabel("Pixel value (0–1)")

    axes[0].set_ylabel("Density")
    axes[0].legend(title="Split")
    fig.suptitle("Per-channel pixel value distributions", fontsize=13)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# 4. Foreground (polyp) area distribution
# ──────────────────────────────────────────────────────────────────────────


def plot_mask_area_dist(
    splits: list[str] | None = None,
    bins: int = 40,
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """Histogram + KDE of polyp foreground pixel ratio, per split.

    Args:
        splits:  Splits to overlay. Defaults to `["train", "val"]`.
        bins:    Number of histogram bins.
        figsize: Figure size `(width, height)`.

    Returns:
        The :class:`matplotlib.figure.Figure`.
    """
    if splits is None:
        splits = ["train", "val"]

    fig, (ax_hist, ax_box) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [3, 1]},
    )

    data_by_split: dict[str, np.ndarray] = {}
    for split in splits:
        _, msks = _load(split)
        data_by_split[split] = _foreground_ratios(msks) * 100.0

    # Histogram + KDE
    for i, (split, ratios) in enumerate(data_by_split.items()):
        color = _PALETTE[i]
        sns.histplot(
            ratios,
            bins=bins,
            ax=ax_hist,
            color=color,
            label=f"{split} (n={len(ratios)})",
            stat="density",
            alpha=0.4,
            linewidth=0,
        )
        sns.kdeplot(ratios, ax=ax_hist, color=color, linewidth=2)

    ax_hist.set_xlabel("Foreground pixel ratio (%)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Polyp area distribution")
    ax_hist.legend()

    # Box plot
    box_data = [v for v in data_by_split.values()]
    ax_box.boxplot(
        box_data,
        labels=list(data_by_split.keys()),
        patch_artist=True,
        boxprops=dict(facecolor=_PALETTE[2], alpha=0.6),
        medianprops=dict(color="black", linewidth=2),
    )
    ax_box.set_ylabel("Foreground pixel ratio (%)")
    ax_box.set_title("Box plot")

    fig.suptitle("Kvasir-SEG — polyp coverage statistics", fontsize=13)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# 5. Spatial mask heatmap  (where do polyps appear?)
# ──────────────────────────────────────────────────────────────────────────


def plot_spatial_heatmap(
    splits: list[str] | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """Show average mask activation per pixel to reveal spatial bias.

    A uniform heatmap → no spatial bias; a concentrated hot-spot → the model
    might overfit to that region.

    Args:
        splits:  Splits to visualize. Defaults to `["train", "val"]`.
        figsize: Figure size `(width, height)`.

    Returns:
        The :class:`matplotlib.figure.Figure`.
    """
    if splits is None:
        splits = ["train", "val"]

    fig, axes = plt.subplots(1, len(splits), figsize=figsize, squeeze=False)

    for ax, split in zip(axes[0], splits):
        _, msks = _load(split)
        heatmap = msks[:, 0, :, :].mean(axis=0)  # (H, W)
        im = ax.imshow(heatmap, cmap="YlOrRd", vmin=0, vmax=heatmap.max())
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean activation")
        ax.set_title(f"{split} — spatial mask heatmap", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Average polyp location across samples", fontsize=13)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# 6. Augmentation preview  (original vs augmented)
# ──────────────────────────────────────────────────────────────────────────


def plot_augmentation_preview(
    n: int = 4,
    seed: int = 7,
    figsize_per_col: float = 3.0,
) -> plt.Figure:
    """Side-by-side comparison of original vs augmented training samples.

    Uses `KvasirSegDataset` directly so the exact training augmentation
    pipeline (including spatial sync) is exercised.

    Args:
        n:               Number of samples to preview.
        seed:            NumPy seed for sample selection.
        figsize_per_col: Width per column.

    Returns:
        The :class:`matplotlib.figure.Figure`.
    """
    # import here to avoid circular import at module level
    from source.data import KvasirSegDataset

    ds_orig = KvasirSegDataset("train", augment=False)
    ds_aug = KvasirSegDataset("train", augment=True)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds_orig), size=min(n, len(ds_orig)), replace=False)

    ncols = 4  # orig_img | orig_overlay | aug_img | aug_overlay
    nrows = len(idx)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_col * ncols, figsize_per_col * nrows),
        squeeze=False,
    )
    col_titles = ["Original", "Original\nOverlay", "Augmented", "Augmented\nOverlay"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for row, i in enumerate(idx):
        img_o, msk_o = ds_orig[int(i)]
        img_a, msk_a = ds_aug[int(i)]

        # un-normalize for display (reverse ImageNet normalization)
        def _denorm(t):
            arr = t.data.astype(np.float32)  # (3,H,W)
            arr = arr * _IMG_STD[:, None, None] + _IMG_MEAN[:, None, None]
            return np.clip(arr.transpose(1, 2, 0), 0, 1)

        img_o_hwc = _denorm(img_o)
        img_a_hwc = _denorm(img_a)
        mask_o_hw = msk_o.data[0]
        mask_a_hw = msk_a.data[0]

        axes[row, 0].imshow(img_o_hwc)
        axes[row, 1].imshow(_mask_overlay(img_o_hwc, mask_o_hw))
        axes[row, 2].imshow(img_a_hwc)
        axes[row, 3].imshow(_mask_overlay(img_a_hwc, mask_a_hw))
        axes[row, 0].set_ylabel(f"#{int(i)}", fontsize=9)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    red_patch = mpatches.Patch(color=(1.0, 0.2, 0.2), alpha=0.6, label="Polyp mask")
    fig.legend(
        handles=[red_patch],
        loc="lower center",
        ncol=1,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=10,
    )
    fig.suptitle("Augmentation preview — train split", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────
# 7. Per-split image mean / std bar chart
# ──────────────────────────────────────────────────────────────────────────


def plot_channel_mean_std(
    splits: list[str] | None = None,
    figsize: tuple[float, float] = (9, 4),
) -> plt.Figure:
    """Bar chart of per-channel mean ± std for each split.

    Args:
        splits:  Splits to include. Defaults to `["train", "val"]`.
        figsize: Figure size `(width, height)`.

    Returns:
        The :class:`matplotlib.figure.Figure`.
    """
    if splits is None:
        splits = ["train", "val"]

    channel_names = ["Red", "Green", "Blue"]
    x = np.arange(len(channel_names))
    width = 0.8 / len(splits)

    fig, (ax_mean, ax_std) = plt.subplots(1, 2, figsize=figsize)

    for i, split in enumerate(splits):
        imgs, _ = _load(split)
        flat = imgs.transpose(0, 2, 3, 1).reshape(-1, 3)
        means = flat.mean(axis=0)
        stds = flat.std(axis=0)
        offset = (i - len(splits) / 2 + 0.5) * width
        color = _PALETTE[i]
        ax_mean.bar(
            x + offset,
            means,
            width=width * 0.9,
            label=split,
            color=color,
            alpha=0.8,
            yerr=None,
        )
        ax_std.bar(
            x + offset, stds, width=width * 0.9, label=split, color=color, alpha=0.8
        )

    for ax, title, ylabel in [
        (ax_mean, "Per-channel mean", "Mean pixel value"),
        (ax_std, "Per-channel std", "Std pixel value"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(channel_names)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.set_ylim(0, None)

    fig.suptitle("Channel statistics by split", fontsize=13)
    fig.tight_layout()
    return fig
