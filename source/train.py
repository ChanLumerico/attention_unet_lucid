#!/usr/bin/env python3
"""Train AttentionUNet2d on Kvasir-SEG with YAML-managed experiments.

Usage:
    python source/train.py --config config/attention_unet.yaml
    python source/train.py --save-default-config config/attention_unet.yaml

Outputs:
    checkpoints/<config_name>/
      - experiment_config.yaml
      - latest/
          - model.safetensors
          - training_state.lcd
          - metrics.npz
      - best/
          - model.safetensors
          - training_state.lcd
          - metrics.npz

    out/<config_name>/epoch_xxx.png
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lucid
import lucid.optim as optim
import lucid.optim.lr_scheduler as lr_scheduler
from lucid import nn
from lucid.nn import functional as F

from source.data import KvasirSegDataset, get_train_loader, get_val_loader
from source.model import (
    build_model,
    config_to_dict,
    load_config as load_model_config,
)


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.9


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    monitor: str = "valid_loss"
    mode: str = "min"
    t_max: int | None = None
    eta_min: float = 1e-6
    step_size: int = 10
    gamma: float = 0.5
    factor: float = 0.5
    patience: int = 3
    min_lr: float = 1e-6


@dataclass
class LossConfig:
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    aux_weight: float = 0.3
    dice_smooth: float = 1.0


@dataclass
class TrainConfig:
    run_name: str = "attention_unet_baseline"
    output_root: str = "checkpoints"
    seed: int = 42
    device: str = "gpu"
    batch_size: int = 8
    epochs: int = 30
    log_every: int = 10
    threshold: float = 0.5
    monitor: str = "valid_dice"
    monitor_mode: str = "max"
    save_best: bool = True
    save_last: bool = True
    resume_from: str | None = None
    max_train_steps_per_epoch: int | None = None
    max_val_steps_per_epoch: int | None = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)


def _as_tuple(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise ValueError(f"Expected a length-2 list/tuple, got {value!r}")


def load_train_config(path: str | Path) -> TrainConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Train config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    train_raw = raw.get("train", raw)
    optimizer_raw = train_raw.get("optimizer", {})
    scheduler_raw = train_raw.get("scheduler", {})
    loss_raw = train_raw.get("loss", {})

    return TrainConfig(
        run_name=train_raw.get("run_name", "attention_unet_baseline"),
        output_root=train_raw.get("output_root", "checkpoints"),
        seed=int(train_raw.get("seed", 42)),
        device=train_raw.get("device", "gpu"),
        batch_size=int(train_raw.get("batch_size", 8)),
        epochs=int(train_raw.get("epochs", 30)),
        log_every=int(train_raw.get("log_every", 10)),
        threshold=float(train_raw.get("threshold", 0.5)),
        monitor=_canonical_monitor(train_raw.get("monitor", "valid_dice")),
        monitor_mode=train_raw.get("monitor_mode", "max"),
        save_best=bool(train_raw.get("save_best", True)),
        save_last=bool(train_raw.get("save_last", True)),
        resume_from=train_raw.get("resume_from"),
        max_train_steps_per_epoch=train_raw.get("max_train_steps_per_epoch"),
        max_val_steps_per_epoch=train_raw.get("max_val_steps_per_epoch"),
        optimizer=OptimizerConfig(
            name=optimizer_raw.get("name", "adamw"),
            lr=float(optimizer_raw.get("lr", 3e-4)),
            weight_decay=float(optimizer_raw.get("weight_decay", 1e-4)),
            betas=_as_tuple(optimizer_raw.get("betas"), (0.9, 0.999)),
            eps=float(optimizer_raw.get("eps", 1e-8)),
            momentum=float(optimizer_raw.get("momentum", 0.9)),
        ),
        scheduler=SchedulerConfig(
            name=scheduler_raw.get("name", "cosine"),
            monitor=_canonical_monitor(scheduler_raw.get("monitor", "valid_loss")),
            mode=scheduler_raw.get("mode", "min"),
            t_max=scheduler_raw.get("t_max"),
            eta_min=float(scheduler_raw.get("eta_min", 1e-6)),
            step_size=int(scheduler_raw.get("step_size", 10)),
            gamma=float(scheduler_raw.get("gamma", 0.5)),
            factor=float(scheduler_raw.get("factor", 0.5)),
            patience=int(scheduler_raw.get("patience", 3)),
            min_lr=float(scheduler_raw.get("min_lr", 1e-6)),
        ),
        loss=LossConfig(
            bce_weight=float(loss_raw.get("bce_weight", 0.5)),
            dice_weight=float(loss_raw.get("dice_weight", 0.5)),
            aux_weight=float(loss_raw.get("aux_weight", 0.3)),
            dice_smooth=float(loss_raw.get("dice_smooth", 1.0)),
        ),
    )


def _default_model_section() -> dict[str, Any]:
    return {
        "in_channels": 3,
        "out_channels": 1,
        "norm": "batch",
        "act": "relu",
        "downsample_mode": "maxpool",
        "upsample_mode": "bilinear",
        "deep_supervision": True,
        "align_corners": False,
        "bias": None,
        "stem_channels": None,
        "final_kernel_size": 1,
        "encoder_stages": [
            {
                "channels": 64,
                "num_blocks": 2,
                "kernel_size": 3,
                "dilation": 1,
                "use_attention": False,
                "dropout": 0.0,
            },
            {
                "channels": 128,
                "num_blocks": 2,
                "kernel_size": 3,
                "dilation": 1,
                "use_attention": False,
                "dropout": 0.0,
            },
            {
                "channels": 256,
                "num_blocks": 2,
                "kernel_size": 3,
                "dilation": 1,
                "use_attention": False,
                "dropout": 0.0,
            },
            {
                "channels": 512,
                "num_blocks": 2,
                "kernel_size": 3,
                "dilation": 1,
                "use_attention": False,
                "dropout": 0.0,
            },
        ],
        "decoder_stages": None,
        "bottleneck": {
            "channels": 1024,
            "num_blocks": 2,
            "kernel_size": 3,
            "dilation": 1,
            "use_attention": False,
            "dropout": 0.0,
        },
        "attention": {
            "enabled": True,
            "mode": "additive",
            "gate_activation": "relu",
            "attention_activation": "sigmoid",
            "use_grid_attention": True,
            "inter_channels": None,
            "attention_channels": 1,
            "resample_mode": "bilinear",
            "gate_on_skips": None,
            "skip_low_level_gates": False,
            "use_multi_scale_gating": True,
            "project_skip_with_1x1": True,
            "project_gating_with_1x1": True,
            "init_pass_through": True,
        },
    }


def save_experiment_config(
    train_config: TrainConfig,
    model_config,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = _experiment_config_doc(train_config, model_config)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)


def save_default_experiment_config(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "train": asdict(TrainConfig()),
        "model": _default_model_section(),
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)


def _canonical_monitor(name: str) -> str:
    if name.startswith("val_"):
        return "valid_" + name.removeprefix("val_")
    return name


def _normalize_config_doc(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize_config_doc(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_config_doc(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _experiment_config_doc(train_config: TrainConfig, model_config) -> dict[str, Any]:
    return {
        "train": _normalize_config_doc(asdict(train_config)),
        "model": _normalize_config_doc(config_to_dict(model_config)),
    }


def _load_yaml_doc(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return raw


def _experiment_config_matches(checkpoint_root: Path, current_doc: dict[str, Any]) -> bool:
    saved_path = checkpoint_root / "experiment_config.yaml"
    if not saved_path.exists():
        return True
    saved_doc = _load_yaml_doc(saved_path)
    return _normalize_config_doc(saved_doc) == _normalize_config_doc(current_doc)


def _sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in name).strip(
        "_"
    ) or "run"


def resolve_config_name(config_path: str | Path) -> str:
    return _sanitize_name(Path(config_path).stem)


def get_checkpoint_root(
    config_name: str, output_root: str = "checkpoints"
) -> Path:
    return ROOT / output_root / _sanitize_name(config_name)


def get_bundle_dir(
    config_name: str,
    which: str = "latest",
    output_root: str = "checkpoints",
) -> Path:
    if which not in {"latest", "best"}:
        raise ValueError("which must be 'latest' or 'best'")
    return get_checkpoint_root(config_name, output_root=output_root) / which


def _has_checkpoint_bundle(bundle_dir: Path) -> bool:
    return (
        (bundle_dir / "model.safetensors").exists()
        and (bundle_dir / "training_state.lcd").exists()
        and (bundle_dir / "metrics.npz").exists()
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    lucid.random.seed(seed)


def _validate_config(config: TrainConfig) -> None:
    valid_monitors = {
        "train_loss",
        "train_dice",
        "train_iou",
        "train_foreground_dice",
        "valid_loss",
        "valid_dice",
        "valid_iou",
        "valid_foreground_dice",
    }
    if config.device not in {"cpu", "gpu"}:
        raise ValueError(f"device must be 'cpu' or 'gpu', got {config.device!r}")
    if config.monitor_mode not in {"min", "max"}:
        raise ValueError(
            f"monitor_mode must be 'min' or 'max', got {config.monitor_mode!r}"
        )
    if config.scheduler.mode not in {"min", "max"}:
        raise ValueError(
            f"scheduler.mode must be 'min' or 'max', got {config.scheduler.mode!r}"
        )
    if config.monitor not in valid_monitors:
        raise ValueError(f"Unsupported monitor: {config.monitor!r}")
    if config.scheduler.monitor not in valid_monitors:
        raise ValueError(f"Unsupported scheduler monitor: {config.scheduler.monitor!r}")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.max_train_steps_per_epoch is not None and config.max_train_steps_per_epoch <= 0:
        raise ValueError("max_train_steps_per_epoch must be positive or null")
    if config.max_val_steps_per_epoch is not None and config.max_val_steps_per_epoch <= 0:
        raise ValueError("max_val_steps_per_epoch must be positive or null")


def _build_optimizer(model: nn.Module, config: TrainConfig):
    name = config.optimizer.name.lower()
    params = model.parameters()

    if name == "adam":
        return optim.Adam(
            params,
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )
    if name == "adamw":
        return optim.AdamW(
            params,
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )
    if name == "sgd":
        return optim.SGD(
            params,
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer.name!r}")


def _build_scheduler(optimizer: optim.Optimizer, config: TrainConfig):
    name = config.scheduler.name.lower()
    if name == "none":
        return None
    if name == "cosine":
        t_max = config.scheduler.t_max or config.epochs
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=config.scheduler.eta_min,
        )
    if name == "step":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler.step_size,
            gamma=config.scheduler.gamma,
        )
    if name == "plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.scheduler.mode,
            factor=config.scheduler.factor,
            patience=config.scheduler.patience,
            min_lr=config.scheduler.min_lr,
        )
    raise ValueError(f"Unsupported scheduler: {config.scheduler.name!r}")


def _extract_logits(outputs: Any) -> tuple[Any, list[Any]]:
    if isinstance(outputs, dict):
        main = outputs["out"]
        aux = list(outputs.get("aux", []))
        return main, aux
    return outputs, []


def _resize_like(logits, target):
    if logits.shape[-2:] == target.shape[-2:]:
        return logits
    return F.interpolate(
        logits,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def _dice_loss(logits, targets, smooth: float):
    probs = F.sigmoid(lucid.clip(logits, -60.0, 60.0))
    dims = tuple(range(1, probs.ndim))
    intersection = lucid.sum(probs * targets, axis=dims)
    denominator = lucid.sum(probs, axis=dims) + lucid.sum(targets, axis=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - lucid.mean(dice)


def _segmentation_loss(outputs: Any, targets, config: TrainConfig):
    main_logits, aux_logits = _extract_logits(outputs)

    def _single_loss(logits):
        logits = _resize_like(logits, targets)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        dice_loss = _dice_loss(logits, targets, smooth=config.loss.dice_smooth)
        return (
            config.loss.bce_weight * bce_loss
            + config.loss.dice_weight * dice_loss
        )

    total_loss = _single_loss(main_logits)
    if aux_logits and config.loss.aux_weight > 0:
        aux_loss = None
        for aux in aux_logits:
            current = _single_loss(aux)
            aux_loss = current if aux_loss is None else aux_loss + current
        aux_loss = aux_loss / len(aux_logits)
        total_loss = total_loss + config.loss.aux_weight * aux_loss

    return total_loss, _resize_like(main_logits, targets)


def _batch_metrics(logits, targets, threshold: float, eps: float = 1e-7) -> dict[str, float]:
    logits_np = np.clip(_to_numpy_float32(logits), -60.0, 60.0)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    preds = (probs >= threshold).astype(np.float32)
    target_np = _to_numpy_float32(targets)

    axes = (1, 2, 3)
    intersection = (preds * target_np).sum(axis=axes)
    pred_area = preds.sum(axis=axes)
    target_area = target_np.sum(axis=axes)
    union = pred_area + target_area - intersection

    dice = (2.0 * intersection + eps) / (pred_area + target_area + eps)
    iou = (intersection + eps) / (union + eps)
    foreground_mask = target_area > eps

    return {
        "dice_sum": float(dice.sum()),
        "dice_count": int(dice.shape[0]),
        "iou_sum": float(iou.sum()),
        "iou_count": int(iou.shape[0]),
        "foreground_dice_sum": float(dice[foreground_mask].sum())
        if foreground_mask.any()
        else 0.0,
        "foreground_count": int(foreground_mask.sum()),
    }


def _current_lr(optimizer: optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _display_metric(sum_value: float, count: int) -> str:
    if count <= 0:
        return "nan"
    return f"{sum_value / count:.4f}"


def _run_epoch(
    *,
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer | None,
    config: TrainConfig,
    training: bool,
) -> tuple[dict[str, float], list[float]]:
    model.train(training)

    totals = {
        "loss": 0.0,
        "samples": 0,
        "dice_sum": 0.0,
        "dice_count": 0,
        "foreground_dice_sum": 0.0,
        "foreground_count": 0,
        "iou_sum": 0.0,
        "iou_count": 0,
    }
    step_limit = (
        config.max_train_steps_per_epoch if training else config.max_val_steps_per_epoch
    )
    total_steps = len(loader)
    active_steps = (
        min(total_steps, step_limit) if step_limit is not None else total_steps
    )
    phase = "train" if training else "valid"
    pbar = tqdm(loader, total=active_steps, desc=phase, leave=False, dynamic_ncols=True)
    running_loss_trace: list[float] = []

    for step, (images, masks) in enumerate(pbar, start=1):
        if step_limit is not None and step > step_limit:
            break

        images = images.to(config.device)
        masks = masks.to(config.device)
        batch_size = int(images.shape[0])

        if training:
            optimizer.zero_grad()
            outputs = model(images)
            loss, logits = _segmentation_loss(outputs, masks, config)
            loss.backward()
            optimizer.step()
        else:
            with lucid.no_grad():
                outputs = model(images)
                loss, logits = _segmentation_loss(outputs, masks, config)

        metrics = _batch_metrics(logits, masks, threshold=config.threshold)
        totals["loss"] += float(loss.item()) * batch_size
        totals["dice_sum"] += metrics["dice_sum"]
        totals["dice_count"] += metrics["dice_count"]
        totals["foreground_dice_sum"] += metrics["foreground_dice_sum"]
        totals["foreground_count"] += metrics["foreground_count"]
        totals["iou_sum"] += metrics["iou_sum"]
        totals["iou_count"] += metrics["iou_count"]
        totals["samples"] += batch_size

        running_loss = totals["loss"] / max(1, totals["samples"])
        postfix = {
            "loss": f"{running_loss:.4f}",
            "dice": _display_metric(totals["dice_sum"], totals["dice_count"]),
            "fg_dice": _display_metric(
                totals["foreground_dice_sum"], totals["foreground_count"]
            ),
        }
        if training:
            postfix["lr"] = f"{_current_lr(optimizer):.2e}"
            running_loss_trace.append(float(running_loss))
        pbar.set_postfix(postfix)

    pbar.close()

    samples = max(1, totals["samples"])
    foreground_count = max(1, totals["foreground_count"])
    metrics = {
        "loss": totals["loss"] / samples,
        "dice": totals["dice_sum"] / max(1, totals["dice_count"]),
        "foreground_dice": totals["foreground_dice_sum"] / foreground_count,
        "iou": totals["iou_sum"] / max(1, totals["iou_count"]),
    }
    return metrics, running_loss_trace


def _is_better(current: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return current < best
    return current > best


def _metric_table(
    train_metrics: dict[str, float], val_metrics: dict[str, float]
) -> dict[str, float]:
    return {
        "train_loss": train_metrics["loss"],
        "train_dice": train_metrics["dice"],
        "train_iou": train_metrics["iou"],
        "train_foreground_dice": train_metrics["foreground_dice"],
        "valid_loss": val_metrics["loss"],
        "valid_dice": val_metrics["dice"],
        "valid_iou": val_metrics["iou"],
        "valid_foreground_dice": val_metrics["foreground_dice"],
    }


def _empty_history() -> dict[str, list[float]]:
    return {
        "epoch": [],
        "lr": [],
        "train_iteration": [],
        "running_train_loss": [],
        "train_loss": [],
        "train_dice": [],
        "train_foreground_dice": [],
        "valid_loss": [],
        "valid_dice": [],
        "valid_foreground_dice": [],
    }


def _append_running_train_history(
    history: dict[str, list[float]],
    running_loss_trace: list[float],
) -> None:
    start_iter = int(history["train_iteration"][-1]) if history["train_iteration"] else 0
    for offset, loss in enumerate(running_loss_trace, start=1):
        history["train_iteration"].append(float(start_iter + offset))
        history["running_train_loss"].append(float(loss))


def _append_epoch_history(
    history: dict[str, list[float]], epoch: int, lr: float, metrics_table: dict[str, float]
) -> None:
    history["epoch"].append(float(epoch))
    history["lr"].append(float(lr))
    for key in history:
        if key in {"epoch", "lr", "train_iteration", "running_train_loss"}:
            continue
        history[key].append(float(metrics_table[key]))


def _history_to_npz_dict(history: dict[str, list[float]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, values in history.items():
        dtype = np.int32 if key in {"epoch", "train_iteration"} else np.float32
        out[key] = np.asarray(values, dtype=dtype)
    return out


def _load_history_npz(path: Path) -> dict[str, list[float]]:
    history = _empty_history()
    data = np.load(path)
    for key in history:
        if key in data:
            history[key] = data[key].tolist()
    return history


def _save_checkpoint_bundle(
    *,
    bundle_dir: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    history: dict[str, list[float]],
    epoch: int,
    best_epoch: int | None,
    best_value: float | None,
    monitor: str,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    lucid.save(model.state_dict(), bundle_dir / "model.safetensors")
    lucid.save(
        {
            "epoch": int(epoch),
            "best_epoch": -1 if best_epoch is None else int(best_epoch),
            "best_value": None if best_value is None else float(best_value),
            "monitor": monitor,
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
        },
        bundle_dir / "training_state.lcd",
    )
    np.savez_compressed(bundle_dir / "metrics.npz", **_history_to_npz_dict(history))


def _load_checkpoint_bundle(
    *,
    bundle_dir: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
) -> tuple[int, int | None, float | None, dict[str, list[float]]]:
    model_path = bundle_dir / "model.safetensors"
    state_path = bundle_dir / "training_state.lcd"
    metrics_path = bundle_dir / "metrics.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint model not found: {model_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"Checkpoint optimizer/scheduler state not found: {state_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Checkpoint metrics not found: {metrics_path}")

    model.load_state_dict(lucid.load(model_path))
    state = lucid.load(state_path)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])

    best_epoch_raw = int(state.get("best_epoch", -1))
    best_epoch = None if best_epoch_raw < 0 else best_epoch_raw
    best_value = state.get("best_value")
    history = _load_history_npz(metrics_path)
    return int(state["epoch"]), best_epoch, best_value, history


def _plot_history(
    history: dict[str, list[float]], *, out_dir: Path, epoch: int
) -> Path:
    fig = make_history_figure(history)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "epoch_latest.png"
    fig.savefig(plot_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def make_history_figure(history: dict[str, list[float]]) -> plt.Figure:
    epochs = np.asarray(history["epoch"], dtype=np.int32)
    train_iterations = np.asarray(history["train_iteration"], dtype=np.int32)
    running_train_loss = np.asarray(history["running_train_loss"], dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(
        train_iterations,
        running_train_loss,
        color="#4C72B0",
        linewidth=1.8,
    )
    axes[0].set_title("Running Train Loss")
    axes[0].set_xlabel("Train Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    if len(train_iterations) == 1:
        axes[0].set_xlim(float(train_iterations[0]) - 0.5, float(train_iterations[0]) + 0.5)

    specs = [
        ("loss", "Loss"),
        ("dice", "Dice"),
    ]

    for ax, (suffix, title) in zip(axes[1:], specs):
        ax.plot(
            epochs,
            history[f"train_{suffix}"],
            label="train",
            linewidth=2,
            marker="o",
            markersize=6,
        )
        ax.plot(
            epochs,
            history[f"valid_{suffix}"],
            label="valid",
            linewidth=2,
            marker="o",
            markersize=6,
        )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        if len(epochs) == 1:
            ax.set_xlim(float(epochs[0]) - 0.5, float(epochs[0]) + 0.5)
        if suffix == "loss":
            ax.set_ylabel("Value")
        if suffix != "loss":
            ax.set_ylim(0.0, 1.0)
        ax.legend()

    fig.tight_layout()
    return fig


def load_history_from_bundle(bundle_dir: str | Path) -> dict[str, list[float]]:
    bundle_dir = Path(bundle_dir)
    return _load_history_npz(bundle_dir / "metrics.npz")


def plot_checkpoint_history(
    config_path: str | Path, which: str = "latest"
) -> plt.Figure:
    config_path = Path(config_path)
    train_config = load_train_config(config_path)
    config_name = resolve_config_name(config_path)
    bundle_dir = get_bundle_dir(
        config_name,
        which=which,
        output_root=train_config.output_root,
    )
    history = load_history_from_bundle(bundle_dir)
    return make_history_figure(history)


_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_numpy_float32(tensor) -> np.ndarray:
    return tensor.numpy().astype(np.float32)


def _denorm_image(img_chw: np.ndarray) -> np.ndarray:
    img = img_chw.astype(np.float32) * _IMG_STD[:, None, None] + _IMG_MEAN[:, None, None]
    return np.clip(img.transpose(1, 2, 0), 0.0, 1.0)


def _blend_mask(
    img_hwc: np.ndarray,
    mask_hw: np.ndarray,
    *,
    color: tuple[float, float, float],
    alpha: float,
) -> np.ndarray:
    overlay = img_hwc.copy()
    active = mask_hw > 0.5
    for channel, value in enumerate(color):
        overlay[..., channel] = np.where(
            active,
            alpha * value + (1.0 - alpha) * overlay[..., channel],
            overlay[..., channel],
        )
    return np.clip(overlay, 0.0, 1.0)


def load_model_from_checkpoint(
    config_path: str | Path,
    *,
    which: str = "best",
    device: str | None = None,
):
    config_path = Path(config_path)
    train_config = load_train_config(config_path)
    config_name = resolve_config_name(config_path)
    bundle_dir = get_bundle_dir(
        config_name,
        which=which,
        output_root=train_config.output_root,
    )
    return load_model_from_bundle(config_path, bundle_dir=bundle_dir, device=device)


def load_model_from_bundle(
    config_path: str | Path,
    *,
    bundle_dir: str | Path,
    device: str | None = None,
):
    config_path = Path(config_path)
    bundle_dir = Path(bundle_dir)
    if not _has_checkpoint_bundle(bundle_dir):
        raise FileNotFoundError(f"Checkpoint bundle is incomplete: {bundle_dir}")
    train_config = load_train_config(config_path)
    target_device = train_config.device if device is None else device
    model_config = load_model_config(config_path)
    model = build_model(model_config).to(target_device)
    model.load_state_dict(lucid.load(bundle_dir / "model.safetensors"))
    model.eval()
    return model, train_config, bundle_dir


def _predict_dataset_split(
    config_path: str | Path,
    *,
    which: str = "best",
    split: str = "val",
    device: str | None = None,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    model, train_config, _ = load_model_from_checkpoint(
        config_path,
        which=which,
        device=device,
    )
    if split == "train":
        loader = get_train_loader(batch_size=train_config.batch_size, shuffle=False)
    elif split == "val":
        loader = get_val_loader(batch_size=train_config.batch_size, shuffle=False)
    else:
        raise ValueError("split must be 'train' or 'val'")

    dice_values: list[np.ndarray] = []
    pred_ratios: list[np.ndarray] = []
    target_ratios: list[np.ndarray] = []

    for step, (images, masks) in enumerate(loader, start=1):
        if max_batches is not None and step > max_batches:
            break
        images = images.to(model.device)
        with lucid.no_grad():
            outputs = model(images)
            logits, _ = _extract_logits(outputs)
        logits_np = np.clip(_to_numpy_float32(logits), -60.0, 60.0)
        probs = 1.0 / (1.0 + np.exp(-logits_np))
        preds = (probs >= train_config.threshold).astype(np.float32)
        target_np = _to_numpy_float32(masks)
        axes = (1, 2, 3)
        intersection = (preds * target_np).sum(axis=axes)
        pred_area = preds.sum(axis=axes)
        target_area = target_np.sum(axis=axes)
        dice = (2.0 * intersection + 1e-7) / (pred_area + target_area + 1e-7)

        dice_values.append(dice)
        pred_ratios.append(pred_area / np.prod(target_np.shape[-2:]))
        target_ratios.append(target_area / np.prod(target_np.shape[-2:]))

    return {
        "dice": np.concatenate(dice_values, axis=0),
        "pred_ratio": np.concatenate(pred_ratios, axis=0),
        "target_ratio": np.concatenate(target_ratios, axis=0),
    }


def sweep_thresholds(
    config_path: str | Path,
    *,
    which: str = "best",
    split: str = "val",
    thresholds: list[float] | tuple[float, ...] | np.ndarray | None = None,
    device: str | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    train_config = load_train_config(config_path)
    config_name = resolve_config_name(config_path)
    bundle_dir = get_bundle_dir(
        config_name,
        which=which,
        output_root=train_config.output_root,
    )
    return sweep_thresholds_from_bundle(
        config_path,
        bundle_dir=bundle_dir,
        split=split,
        thresholds=thresholds,
        device=device,
        max_batches=max_batches,
    )


def sweep_thresholds_from_bundle(
    config_path: str | Path,
    *,
    bundle_dir: str | Path,
    split: str = "val",
    thresholds: list[float] | tuple[float, ...] | np.ndarray | None = None,
    device: str | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model, train_config, _ = load_model_from_bundle(
        config_path,
        bundle_dir=bundle_dir,
        device=device,
    )
    if split == "train":
        loader = get_train_loader(batch_size=1, shuffle=False)
    elif split == "val":
        loader = get_val_loader(batch_size=1, shuffle=False)
    else:
        raise ValueError("split must be 'train' or 'val'")

    threshold_values = np.asarray(
        np.linspace(0.2, 0.6, 17) if thresholds is None else thresholds,
        dtype=np.float32,
    )
    if threshold_values.ndim != 1 or len(threshold_values) == 0:
        raise ValueError("thresholds must be a non-empty 1D sequence")

    prob_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []

    for step, (images, masks) in enumerate(loader, start=1):
        if max_batches is not None and step > max_batches:
            break
        images = images.to(model.device)
        with lucid.no_grad():
            outputs = model(images)
            logits, _ = _extract_logits(outputs)
        logits_np = np.clip(_to_numpy_float32(logits), -60.0, 60.0)
        probs = 1.0 / (1.0 + np.exp(-logits_np))
        prob_batches.append(probs)
        target_batches.append(_to_numpy_float32(masks))

    probs_all = np.concatenate(prob_batches, axis=0)
    targets_all = np.concatenate(target_batches, axis=0)
    axes = (1, 2, 3)
    target_area = targets_all.sum(axis=axes)
    pixels_per_sample = float(np.prod(targets_all.shape[-2:]))

    mean_dice = []
    median_dice = []
    std_dice = []
    mean_pred_ratio = []

    for threshold in threshold_values:
        preds = (probs_all >= float(threshold)).astype(np.float32)
        intersection = (preds * targets_all).sum(axis=axes)
        pred_area = preds.sum(axis=axes)
        dice = (2.0 * intersection + 1e-7) / (pred_area + target_area + 1e-7)
        mean_dice.append(float(dice.mean()))
        median_dice.append(float(np.median(dice)))
        std_dice.append(float(dice.std()))
        mean_pred_ratio.append(float((pred_area / pixels_per_sample).mean()))

    mean_dice_arr = np.asarray(mean_dice, dtype=np.float32)
    best_index = int(np.argmax(mean_dice_arr))
    return {
        "thresholds": threshold_values,
        "mean_dice": mean_dice_arr,
        "median_dice": np.asarray(median_dice, dtype=np.float32),
        "std_dice": np.asarray(std_dice, dtype=np.float32),
        "mean_pred_ratio": np.asarray(mean_pred_ratio, dtype=np.float32),
        "mean_target_ratio": float((target_area / pixels_per_sample).mean()),
        "best_index": best_index,
        "best_threshold": float(threshold_values[best_index]),
        "best_mean_dice": float(mean_dice_arr[best_index]),
    }


def plot_prediction_diagnostics(
    config_path: str | Path,
    *,
    which: str = "best",
    split: str = "val",
    device: str | None = None,
) -> plt.Figure:
    diagnostics = _predict_dataset_split(
        config_path,
        which=which,
        split=split,
        device=device,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(diagnostics["dice"], bins=20, color="#4C72B0", alpha=0.85)
    axes[0].set_title(f"{split} dice distribution")
    axes[0].set_xlabel("Dice")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(
        diagnostics["target_ratio"] * 100.0,
        diagnostics["pred_ratio"] * 100.0,
        alpha=0.7,
        s=24,
        color="#DD8452",
    )
    limit = max(
        float(diagnostics["target_ratio"].max()),
        float(diagnostics["pred_ratio"].max()),
        1e-6,
    )
    axes[1].plot([0, limit * 100.0], [0, limit * 100.0], "--", color="black", linewidth=1)
    axes[1].set_title(f"{split} foreground ratio: target vs prediction")
    axes[1].set_xlabel("Target foreground (%)")
    axes[1].set_ylabel("Predicted foreground (%)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_prediction_grid(
    config_path: str | Path,
    *,
    which: str = "best",
    split: str = "val",
    n: int = 16,
    seed: int = 0,
    device: str | None = None,
) -> plt.Figure:
    model, train_config, _ = load_model_from_checkpoint(
        config_path,
        which=which,
        device=device,
    )
    ds = KvasirSegDataset(split, augment=False)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)

    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    axes = axes.reshape(rows, cols)

    for ax, idx in zip(axes.flat, indices):
        image_t, mask_t = ds[int(idx)]
        batch = lucid.stack((image_t,), axis=0).to(model.device)
        with lucid.no_grad():
            outputs = model(batch)
            logits, _ = _extract_logits(outputs)

        logits_np = _to_numpy_float32(logits)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits_np[0, 0], -60.0, 60.0)))
        pred = (probs >= train_config.threshold).astype(np.float32)
        target = _to_numpy_float32(mask_t)[0]
        intersection = float((pred * target).sum())
        dice = (2.0 * intersection + 1e-7) / (float(pred.sum() + target.sum()) + 1e-7)

        image = _denorm_image(_to_numpy_float32(image_t))
        overlay = _blend_mask(image, target, color=(1.0, 0.2, 0.2), alpha=0.20)
        overlay = _blend_mask(overlay, pred, color=(0.15, 0.8, 0.25), alpha=0.35)
        ax.imshow(overlay)
        ax.set_title(f"#{int(idx)}  dice={dice:.3f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes.flat[len(indices) :]:
        ax.axis("off")

    fig.suptitle(
        f"{split} prediction overlays ({which} checkpoint)\nred=GT, green=prediction",
        fontsize=14,
    )
    fig.tight_layout()
    return fig


def _detect_resume_bundle(
    config: TrainConfig,
    config_name: str,
    current_doc: dict[str, Any],
) -> tuple[Path, Path | None, int | None, bool]:
    checkpoint_root = get_checkpoint_root(config_name, output_root=config.output_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    if config.resume_from:
        bundle_dir = Path(config.resume_from).expanduser().resolve()
        if not _has_checkpoint_bundle(bundle_dir):
            raise FileNotFoundError(f"Resume bundle is incomplete: {bundle_dir}")
        resume_root = bundle_dir.parent
        if not _experiment_config_matches(resume_root, current_doc):
            raise ValueError(
                "Explicit resume bundle was created from a different experiment config. "
                "Point --resume-from at a matching checkpoint or clear the old artifacts."
            )
        state = lucid.load(bundle_dir / "training_state.lcd")
        return checkpoint_root, bundle_dir, int(state["epoch"]), False

    latest_dir = checkpoint_root / "latest"
    if _has_checkpoint_bundle(latest_dir):
        if not _experiment_config_matches(checkpoint_root, current_doc):
            return checkpoint_root, None, None, True
        state = lucid.load(latest_dir / "training_state.lcd")
        latest_epoch = int(state["epoch"])
        if latest_epoch < config.epochs:
            return checkpoint_root, latest_dir, latest_epoch, False
        return checkpoint_root, None, latest_epoch, False

    return checkpoint_root, None, None, False


def train(config: TrainConfig, *, config_name: str, config_path: str | Path) -> Path:
    _validate_config(config)
    _seed_everything(config.seed)
    config_path = Path(config_path)
    model_config = load_model_config(config_path)
    current_doc = _experiment_config_doc(config, model_config)

    run_dir, resume_bundle, resume_epoch, ignored_stale_resume = _detect_resume_bundle(
        config,
        config_name,
        current_doc,
    )

    latest_dir = run_dir / "latest"
    best_dir = run_dir / "best"
    plot_dir = ROOT / "out" / config_name

    if resume_bundle is None and resume_epoch is not None and resume_epoch >= config.epochs:
        print(f"Checkpoint already reached epoch {resume_epoch}, target epochs={config.epochs}.")
        print(f"Artifacts already available under: {run_dir}")
        return run_dir

    save_experiment_config(config, model_config, run_dir / "experiment_config.yaml")

    train_loader = get_train_loader(batch_size=config.batch_size)
    val_loader = get_val_loader(batch_size=config.batch_size)

    model = build_model(model_config).to(config.device)
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)

    best_value: float | None = None
    best_epoch: int | None = None
    start_epoch = 1
    history = _empty_history()

    if resume_bundle is not None:
        loaded_epoch, best_epoch, best_value, history = _load_checkpoint_bundle(
            bundle_dir=resume_bundle,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = loaded_epoch + 1
        print(f"Resumed from: {resume_bundle}")
        print(f"Resume epoch: {loaded_epoch}")
    elif ignored_stale_resume:
        print("Existing checkpoint config differs from the current YAML. Starting fresh.")

    print(f"Run directory: {run_dir}")
    print(f"Device: {config.device}")
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
    print(f"Best criterion: {config.monitor} ({config.monitor_mode})")

    for epoch in range(start_epoch, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        print(f"lr={_current_lr(optimizer):.6f}")

        train_metrics, running_loss_trace = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            config=config,
            training=True,
        )
        val_metrics, _ = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            config=config,
            training=False,
        )

        metrics_table = _metric_table(train_metrics, val_metrics)
        _append_running_train_history(history, running_loss_trace)
        _append_epoch_history(history, epoch, _current_lr(optimizer), metrics_table)

        if scheduler is not None:
            if config.scheduler.name.lower() == "plateau":
                scheduler.step(metrics_table[config.scheduler.monitor])
            else:
                scheduler.step()

        print(
            "  summary "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"train_fg_dice={train_metrics['foreground_dice']:.4f} "
            f"valid_loss={val_metrics['loss']:.4f} "
            f"valid_dice={val_metrics['dice']:.4f} "
            f"valid_fg_dice={val_metrics['foreground_dice']:.4f}"
        )

        monitor_value = metrics_table[config.monitor]

        if config.save_best and _is_better(
            monitor_value,
            best_value,
            mode=config.monitor_mode,
        ):
            best_value = monitor_value
            best_epoch = epoch
            _save_checkpoint_bundle(
                bundle_dir=best_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                epoch=epoch,
                best_epoch=best_epoch,
                best_value=best_value,
                monitor=config.monitor,
            )
            print(f"  updated best checkpoint -> {best_dir}")

        _save_checkpoint_bundle(
            bundle_dir=latest_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            epoch=epoch,
            best_epoch=best_epoch,
            best_value=best_value,
            monitor=config.monitor,
        )
        _plot_history(history, out_dir=plot_dir, epoch=epoch)
        print(f"  updated latest checkpoint -> {latest_dir}")

    print("\nTraining complete.")
    if best_epoch is not None:
        print(f"Best {config.monitor}: {best_value:.6f} at epoch {best_epoch}")
    print(f"Artifacts saved under: {run_dir}")
    return run_dir


def train_from_config(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = load_train_config(config_path)
    return train(config, config_name=resolve_config_name(config_path), config_path=config_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="config/attention_unet.yaml",
        help="Path to the YAML training config.",
    )
    parser.add_argument(
        "--save-default-config",
        type=str,
        default=None,
        help="Write a default training config and exit.",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    return parser.parse_args()


def _apply_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.device is not None:
        config.device = args.device
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.resume_from is not None:
        config.resume_from = args.resume_from
    if args.max_train_steps is not None:
        config.max_train_steps_per_epoch = args.max_train_steps
    if args.max_val_steps is not None:
        config.max_val_steps_per_epoch = args.max_val_steps
    return config


def main() -> None:
    args = _parse_args()
    if args.save_default_config is not None:
        save_default_experiment_config(args.save_default_config)
        print(f"Default train config saved -> {args.save_default_config}")
        return

    config_path = Path(args.config)
    config = load_train_config(config_path)
    config = _apply_overrides(config, args)
    train(config, config_name=resolve_config_name(config_path), config_path=config_path)


if __name__ == "__main__":
    main()
