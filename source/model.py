"""AttentionUNet2d model builder with YAML config persistence.

Usage
-----
    from source.model import load_config, save_config, build_model

    # Build from YAML
    config = load_config("config/attention_unet.yaml")
    model  = build_model(config)

    # Round-trip: save the resolved (fully-expanded) config back to YAML
    save_config(config, "config/attention_unet_resolved.yaml")

Config hierarchy (all dataclasses live in lucid.models)
-------------------------------------------------------
    UNetStageConfig          - per-stage hyperparams (channels, blocks, …)
    AttentionUNetGateConfig  - attention gate hyperparams
    AttentionUNetConfig      - full model config (extends UNetConfig)

YAML keys mirror the dataclass field names exactly.
Optional fields (decoder_stages, bottleneck, bias, stem_channels,
inter_channels, gate_on_skips) accept null → auto-derived by __post_init__.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lucid.models import UNetStageConfig
from lucid.models import (
    AttentionUNetGateConfig,
    AttentionUNetConfig,
    AttentionUNet2d,
)

# ──────────────────────────────────────────────────────────────
# Internal serialization helpers
# ──────────────────────────────────────────────────────────────


def _stage_from_dict(d: dict[str, Any]) -> UNetStageConfig:
    return UNetStageConfig(
        channels=int(d["channels"]),
        num_blocks=int(d.get("num_blocks", 2)),
        kernel_size=int(d.get("kernel_size", 3)),
        dilation=int(d.get("dilation", 1)),
        use_attention=bool(d.get("use_attention", False)),
        dropout=float(d.get("dropout", 0.0)),
    )


def _stage_to_dict(s: UNetStageConfig) -> dict[str, Any]:
    return {
        "channels": s.channels,
        "num_blocks": s.num_blocks,
        "kernel_size": s.kernel_size,
        "dilation": s.dilation,
        "use_attention": s.use_attention,
        "dropout": s.dropout,
    }


def _gate_config_from_dict(d: dict[str, Any]) -> AttentionUNetGateConfig:
    inter = d.get("inter_channels")
    if isinstance(inter, list):
        inter = tuple(inter)

    gate_on_skips = d.get("gate_on_skips")
    if isinstance(gate_on_skips, list):
        gate_on_skips = tuple(gate_on_skips)

    return AttentionUNetGateConfig(
        enabled=bool(d.get("enabled", True)),
        mode=d.get("mode", "additive"),
        gate_activation=d.get("gate_activation", "relu"),
        attention_activation=d.get("attention_activation", "sigmoid"),
        use_grid_attention=bool(d.get("use_grid_attention", True)),
        inter_channels=inter,
        attention_channels=int(d.get("attention_channels", 1)),
        resample_mode=d.get("resample_mode", "bilinear"),
        gate_on_skips=gate_on_skips,
        skip_low_level_gates=bool(d.get("skip_low_level_gates", False)),
        use_multi_scale_gating=bool(d.get("use_multi_scale_gating", True)),
        project_skip_with_1x1=bool(d.get("project_skip_with_1x1", True)),
        project_gating_with_1x1=bool(d.get("project_gating_with_1x1", True)),
        init_pass_through=bool(d.get("init_pass_through", True)),
    )


def _gate_config_to_dict(g: AttentionUNetGateConfig) -> dict[str, Any]:
    inter = g.inter_channels
    if isinstance(inter, tuple):
        inter = list(inter)

    gate_on_skips = g.gate_on_skips
    if isinstance(gate_on_skips, tuple):
        gate_on_skips = list(gate_on_skips)

    return {
        "enabled": g.enabled,
        "mode": g.mode,
        "gate_activation": g.gate_activation,
        "attention_activation": g.attention_activation,
        "use_grid_attention": g.use_grid_attention,
        "inter_channels": inter,
        "attention_channels": g.attention_channels,
        "resample_mode": g.resample_mode,
        "gate_on_skips": gate_on_skips,
        "skip_low_level_gates": g.skip_low_level_gates,
        "use_multi_scale_gating": g.use_multi_scale_gating,
        "project_skip_with_1x1": g.project_skip_with_1x1,
        "project_gating_with_1x1": g.project_gating_with_1x1,
        "init_pass_through": g.init_pass_through,
    }


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────


def load_config(path: str | Path) -> AttentionUNetConfig:
    """Load an `AttentionUNetConfig` from a YAML file.

    Optional YAML keys (`decoder_stages`, `bottleneck`, `bias`,
    `stem_channels`, `inter_channels`, `gate_on_skips`) may be omitted
    or set to `null`; the dataclass `__post_init__` will auto-derive them.

    Args:
        path: Path to the YAML config file (for example `"config/attention_unet.yaml"`).

    Returns:
        A fully validated and resolved `AttentionUNetConfig`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    if "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]

    # ── encoder stages (required) ──────────────────────────────
    encoder_stages = tuple(_stage_from_dict(s) for s in raw["encoder_stages"])

    # ── decoder stages (optional → None triggers auto-derive) ──
    raw_dec = raw.get("decoder_stages")
    decoder_stages = (
        tuple(_stage_from_dict(s) for s in raw_dec) if raw_dec is not None else None
    )

    # ── bottleneck (optional → None triggers auto-derive) ──────
    raw_bn = raw.get("bottleneck")
    bottleneck = _stage_from_dict(raw_bn) if raw_bn is not None else None

    # ── attention gate config ──────────────────────────────────
    attention = _gate_config_from_dict(raw.get("attention", {}))

    # ── scalar fields ──────────────────────────────────────────
    return AttentionUNetConfig(
        in_channels=int(raw["in_channels"]),
        out_channels=int(raw["out_channels"]),
        encoder_stages=encoder_stages,
        decoder_stages=decoder_stages,
        bottleneck=bottleneck,
        norm=raw.get("norm", "batch"),
        act=raw.get("act", "relu"),
        downsample_mode=raw.get("downsample_mode", "maxpool"),
        upsample_mode=raw.get("upsample_mode", "bilinear"),
        deep_supervision=bool(raw.get("deep_supervision", True)),
        align_corners=bool(raw.get("align_corners", False)),
        bias=raw.get("bias"),  # None → auto
        stem_channels=raw.get("stem_channels"),  # None → auto
        final_kernel_size=int(raw.get("final_kernel_size", 1)),
        attention=attention,
    )


def config_to_dict(config: AttentionUNetConfig) -> dict[str, Any]:
    """Serialize a resolved `AttentionUNetConfig` into a plain Python dict."""
    return {
        "in_channels": config.in_channels,
        "out_channels": config.out_channels,
        "norm": config.norm,
        "act": config.act,
        "downsample_mode": config.downsample_mode,
        "upsample_mode": config.upsample_mode,
        "deep_supervision": config.deep_supervision,
        "align_corners": config.align_corners,
        "bias": config.bias,
        "stem_channels": config.stem_channels,
        "final_kernel_size": config.final_kernel_size,
        "encoder_stages": [_stage_to_dict(s) for s in config.encoder_stages],
        "decoder_stages": [_stage_to_dict(s) for s in config.decoder_stages],
        "bottleneck": _stage_to_dict(config.bottleneck),
        "attention": _gate_config_to_dict(config.attention),
    }


def save_config(config: AttentionUNetConfig, path: str | Path) -> None:
    """Serialize a resolved `AttentionUNetConfig` to a YAML file.

    The saved YAML is fully expanded (no `null` auto-derived fields), so it
    can be used as a self-contained reproducibility artifact.

    Args:
        config: The config to serialize (must be a resolved `AttentionUNetConfig`).
        path:   Destination YAML path. Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = config_to_dict(config)

    with path.open("w") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Config saved → {path}")


def build_model(config: AttentionUNetConfig) -> AttentionUNet2d:
    """Instantiate an `AttentionUNet2d` from a resolved config.

    Args:
        config: A resolved `AttentionUNetConfig` (e.g. from `load_config`).

    Returns:
        An `AttentionUNet2d` module with randomly initialized weights.
    """
    if not isinstance(config, AttentionUNetConfig):
        raise TypeError(f"Expected AttentionUNetConfig, got {type(config).__name__}")
    return AttentionUNet2d(config)
