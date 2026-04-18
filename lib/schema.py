"""
Shared schema constants + metadata builder for v2 pipeline.

Each layer's meta/output should include the standard fields from `base_meta()`
so Stage B validation and replay can match versions strictly. Version-bump
protocol:
  - SCHEMA_VERSION   — bump on breaking output format changes (column rename,
                       type change, semantic redefinition)
  - FEATURE_SET_VERSION — bump on addition/removal/transform change of any
                       state-panel feature. Compare by string equality.
  - Individual layer meta may add model_version for per-model training
                       configuration (e.g. K, random seed) changes.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

SCHEMA_VERSION = "v2.1"
FEATURE_SET_VERSION = "2026-04-18-a"   # bump whenever FEATURE_SPEC/derived/vintage change
RETURN_TYPE = "log_total_return"       # applies to all *_ret features
CURRENCY = "USD"
PANEL_FREQUENCY = "monthly_end"


def base_meta(
    *,
    layer: str,
    data_asof: str | None = None,
    model_version: str | None = None,
    extra: dict | None = None,
) -> dict:
    """
    Standard meta block for every v2 layer output. Caller fills in `layer`
    and optionally `data_asof` (max date in input) / `model_version` (per-layer
    training config hash or label).
    """
    m = {
        "schema_version": SCHEMA_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "panel_frequency": PANEL_FREQUENCY,
        "return_type": RETURN_TYPE,
        "currency": CURRENCY,
        "layer": layer,
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    if data_asof is not None:
        m["data_asof"] = data_asof
    if model_version is not None:
        m["model_version"] = model_version
    if extra:
        m.update(extra)
    return m
