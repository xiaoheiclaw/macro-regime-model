#!/usr/bin/env python3
"""
v2 pipeline orchestrator. Calls each layer in order; each layer is end-to-end
runnable on its own. Stubs (regime, forecast) return placeholder outputs so
downstream (allocation) can be wired before those layers are real.

Layer contract (Layer ABC):
  name: str
  inputs: list of file paths it reads
  outputs: list of file paths it writes
  run(asof) -> dict  # metadata summary

Usage:
  uv run python scripts/v2_pipeline.py                # run all phases
  uv run python scripts/v2_pipeline.py --only mask    # single layer
"""
from __future__ import annotations

import argparse
import json
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.paths import DATA_DIR, ANALYSIS_DIR  # type: ignore

SCHEMA_VERSION = "v2.1"


class Layer(ABC):
    name: str = "layer"

    @property
    @abstractmethod
    def inputs(self) -> list[Path]: ...

    @property
    @abstractmethod
    def outputs(self) -> list[Path]: ...

    @abstractmethod
    def run(self, asof: str | None = None) -> dict: ...

    def check_inputs(self) -> None:
        missing = [p for p in self.inputs if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{self.name}: missing inputs {[str(p) for p in missing]}"
            )


# ── Phase 1: feature mask ────────────────────────────────
class FeatureMaskLayer(Layer):
    name = "feature_mask"

    @property
    def inputs(self) -> list[Path]:
        return [Path(DATA_DIR) / "state_features.parquet"]

    @property
    def outputs(self) -> list[Path]:
        return [
            Path(DATA_DIR) / "v2" / "feature_mask.parquet",
            Path(DATA_DIR) / "v2" / "feature_mask_meta.json",
        ]

    def run(self, asof: str | None = None) -> dict:
        from feature_mask import run as run_mask
        result = run_mask(asof=asof)
        return {"layer": self.name, **result.meta}


# ── Phase 2a: global macro template (real baseline) ─────
class GlobalTemplateLayer(Layer):
    name = "global_template"

    @property
    def inputs(self) -> list[Path]:
        return [Path(DATA_DIR) / "state_features.parquet"]

    @property
    def outputs(self) -> list[Path]:
        out = Path(DATA_DIR) / "v2"
        return [
            out / "global_templates.parquet",
            out / "global_template_centroids.npz",
        ]

    def run(self, asof: str | None = None) -> dict:
        from global_template import run as run_template
        meta = run_template(asof=asof)
        return {"layer": self.name, **meta}


# ── Phase 2b: per-asset regime (stub) ────────────────────
class AssetRegimeLayer(Layer):
    """Stub: uniform K=1 assignment per asset. Replace with per-asset Jump Model."""
    name = "asset_regime"

    @property
    def inputs(self) -> list[Path]:
        return [Path(DATA_DIR) / "state_features.parquet"]

    @property
    def outputs(self) -> list[Path]:
        return [Path(DATA_DIR) / "v2" / "asset_regime_probs.parquet"]

    def run(self, asof: str | None = None) -> dict:
        state = pd.read_parquet(self.inputs[0])
        out_dir = Path(DATA_DIR) / "v2"
        out_dir.mkdir(parents=True, exist_ok=True)

        from feature_mask import ASSET_FEATURES
        assets = [a for a in ASSET_FEATURES if a in state.columns]
        rows = [
            {"date": dt, "asset": asset, "regime_id": 0, "prob": 1.0}
            for asset in assets for dt in state.index
        ]
        pd.DataFrame(rows).to_parquet(out_dir / "asset_regime_probs.parquet", index=False)

        return {
            "layer": self.name,
            "status": "stub",
            "n_assets": len(assets),
            "note": "placeholder — replace with per-asset Jump Model (Shu/Mulvey)",
        }


# ── Phase 3: forecast (stub) ─────────────────────────────
class ForecastLayer(Layer):
    """Stub: emits empty scenario table with correct schema."""
    name = "forecast"

    @property
    def inputs(self) -> list[Path]:
        return [
            Path(DATA_DIR) / "state_features.parquet",
            Path(DATA_DIR) / "v2" / "global_templates.parquet",
            Path(DATA_DIR) / "v2" / "asset_regime_probs.parquet",
        ]

    @property
    def outputs(self) -> list[Path]:
        return [
            Path(DATA_DIR) / "v2" / "forecast_scenarios.parquet",
            Path(DATA_DIR) / "v2" / "forecast_summary.parquet",
        ]

    def run(self, asof: str | None = None) -> dict:
        out_dir = Path(DATA_DIR) / "v2"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Empty scenarios with canonical schema
        scn_schema = pd.DataFrame({
            "asof_date": pd.Series(dtype="datetime64[ns]"),
            "scenario_id": pd.Series(dtype="int64"),
            "asset": pd.Series(dtype="object"),
            "horizon": pd.Series(dtype="int64"),
            "log_return": pd.Series(dtype="float64"),
            "weight": pd.Series(dtype="float64"),
        })
        scn_schema.to_parquet(out_dir / "forecast_scenarios.parquet", index=False)

        sum_schema = pd.DataFrame({
            "asof_date": pd.Series(dtype="datetime64[ns]"),
            "asset": pd.Series(dtype="object"),
            "horizon": pd.Series(dtype="int64"),
            "p10": pd.Series(dtype="float64"),
            "p50": pd.Series(dtype="float64"),
            "p90": pd.Series(dtype="float64"),
            "mean": pd.Series(dtype="float64"),
            "std": pd.Series(dtype="float64"),
        })
        sum_schema.to_parquet(out_dir / "forecast_summary.parquet", index=False)

        return {
            "layer": self.name,
            "status": "stub",
            "n_scenarios": 0,
            "note": "placeholder schema — replace with KAF + tethering in Phase 3",
        }


# ── Runner ───────────────────────────────────────────────
LAYERS: dict[str, type[Layer]] = {
    "mask": FeatureMaskLayer,
    "global_template": GlobalTemplateLayer,
    "asset_regime": AssetRegimeLayer,
    "forecast": ForecastLayer,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(LAYERS), help="run a single layer")
    parser.add_argument("--asof", type=str, default=None)
    args = parser.parse_args()

    layer_names = [args.only] if args.only else list(LAYERS)

    print("=" * 60)
    print(f"v2 pipeline | schema={SCHEMA_VERSION} | asof={args.asof or 'latest'}")
    print("=" * 60)

    run_log = []
    for name in layer_names:
        layer = LAYERS[name]()
        print(f"\n── {layer.name} ──")
        layer.check_inputs()
        t0 = datetime.now()
        meta = layer.run(asof=args.asof)
        dt = (datetime.now() - t0).total_seconds()
        meta["elapsed_sec"] = round(dt, 2)
        run_log.append(meta)
        print(f"  ✓ {layer.name} done in {dt:.1f}s")
        for p in layer.outputs:
            status = "✓" if p.exists() else "✗"
            print(f"    {status} {p}")

    # Persist run log
    log_path = Path(DATA_DIR) / "v2" / "pipeline_runs.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps({
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "schema_version": SCHEMA_VERSION,
            "asof": args.asof,
            "layers": run_log,
        }) + "\n")

    print(f"\n✓ pipeline complete. log → {log_path}")


if __name__ == "__main__":
    main()
