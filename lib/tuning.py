"""Load tunable parameters with safe defaults."""
import json
import os
from lib.paths import TUNING_PARAMS

DEFAULTS = {
    "omega_scale": 1.0,
    "ensemble_weights": {"markov": 0.6, "wkm": 0.4},
    "correlation_thresholds": {
        "oil_bond_decouple": -0.2,
        "oil_bond_weak": 0.1,
        "spx_bond_riskoff": -0.3,
        "gold_btc_linked": 0.4,
    },
}


def load_tuning_params() -> dict:
    """Load tuning_params.json, falling back to defaults for missing keys."""
    if os.path.exists(TUNING_PARAMS):
        with open(TUNING_PARAMS) as f:
            params = json.load(f)
        for k, v in DEFAULTS.items():
            if k not in params:
                params[k] = v
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if sub_k not in params[k]:
                        params[k][sub_k] = sub_v
        return params
    return dict(DEFAULTS)
