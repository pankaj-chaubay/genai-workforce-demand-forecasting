"""Cost impact of forecast errors.

The thesis estimated savings from a lower cost per hire, using assumed figures.
This module works it out from the backtest instead. Every role a model misses
in a quarter has to be filled at short notice (urgent premium). Every role it
over-forecasts is hired without need (over-hire cost). The money figures are
assumptions in config/default.yaml.
"""

from __future__ import annotations

import pandas as pd

from .evaluation import quarterly


def cost_impact(bt: pd.DataFrame, params: dict, baseline: str = "seasonal_naive") -> pd.DataFrame:
    q = quarterly(bt)
    q["under"] = (q["actual"] - q["forecast"]).clip(lower=0)
    q["over"] = (q["forecast"] - q["actual"]).clip(lower=0)
    g = q.groupby("model").agg(missed_roles=("under", "sum"), extra_roles=("over", "sum"), hires=("actual", "sum")).reset_index()
    g["error_cost_eur"] = g["missed_roles"] * params["urgent_hire_premium"] + g["extra_roles"] * params["overhire_cost"]
    g["hiring_cost_eur"] = g["hires"] * params["cost_per_hire"] + g["error_cost_eur"]
    base = float(g.loc[g["model"] == baseline, "hiring_cost_eur"].iloc[0])
    g["saving_vs_baseline_pct"] = (100 * (base - g["hiring_cost_eur"]) / base).round(1)
    g["cost_per_hire_eur"] = (g["hiring_cost_eur"] / g["hires"]).round(0)
    for c in ["missed_roles", "extra_roles"]:
        g[c] = g[c].round(0).astype(int)
    for c in ["error_cost_eur", "hiring_cost_eur"]:
        g[c] = g[c].round(-2).astype(int)
    return g.sort_values("hiring_cost_eur").reset_index(drop=True)
