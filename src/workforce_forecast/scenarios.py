"""Output layer: prediction intervals, what-if scenarios and the recruiting gap.

Scenarios work like the "policy-aware scenario injector" in the thesis glossary:
a stakeholder states a possible policy or market change, and the forecast is
shown with and without it. The base forecast never changes.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import poisson


def add_intervals(fc: pd.DataFrame, level: float) -> pd.DataFrame:
    out = fc.copy()
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    out["low"] = poisson.ppf(lo, out["forecast"].clip(lower=1e-6)).astype(int)
    out["high"] = poisson.ppf(hi, out["forecast"].clip(lower=1e-6)).astype(int)
    return out


def apply_scenarios(fc: pd.DataFrame, scenarios: list[dict]) -> pd.DataFrame:
    out = fc.copy()
    for s in scenarios:
        hit = (
            (out["technology"] == s["technology"])
            & (out["country"] == s["country"])
            & out["role_family"].isin(s["role_families"])
            & (out["month"] >= pd.Period(s["start"], "M"))
        )
        out[s["name"]] = out["forecast"].where(~hit, out["forecast"] * (1 + s["change_pct"] / 100.0))
    return out


def recruiting_gap(fc: pd.DataFrame, capacity: dict) -> pd.DataFrame:
    df = fc.copy()
    df["quarter"] = df["month"].map(lambda m: m.asfreq("Q"))
    g = df.groupby(["quarter", "role_family"]).agg(forecast_hires=("forecast", "sum"), months=("month", "nunique")).reset_index()
    g = g[g["months"] == 3].drop(columns="months")
    g["capacity"] = g["role_family"].map(capacity)
    g["forecast_hires"] = g["forecast_hires"].round(0).astype(int)
    g["gap"] = g["forecast_hires"] - g["capacity"]
    g["action"] = g["gap"].map(lambda v: "add recruiting capacity or start earlier" if v > 0 else "within capacity")
    return g
