"""Backtesting and accuracy metrics.

Rolling-origin backtest: for each origin the models are trained only on months
before it and forecast the next 12 months. Accuracy is reported as
1 - WAPE (weighted absolute percentage error), the share of hires forecast
correctly, and as MAE in roles per segment and quarter, the unit HR plans in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import Context


def backtest(model_classes, ctx: Context, origins: list[pd.Period], horizon: int) -> pd.DataFrame:
    frames = []
    for first_forecast_month in origins:
        o = ctx.month_index(first_forecast_month) - 1  # last observed month
        seen = ctx.masked(ctx.months[o])
        for cls in model_classes:
            model = cls().fit(seen, o)
            fc = model.predict(seen, o, horizon)
            fc["model"] = cls.name
            frames.append(fc)
    out = pd.concat(frames, ignore_index=True)
    actual = {(s, m): ctx.y[i, ctx.month_index(m)] for i, s in enumerate(ctx.segments) for m in ctx.months}
    out["actual"] = [actual[((r.technology, r.country, r.role_family), r.month)] for r in out.itertuples(index=False)]
    return out.dropna(subset=["actual"])


def wape(actual, forecast) -> float:
    a = np.asarray(actual, float)
    return float(np.abs(np.asarray(forecast, float) - a).sum() / max(a.sum(), 1e-9))


def horizon_bucket(h: int, buckets: dict) -> str:
    for name, (lo, hi) in buckets.items():
        if lo <= h <= hi:
            return name
    return "other"


def accuracy_table(bt: pd.DataFrame, buckets: dict, by: list[str] | None = None) -> pd.DataFrame:
    df = bt.copy()
    df["horizon_bucket"] = df["horizon"].map(lambda h: horizon_bucket(h, buckets))
    groups = ["model"] + (by or []) + ["horizon_bucket"]
    rows = []
    for key, g in df.groupby(groups):
        rec = dict(zip(groups, key if isinstance(key, tuple) else (key,)))
        rec["accuracy_pct"] = round(100 * (1 - wape(g["actual"], g["forecast"])), 1)
        rec["mae_monthly"] = round(float(np.abs(g["forecast"] - g["actual"]).mean()), 2)
        rows.append(rec)
    return pd.DataFrame(rows)


def quarterly(bt: pd.DataFrame) -> pd.DataFrame:
    """Sum forecasts and actuals to segment-quarter level within each origin."""
    df = bt.copy()
    df["quarter"] = df["month"].map(lambda m: m.asfreq("Q"))
    keys = ["model", "origin", "technology", "country", "role_family", "quarter"]
    q = df.groupby(keys).agg(forecast=("forecast", "sum"), actual=("actual", "sum"), months=("month", "nunique")).reset_index()
    return q[q["months"] == 3].drop(columns="months")


def summary(bt: pd.DataFrame) -> pd.DataFrame:
    q = quarterly(bt)
    # division = technology x country, all role families together
    div = q.groupby(["model", "origin", "technology", "country", "quarter"])[["forecast", "actual"]].sum().reset_index()
    rows = []
    for model, g in bt.groupby("model"):
        gq = q[q["model"] == model]
        gd = div[div["model"] == model]
        rows.append(
            {
                "model": model,
                "accuracy_pct": round(100 * (1 - wape(g["actual"], g["forecast"])), 1),
                "quarterly_accuracy_pct": round(100 * (1 - wape(gq["actual"], gq["forecast"])), 1),
                "division_quarterly_accuracy_pct": round(100 * (1 - wape(gd["actual"], gd["forecast"])), 1),
                "mae_roles_per_quarter": round(float(np.abs(gq["forecast"] - gq["actual"]).mean()), 2),
                "bias_pct": round(100 * (g["forecast"].sum() - g["actual"].sum()) / g["actual"].sum(), 1),
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy_pct", ascending=False).reset_index(drop=True)
