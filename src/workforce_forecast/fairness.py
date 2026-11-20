"""Bias audits. Two different questions, kept apart on purpose.

1. Forecast bias: does a model keep over- or under-forecasting a role family?
   The thesis found technical roles over-forecast because the history held a
   hiring wave that was not going to repeat. A systematic error like this
   pushes recruiting budgets and job adverts in the wrong direction.

2. Representation: who was hired in the past. The forecast is built at segment
   level and never sees gender, so it cannot score individuals. But if the
   history is skewed, a plan copied from it carries the skew forward. This
   audit uses aggregated counts only, and suppresses small groups.
"""

from __future__ import annotations

import pandas as pd

from .privacy import suppress_small_cells


def forecast_bias(bt: pd.DataFrame, threshold_pct: float) -> pd.DataFrame:
    g = bt.groupby(["model", "role_family"]).agg(forecast=("forecast", "sum"), actual=("actual", "sum")).reset_index()
    g["bias_pct"] = (100 * (g["forecast"] - g["actual"]) / g["actual"]).round(1)
    g["flag"] = g["bias_pct"].abs() > threshold_pct
    return g[["model", "role_family", "bias_pct", "flag"]]


def representation(audit_records: pd.DataFrame, targets: dict, min_cell: int) -> pd.DataFrame:
    df = audit_records.copy()
    df["year"] = df["hire_month"].map(lambda m: m.year)
    g = df.groupby(["role_family", "year"]).agg(hires=("person_key", "count"), women=("gender", lambda s: int((s == "F").sum())))
    g = g.reset_index()
    g["women_share"] = (g["women"] / g["hires"]).round(3)
    g["target"] = g["role_family"].map(targets)
    g["gap_to_target"] = (g["women_share"] - g["target"]).round(3)
    return suppress_small_cells(g, "hires", ["women", "women_share", "gap_to_target"], min_cell)
