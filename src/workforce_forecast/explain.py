"""Explainability. In the thesis most HR stakeholders found the model output hard
to follow, and SHAP was the proposed fix. Here SHAP values are computed for the
hybrid model and turned into short sentences an HR partner can read.

The model uses a Poisson (log) link, so SHAP values are on the log scale.
exp(value) - 1 gives the approximate percentage effect on the forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

READABLE = {
    "horizon": "months ahead",
    "lag_0": "hires last month",
    "lag_1": "hires two months ago",
    "lag_2": "hires three months ago",
    "mean_3": "average hires, last 3 months",
    "mean_6": "average hires, last 6 months",
    "mean_12": "average hires, last 12 months",
    "same_month_last_year": "hires in the same month last year",
    "target_month": "calendar month",
    "technology_code": "technology",
    "country_code": "country",
    "role_code": "role family",
    "pipeline_driver": "project pipeline in the target month",
    "pipeline_driver_at_origin": "project pipeline today",
    "hr_peak_season": "HR peak season",
    "hr_one_off_period": "one-off hiring period (target month)",
    "hr_one_off_at_origin": "one-off hiring period (today)",
    "policy_expected_change": "announced policy change",
}


def shap_values(model, X: pd.DataFrame) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model.model)
    vals = explainer.shap_values(X)
    return pd.DataFrame(vals, columns=X.columns, index=X.index)


def global_importance(sv: pd.DataFrame) -> pd.DataFrame:
    imp = sv.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({"feature": imp.index, "driver": [READABLE.get(f, f) for f in imp.index], "mean_abs_shap": imp.round(4).values})


def explain_row(sv_row: pd.Series, top: int = 3) -> str:
    parts = []
    for f in sv_row.abs().sort_values(ascending=False).index[:top]:
        pct = (np.exp(sv_row[f]) - 1) * 100
        word = "raises" if pct >= 0 else "lowers"
        parts.append(f"{READABLE.get(f, f)} {word} it by about {abs(pct):.0f}%")
    return "; ".join(parts) + "."
