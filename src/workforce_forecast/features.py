"""Input layer: turn hires, project pipeline and stakeholder inputs into model features.

The temporal model is a direct multi-horizon model: one row per segment,
forecast origin and horizon h (1-12 months). Features from the past stop at the
origin. Only inputs that are known in advance (calendar, planned projects,
announced policy, HR seasonality) are taken at the target month.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import stakeholder, synthetic

KEYS = ["technology", "country", "role_family"]
HISTORY_FEATURES = ["horizon", "lag_0", "lag_1", "lag_2", "mean_3", "mean_6", "mean_12", "same_month_last_year", "target_month"]
SEGMENT_FEATURES = ["technology_code", "country_code", "role_code"]
STAKEHOLDER_FEATURES = [
    "pipeline_driver",
    "pipeline_driver_at_origin",
    "hr_peak_season",
    "hr_one_off_period",
    "hr_one_off_at_origin",
    "policy_expected_change",
]
MIN_HISTORY = 12

ROLE_DRIVER = {
    "engineer": lambda d: d["projects_in_development"] + d["projects_in_construction"],
    "technician": lambda d: d["mw_in_construction"],
    "project_manager": lambda d: d["construction_starts_next_3m"],
    "admin": lambda d: d["projects_in_development"] + d["projects_in_construction"],
}


@dataclass
class Context:
    """Everything the models may see, on one monthly calendar."""

    months: pd.PeriodIndex
    segments: list[tuple[str, str, str]]
    y: np.ndarray        # hires, shape (segments, months); NaN where unknown
    known: dict[str, np.ndarray]  # known-in-advance features, same shape

    def month_index(self, m: pd.Period) -> int:
        return int(m.ordinal - self.months[0].ordinal)

    def masked(self, last_observed: pd.Period) -> "Context":
        """Copy with every actual after last_observed hidden. Guards against leakage in backtests."""
        y = self.y.copy()
        y[:, self.month_index(last_observed) + 1 :] = np.nan
        return Context(self.months, self.segments, y, self.known)


def build_context(panel: pd.DataFrame, projects: pd.DataFrame, inputs: stakeholder.StakeholderInputs, end: pd.Period) -> Context:
    start = panel["month"].min()
    months = pd.period_range(start, end, freq="M")
    segments = sorted({tuple(r) for r in panel[KEYS].itertuples(index=False)})
    y = np.full((len(segments), len(months)), np.nan)
    pivot = panel.set_index(KEYS + ["month"])["hires"].sort_index()
    for i, seg in enumerate(segments):
        s = pivot.loc[seg]
        idx = [int(m.ordinal - months[0].ordinal) for m in s.index]
        y[i, idx] = s.to_numpy(dtype=float)

    drivers = synthetic.pipeline_drivers(projects, months).set_index(["technology", "country", "month"])
    grid = pd.DataFrame([(*seg, m) for seg in segments for m in months], columns=KEYS + ["month"])
    sh = stakeholder.features(inputs, grid)
    known = {k: np.zeros_like(y) for k in ["pipeline_driver", "hr_peak_season", "hr_one_off_period", "policy_expected_change"]}
    for i, (tech, country, role) in enumerate(segments):
        d = drivers.loc[(tech, country)].reindex(months)
        known["pipeline_driver"][i] = ROLE_DRIVER[role](d).to_numpy(dtype=float)
        rows = sh[(sh.technology == tech) & (sh.country == country) & (sh.role_family == role)]
        for k in ["hr_peak_season", "hr_one_off_period", "policy_expected_change"]:
            known[k][i] = rows[k].to_numpy()
    return Context(months, segments, y, known)


def make_rows(ctx: Context, pairs: list[tuple[int, int]]) -> pd.DataFrame:
    """Feature rows for (origin index, horizon) pairs. The origin is the last observed month."""
    techs = sorted({s[0] for s in ctx.segments})
    countries = sorted({s[1] for s in ctx.segments})
    roles = sorted({s[2] for s in ctx.segments})
    out = []
    for si, (tech, country, role) in enumerate(ctx.segments):
        y = ctx.y[si]
        for o, h in pairs:
            t = o + h
            if o < MIN_HISTORY - 1 or t >= len(ctx.months):
                continue
            hist = y[: o + 1]
            out.append(
                {
                    "technology": tech,
                    "country": country,
                    "role_family": role,
                    "origin": ctx.months[o],
                    "month": ctx.months[t],
                    "horizon": h,
                    "lag_0": hist[o],
                    "lag_1": hist[o - 1],
                    "lag_2": hist[o - 2],
                    "mean_3": np.nanmean(hist[o - 2 : o + 1]),
                    "mean_6": np.nanmean(hist[o - 5 : o + 1]),
                    "mean_12": np.nanmean(hist[o - 11 : o + 1]),
                    "same_month_last_year": y[t - 12],
                    "target_month": ctx.months[t].month,
                    "technology_code": techs.index(tech),
                    "country_code": countries.index(country),
                    "role_code": roles.index(role),
                    "pipeline_driver": ctx.known["pipeline_driver"][si, t],
                    "pipeline_driver_at_origin": ctx.known["pipeline_driver"][si, o],
                    "hr_peak_season": ctx.known["hr_peak_season"][si, t],
                    "hr_one_off_period": ctx.known["hr_one_off_period"][si, t],
                    "hr_one_off_at_origin": ctx.known["hr_one_off_period"][si, o],
                    "policy_expected_change": ctx.known["policy_expected_change"][si, t],
                    "actual": y[t],
                }
            )
    return pd.DataFrame(out)
