"""Processing layer: the forecasting models compared in the framework.

* SeasonalNaive    - "same month last year", close to how manual planning often works.
* PoissonRate      - inhomogeneous Poisson rate model (segment, season, trend), after Shen & Huang (2008).
* AdditiveSeasonal - trend with changepoints plus Fourier seasonality per segment. This follows the
                     idea behind Greykite (Hosseini et al., 2022). It is not the Greykite library.
* TemporalML       - gradient-boosted direct multi-horizon model on hiring history only.
* Hybrid           - TemporalML plus the stakeholder inputs (project pipeline, HR seasonality and
                     one-off periods, announced policy changes).

The thesis describes a transformer-based temporal model. With 16 segments and a few years of
monthly data, a gradient-boosted model with a Poisson loss is the more honest choice: it trains
in seconds, does not overfit as easily, and SHAP explains it exactly.

Every model gets a Context in which actuals after the origin are already hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor, Ridge

from .features import HISTORY_FEATURES, SEGMENT_FEATURES, STAKEHOLDER_FEATURES, Context, make_rows


def _output(ctx: Context, si: int, o: int, horizon: int, values) -> list[dict]:
    tech, country, role = ctx.segments[si]
    rows = []
    for h in range(1, horizon + 1):
        if o + h >= len(ctx.months):
            break
        rows.append(
            {
                "technology": tech,
                "country": country,
                "role_family": role,
                "origin": ctx.months[o],
                "month": ctx.months[o + h],
                "horizon": h,
                "forecast": max(float(values[h - 1]), 0.0),
            }
        )
    return rows


class SeasonalNaive:
    name = "seasonal_naive"
    label = "Same month last year (manual baseline)"

    def fit(self, ctx: Context, origin: int) -> "SeasonalNaive":
        return self

    def predict(self, ctx: Context, origin: int, horizon: int) -> pd.DataFrame:
        rows = []
        for si in range(len(ctx.segments)):
            vals = [ctx.y[si, origin + h - 12] for h in range(1, horizon + 1)]
            rows += _output(ctx, si, origin, horizon, vals)
        return pd.DataFrame(rows)


class PoissonRate:
    name = "poisson_rate"
    label = "Inhomogeneous Poisson rate model"

    def _design(self, ctx: Context, si: int, t: np.ndarray) -> np.ndarray:
        n_seg = len(ctx.segments)
        seg = np.zeros((len(t), n_seg))
        seg[:, si] = 1
        month = np.zeros((len(t), 12))
        month[np.arange(len(t)), [ctx.months[i].month - 1 for i in t]] = 1
        trend = (t / 12.0)[:, None]
        # a separate trend per segment, so growing and shrinking segments do not pull on each other
        seg_trend = seg * trend
        return np.hstack([seg, month, seg_trend])

    def fit(self, ctx: Context, origin: int) -> "PoissonRate":
        start = max(origin - 35, 0)  # last three years
        X, y = [], []
        for si in range(len(ctx.segments)):
            t = np.arange(start, origin + 1)
            X.append(self._design(ctx, si, t))
            y.append(ctx.y[si, t])
        self.model = PoissonRegressor(alpha=1e-2, max_iter=3000).fit(np.vstack(X), np.concatenate(y))
        return self

    def predict(self, ctx: Context, origin: int, horizon: int) -> pd.DataFrame:
        rows = []
        for si in range(len(ctx.segments)):
            t = np.arange(origin + 1, min(origin + horizon + 1, len(ctx.months)))
            rows += _output(ctx, si, origin, horizon, self.model.predict(self._design(ctx, si, t)))
        return pd.DataFrame(rows)


class AdditiveSeasonal:
    name = "additive_seasonal"
    label = "Trend + seasonality, Greykite-style"

    @staticmethod
    def _design(t: np.ndarray, changepoints: np.ndarray, months: np.ndarray) -> np.ndarray:
        cols = [t / 12.0]
        cols += [np.maximum(0, t - c) / 12.0 for c in changepoints]
        for k in (1, 2):
            cols += [np.sin(2 * np.pi * k * months / 12), np.cos(2 * np.pi * k * months / 12)]
        return np.column_stack(cols)

    def fit(self, ctx: Context, origin: int) -> "AdditiveSeasonal":
        self.models = []
        t = np.arange(0, origin + 1)
        # changepoints every 6 months, none in the last 6 months so the end of the trend stays stable
        self.changepoints = np.arange(6, max(origin - 5, 7), 6)
        m = np.array([ctx.months[i].month for i in t])
        X = self._design(t, self.changepoints, m)
        for si in range(len(ctx.segments)):
            self.models.append(Ridge(alpha=3.0).fit(X, ctx.y[si, t]))
        return self

    def predict(self, ctx: Context, origin: int, horizon: int) -> pd.DataFrame:
        rows = []
        t = np.arange(origin + 1, min(origin + horizon + 1, len(ctx.months)))
        m = np.array([ctx.months[i].month for i in t])
        X = self._design(t, self.changepoints, m)
        for si, model in enumerate(self.models):
            rows += _output(ctx, si, origin, horizon, model.predict(X))
        return pd.DataFrame(rows)


class TemporalML:
    name = "temporal_ml"
    label = "Temporal ML, hiring history only"
    use_stakeholder = False

    def __init__(self, horizon: int = 12, random_state: int = 0):
        self.horizon = horizon
        self.random_state = random_state

    @property
    def feature_names(self) -> list[str]:
        return HISTORY_FEATURES + SEGMENT_FEATURES + (STAKEHOLDER_FEATURES if self.use_stakeholder else [])

    def fit(self, ctx: Context, origin: int) -> "TemporalML":
        pairs = [(o, h) for h in range(1, self.horizon + 1) for o in range(0, origin - h + 1)]
        rows = make_rows(ctx, pairs).dropna(subset=["actual"])
        self.train_rows = rows
        cats = [f in SEGMENT_FEATURES or f == "target_month" for f in self.feature_names]
        self.model = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            categorical_features=cats,
            random_state=self.random_state,
        ).fit(rows[self.feature_names], rows["actual"])
        return self

    def predict(self, ctx: Context, origin: int, horizon: int) -> pd.DataFrame:
        rows = make_rows(ctx, [(origin, h) for h in range(1, horizon + 1)])
        out = rows[["technology", "country", "role_family", "origin", "month", "horizon"]].copy()
        out["forecast"] = np.clip(self.model.predict(rows[self.feature_names]), 0, None)
        return out

    def feature_frame(self, ctx: Context, origin: int, horizon: int) -> pd.DataFrame:
        return make_rows(ctx, [(origin, h) for h in range(1, horizon + 1)])[self.feature_names]


class StakeholderTemporal(TemporalML):
    """Temporal ML that also sees the stakeholder inputs.

    Announced policy changes are used as an offset, not as a learned feature: each policy event
    is new, so a tree cannot learn its effect from history. The expected change from the policy
    team scales the forecast directly, and training targets are divided by the same factor.
    """

    name = "stakeholder_temporal"
    label = "Temporal ML + stakeholder inputs"
    use_stakeholder = True

    @property
    def feature_names(self) -> list[str]:
        return [f for f in super().feature_names if f != "policy_expected_change"]

    def fit(self, ctx: Context, origin: int) -> "StakeholderTemporal":
        pairs = [(o, h) for h in range(1, self.horizon + 1) for o in range(0, origin - h + 1)]
        rows = make_rows(ctx, pairs).dropna(subset=["actual"])
        factor = 1 + rows["policy_expected_change"]
        cats = [f in SEGMENT_FEATURES or f == "target_month" for f in self.feature_names]
        self.model = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            categorical_features=cats,
            random_state=self.random_state,
        ).fit(rows[self.feature_names], rows["actual"] / factor, sample_weight=factor)
        return self

    def predict(self, ctx: Context, origin: int, horizon: int) -> pd.DataFrame:
        rows = make_rows(ctx, [(origin, h) for h in range(1, horizon + 1)])
        out = rows[["technology", "country", "role_family", "origin", "month", "horizon"]].copy()
        out["forecast"] = np.clip(self.model.predict(rows[self.feature_names]), 0, None) * (1 + rows["policy_expected_change"].to_numpy())
        return out


class StakeholderPoisson:
    """Poisson rate model driven by the stakeholder inputs.

    Hires per segment grow with the project pipeline (log scale, own slope per segment), rise in
    the HR peak season and during one-off periods, and move with announced policy (offset).
    Unlike trees, this model can extrapolate when the pipeline grows past anything seen before.
    """

    name = "stakeholder_poisson"
    label = "Poisson rate on pipeline + stakeholder inputs"

    def _design(self, ctx: Context, si: int, t: np.ndarray) -> np.ndarray:
        n_seg = len(ctx.segments)
        roles = sorted({s[2] for s in ctx.segments})
        r = roles.index(ctx.segments[si][2])
        seg = np.zeros((len(t), n_seg))
        seg[:, si] = 1
        driver = np.log1p(ctx.known["pipeline_driver"][si, t])[:, None]
        month = np.zeros((len(t), 12))
        month[np.arange(len(t)), [ctx.months[i].month - 1 for i in t]] = 1
        season = np.zeros((len(t), len(roles)))
        season[:, r] = ctx.known["hr_peak_season"][si, t]
        one_off = np.zeros((len(t), len(roles)))
        one_off[:, r] = ctx.known["hr_one_off_period"][si, t]
        return np.hstack([seg, seg * driver, month, season, one_off])

    def fit(self, ctx: Context, origin: int) -> "StakeholderPoisson":
        X, y, w = [], [], []
        t = np.arange(0, origin + 1)
        for si in range(len(ctx.segments)):
            factor = 1 + ctx.known["policy_expected_change"][si, t]
            X.append(self._design(ctx, si, t))
            y.append(ctx.y[si, t] / factor)
            w.append(factor)
        self.model = PoissonRegressor(alpha=1e-3, max_iter=5000).fit(np.vstack(X), np.concatenate(y), sample_weight=np.concatenate(w))
        return self

    def predict(self, ctx: Context, origin: int, horizon: int) -> pd.DataFrame:
        rows = []
        for si in range(len(ctx.segments)):
            t = np.arange(origin + 1, min(origin + horizon + 1, len(ctx.months)))
            factor = 1 + ctx.known["policy_expected_change"][si, t]
            rows += _output(ctx, si, origin, horizon, self.model.predict(self._design(ctx, si, t)) * factor)
        return pd.DataFrame(rows)


class Hybrid:
    """The framework model: average of the stakeholder-informed temporal ML and Poisson rate models.

    The thesis proposes combining temporal learning with a Poisson process and expert input.
    The two parts fail in different ways (trees cannot extrapolate, the GLM cannot pick up
    patterns it was not given), so the average is steadier than either one.
    """

    name = "hybrid"
    label = "Hybrid: temporal ML + Poisson + stakeholder inputs (framework)"

    def __init__(self, horizon: int = 12, random_state: int = 0):
        self.temporal = StakeholderTemporal(horizon, random_state)
        self.poisson = StakeholderPoisson()

    def fit(self, ctx: Context, origin: int) -> "Hybrid":
        self.temporal.fit(ctx, origin)
        self.poisson.fit(ctx, origin)
        return self

    def predict(self, ctx: Context, origin: int, horizon: int) -> pd.DataFrame:
        a = self.temporal.predict(ctx, origin, horizon)
        b = self.poisson.predict(ctx, origin, horizon)
        keys = ["technology", "country", "role_family", "origin", "month", "horizon"]
        m = a.merge(b, on=keys, suffixes=("_t", "_p"))
        m["forecast"] = (m["forecast_t"] + m["forecast_p"]) / 2
        return m[keys + ["forecast"]]


ALL_MODELS = [SeasonalNaive, PoissonRate, AdditiveSeasonal, TemporalML, StakeholderTemporal, StakeholderPoisson, Hybrid]
