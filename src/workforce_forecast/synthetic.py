"""Synthetic data for a mid-sized renewable energy developer.

Nothing here comes from a real company. The generator builds three things:

* a project pipeline (solar and wind parks in two countries),
* an HR hiring extract at employee level, shaped like a typical HRIS export,
  including fields that must never reach the model (email, gender),
* hidden "true" policy effects that the stakeholder inputs only approximate.

The hiring process is driven by the project pipeline, seasonality, policy
changes and a one-off backlog after COVID. That gives the models the same kind
of problems the thesis describes: small data, project spikes and policy shocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GEN_START = pd.Period("2021-01", "M")  # one extra year so the retention and date filters have work to do
HIST_END = pd.Period("2025-06", "M")   # last month with hiring actuals
PIPELINE_END = pd.Period("2026-09", "M")  # planned projects reach past the forecast horizon

TECHS = ("solar", "wind")
COUNTRIES = ("DE", "ES")
ROLES = ("engineer", "technician", "project_manager", "admin")

# Project starts per month before any policy effect.
BASE_START_RATE = {("solar", "DE"): 0.9, ("solar", "ES"): 0.7, ("wind", "DE"): 0.35, ("wind", "ES"): 0.45}

# Hidden policy effects on the market. The stakeholder file holds rough estimates of these.
POLICY_ON_STARTS = [
    # (technology, country, first month, last month or None, multiplier on project starts)
    ("solar", "DE", "2023-01", "2024-12", 1.7),
    ("solar", "ES", "2025-01", None, 0.6),
]
POLICY_ON_HIRING = [
    # (technology, country, roles, first month, last month or None, multiplier on hiring)
    ("solar", "DE", ("engineer", "technician"), "2023-04", "2024-09", 1.3),
    ("wind", "ES", ("admin",), "2024-07", None, 1.4),
    ("solar", "ES", ("engineer", "technician", "project_manager"), "2025-03", None, 0.8),
]
BACKLOG = (("engineer", "technician"), "2022-01", "2023-06", 1.55)

SEASONALITY = {
    ("solar", "technician"): {3: 1.45, 4: 1.6, 5: 1.5, 6: 1.3, 11: 0.7, 12: 0.6, 1: 0.7},
    ("wind", "technician"): {4: 1.3, 5: 1.45, 6: 1.4, 7: 1.25, 12: 0.7, 1: 0.7},
    ("solar", "engineer"): {1: 1.15, 9: 1.1},
    ("wind", "engineer"): {1: 1.1, 9: 1.1},
}
SEASONAL_DEFAULT = {8: 0.75, 12: 0.8}  # summer and year-end slowdown for every role

FEMALE_SHARE = {"engineer": 0.19, "technician": 0.07, "project_manager": 0.33, "admin": 0.64}
TEMP_SHARE = {"solar": 0.25, "wind": 0.12}
ANNUAL_ATTRITION = 0.13


@dataclass
class SyntheticData:
    projects: pd.DataFrame
    employees: pd.DataFrame


def _in_window(month: pd.Period, start: str, end: str | None) -> bool:
    if month < pd.Period(start, "M"):
        return False
    return end is None or month <= pd.Period(end, "M")


def generate_projects(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    months = pd.period_range(GEN_START - 12, PIPELINE_END, freq="M")
    pid = 0
    for (tech, country), base in BASE_START_RATE.items():
        for m in months:
            rate = base
            for t, c, start, end, mult in POLICY_ON_STARTS:
                if t == tech and c == country and _in_window(m, start, end):
                    rate *= mult
            for _ in range(rng.poisson(rate)):
                pid += 1
                if tech == "solar":
                    mw = float(np.round(rng.uniform(5, 60), 1))
                    duration = int(rng.integers(4, 9))
                else:
                    mw = float(np.round(rng.uniform(20, 150), 1))
                    duration = int(rng.integers(10, 19))
                rows.append(
                    {
                        "project_id": f"P{pid:04d}",
                        "technology": tech,
                        "country": country,
                        "development_start": str(m - 6),
                        "construction_start": str(m),
                        "construction_months": duration,
                        "capacity_mw": mw,
                    }
                )
    return pd.DataFrame(rows)


def pipeline_drivers(projects: pd.DataFrame, months: pd.PeriodIndex) -> pd.DataFrame:
    """Monthly project activity per technology and country.

    Used by the generator and, through the engineering input, by the hybrid model.
    """
    p = projects.copy()
    p["dev"] = p["development_start"].map(lambda s: pd.Period(s, "M"))
    p["con"] = p["construction_start"].map(lambda s: pd.Period(s, "M"))
    p["con_end"] = p.apply(lambda r: r["con"] + (int(r["construction_months"]) - 1), axis=1)
    rows = []
    for (tech, country), g in p.groupby(["technology", "country"]):
        for m in months:
            in_dev = (g["dev"] <= m) & (g["con"] > m)
            in_con = (g["con"] <= m) & (g["con_end"] >= m)
            starting_soon = (g["con"] > m) & (g["con"] <= m + 3)
            rows.append(
                {
                    "technology": tech,
                    "country": country,
                    "month": m,
                    "projects_in_development": int(in_dev.sum()),
                    "projects_in_construction": int(in_con.sum()),
                    "mw_in_construction": float(g.loc[in_con, "capacity_mw"].sum()),
                    "construction_starts_next_3m": int(starting_soon.sum()),
                }
            )
    return pd.DataFrame(rows)


def _hiring_rate(tech: str, country: str, role: str, m: pd.Period, d: pd.Series) -> float:
    if role == "engineer":
        rate = 1.0 + 0.35 * (d["projects_in_development"] + d["projects_in_construction"])
    elif role == "technician":
        per_mw = 0.03 if tech == "solar" else 0.012  # wind parks need fewer field staff per MW
        rate = 1.0 + per_mw * d["mw_in_construction"]
    elif role == "project_manager":
        rate = 0.4 + 0.45 * d["construction_starts_next_3m"]
    else:
        rate = 0.8 + 0.12 * (d["projects_in_development"] + d["projects_in_construction"])
    if country == "ES":
        rate *= 0.9
    season = SEASONALITY.get((tech, role), {})
    rate *= season.get(m.month, SEASONAL_DEFAULT.get(m.month, 1.0))
    for t, c, roles, start, end, mult in POLICY_ON_HIRING:
        if t == tech and c == country and role in roles and _in_window(m, start, end):
            rate *= mult
    roles_b, start_b, end_b, mult_b = BACKLOG
    if role in roles_b and _in_window(m, start_b, end_b):
        # backlog fades out over its window
        span = pd.Period(end_b, "M").ordinal - pd.Period(start_b, "M").ordinal
        pos = (m.ordinal - pd.Period(start_b, "M").ordinal) / max(span, 1)
        rate *= mult_b - (mult_b - 1.0) * 0.5 * pos
    return rate


def generate_employees(projects: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    months = pd.period_range(GEN_START, HIST_END, freq="M")
    drivers = pipeline_drivers(projects, months).set_index(["technology", "country", "month"])
    rows = []
    n = 0
    for tech in TECHS:
        for country in COUNTRIES:
            for role in ROLES:
                for m in months:
                    lam = _hiring_rate(tech, country, role, m, drivers.loc[(tech, country, m)])
                    lam *= rng.gamma(shape=12.0, scale=1 / 12.0)  # extra month-to-month noise
                    for _ in range(rng.poisson(lam)):
                        n += 1
                        day = int(rng.integers(1, m.days_in_month + 1))
                        hire = m.to_timestamp() + pd.Timedelta(days=day - 1)
                        tenure_days = rng.exponential(365 / ANNUAL_ATTRITION)
                        leave = hire + pd.Timedelta(days=int(tenure_days))
                        rows.append(
                            {
                                "employee_id": f"E{n:05d}",
                                "email": f"e{n:05d}@example.com",
                                "hire_date": hire.date().isoformat(),
                                "leave_date": leave.date().isoformat() if leave <= HIST_END.to_timestamp(how="end") else "",
                                "technology": tech,
                                "country": country,
                                "role_family": role,
                                "contract_type": "temporary" if rng.random() < TEMP_SHARE[tech] else "permanent",
                                "gender": "F" if rng.random() < FEMALE_SHARE[role] else "M",
                            }
                        )
    return pd.DataFrame(rows)


def generate(seed: int = 42) -> SyntheticData:
    rng = np.random.default_rng(seed)
    projects = generate_projects(rng)
    employees = generate_employees(projects, rng)
    return SyntheticData(projects=projects, employees=employees)


def write(data: SyntheticData, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data.projects.to_csv(out_dir / "project_pipeline.csv", index=False)
    data.employees.to_csv(out_dir / "hr_hiring_extract.csv", index=False)
