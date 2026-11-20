"""GDPR/BDSG controls applied before any data reaches a model.

The rules follow the thesis: keep only what the task needs (data minimisation),
replace IDs with pseudonyms, drop old records (storage limitation), keep
special or sensitive attributes out of the model, and never publish tables
with groups small enough to point at a person.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import pandas as pd

DIRECT_IDENTIFIERS = ("email",)
AUDIT_ONLY_COLUMNS = ("gender",)  # used in the aggregated fairness audit, never as a model feature
MODEL_COLUMNS = ("person_key", "hire_month", "technology", "country", "role_family", "contract_type")
DEV_SALT = "local-dev-salt-change-me"


@dataclass
class PreparedHR:
    model_records: pd.DataFrame   # pseudonymised, no sensitive attributes
    audit_records: pd.DataFrame   # pseudonymised, sensitive attributes kept, only for aggregated audits
    log: list[str]


def pseudonymise(ids: pd.Series, salt: str) -> pd.Series:
    return ids.astype(str).map(lambda v: hashlib.sha256(f"{salt}:{v}".encode()).hexdigest()[:16])


def prepare_hr_extract(
    raw: pd.DataFrame,
    as_of: pd.Period,
    retention_months: int,
    start: pd.Period,
    salt_env: str = "WF_PSEUDONYM_SALT",
) -> PreparedHR:
    log: list[str] = []
    df = raw.copy()
    salt = os.environ.get(salt_env, DEV_SALT)
    if salt == DEV_SALT:
        log.append(f"Pseudonym salt not set in {salt_env}; using the development salt.")

    df["person_key"] = pseudonymise(df["employee_id"], salt)
    dropped = [c for c in (*DIRECT_IDENTIFIERS, "employee_id") if c in df.columns]
    df = df.drop(columns=dropped)
    log.append(f"Dropped direct identifiers: {', '.join(dropped)}.")

    df["hire_month"] = pd.to_datetime(df["hire_date"]).dt.to_period("M")
    cutoff = as_of - retention_months
    old = df["hire_month"] < cutoff
    log.append(f"Retention: removed {int(old.sum())} records hired before {cutoff} ({retention_months} months).")
    df = df[~old]

    early = df["hire_month"] < start
    log.append(f"Analysis window: excluded {int(early.sum())} records before {start}.")
    df = df[~early & (df["hire_month"] <= as_of)]

    audit = df[["person_key", "hire_month", "technology", "country", "role_family", *AUDIT_ONLY_COLUMNS]].copy()
    model = df[list(MODEL_COLUMNS)].copy()
    log.append(f"Model records keep {len(MODEL_COLUMNS)} columns; {', '.join(AUDIT_ONLY_COLUMNS)} kept only for the audit.")
    return PreparedHR(model_records=model.reset_index(drop=True), audit_records=audit.reset_index(drop=True), log=log)


def monthly_panel(records: pd.DataFrame, months: pd.PeriodIndex, techs, countries, roles) -> pd.DataFrame:
    """Hires per segment and month, with explicit zeros for months without hires."""
    counts = records.groupby(["technology", "country", "role_family", "hire_month"]).size().rename("hires")
    grid = pd.MultiIndex.from_product([techs, countries, roles, months], names=["technology", "country", "role_family", "month"])
    counts.index = counts.index.set_names(["technology", "country", "role_family", "month"])
    panel = counts.reindex(grid, fill_value=0).reset_index()
    panel["hires"] = panel["hires"].astype(int)
    return panel


def suppress_small_cells(table: pd.DataFrame, size_col: str, value_cols: list[str], min_size: int) -> pd.DataFrame:
    """Blank out values for groups below the minimum size before a table is shared."""
    out = table.copy()
    small = out[size_col] < min_size
    for c in value_cols:
        out[c] = out[c].astype(object)
        out.loc[small, c] = "suppressed"
    out["suppressed"] = small
    return out
