from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from workforce_forecast import cost, evaluation, fairness, features, models, privacy, scenarios, stakeholder, synthetic

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    return synthetic.generate(seed=7)


@pytest.fixture(scope="module")
def prepared(data):
    return privacy.prepare_hr_extract(data.employees, pd.Period("2025-06", "M"), 48, pd.Period("2022-01", "M"))


@pytest.fixture(scope="module")
def ctx(data, prepared):
    months = pd.period_range("2022-01", "2025-06", freq="M")
    panel = privacy.monthly_panel(prepared.model_records, months, synthetic.TECHS, synthetic.COUNTRIES, synthetic.ROLES)
    inputs = stakeholder.load(ROOT / "config" / "stakeholder_inputs.yaml")
    return features.build_context(panel, data.projects, inputs, pd.Period("2026-06", "M"))


# ---- synthetic data ---------------------------------------------------------


def test_generator_is_deterministic():
    a = synthetic.generate(seed=3).employees
    b = synthetic.generate(seed=3).employees
    pd.testing.assert_frame_equal(a, b)


def test_generator_covers_every_segment(data):
    seg = data.employees.groupby(["technology", "country", "role_family"]).size()
    assert len(seg) == len(synthetic.TECHS) * len(synthetic.COUNTRIES) * len(synthetic.ROLES)
    assert seg.min() > 0


# ---- privacy ----------------------------------------------------------------


def test_identifiers_and_sensitive_fields_never_reach_the_model(prepared):
    cols = set(prepared.model_records.columns)
    assert not cols & {"email", "employee_id", "gender"}
    assert "gender" in prepared.audit_records.columns


def test_pseudonyms_are_stable_and_salted():
    ids = pd.Series(["E00001", "E00002"])
    assert privacy.pseudonymise(ids, "a").tolist() == privacy.pseudonymise(ids, "a").tolist()
    assert privacy.pseudonymise(ids, "a").tolist() != privacy.pseudonymise(ids, "b").tolist()


def test_retention_and_window(prepared):
    assert prepared.model_records["hire_month"].min() >= pd.Period("2022-01", "M")
    assert prepared.model_records["hire_month"].max() <= pd.Period("2025-06", "M")


def test_small_cells_are_suppressed():
    t = pd.DataFrame({"n": [3, 30], "share": [0.33, 0.5]})
    out = privacy.suppress_small_cells(t, "n", ["share"], min_size=10)
    assert out.loc[0, "share"] == "suppressed"
    assert out.loc[1, "share"] == 0.5


def test_panel_has_zero_rows_for_quiet_months(prepared):
    months = pd.period_range("2022-01", "2025-06", freq="M")
    panel = privacy.monthly_panel(prepared.model_records, months, synthetic.TECHS, synthetic.COUNTRIES, synthetic.ROLES)
    assert len(panel) == 16 * len(months)
    assert panel["hires"].sum() == len(prepared.model_records)


# ---- stakeholder inputs -----------------------------------------------------


def test_stakeholder_features_follow_the_yaml(ctx):
    si = ctx.segments.index(("solar", "DE", "technician"))
    may = ctx.month_index(pd.Period("2024-05", "M"))
    oct_ = ctx.month_index(pd.Period("2024-10", "M"))
    assert ctx.known["hr_peak_season"][si, may] == 1
    assert ctx.known["hr_peak_season"][si, oct_] == 0
    # subsidy: announced 2023-01, 3-month lag, 18 months
    assert ctx.known["policy_expected_change"][si, ctx.month_index(pd.Period("2023-04", "M"))] == pytest.approx(0.25)
    assert ctx.known["policy_expected_change"][si, ctx.month_index(pd.Period("2024-10", "M"))] == 0


def test_rule_parser_reads_a_note():
    note = {"source": "eng", "text": "Solar parks in ES start construction from 2025-09. Technician hiring up about 20% for six months."}
    out = stakeholder.parse_note_rules(note)
    assert out["technology"] == "solar" and out["country"] == "ES"
    assert out["role_families"] == ["technician"]
    assert out["change_pct"] == 20 and out["duration_months"] == 6
    assert out["status"] == "needs_review"


def test_rule_parser_reads_a_cut():
    out = stakeholder.parse_note_rules({"source": "", "text": "Wind engineers in DE: about 15% fewer hires from 2026-01."})
    assert out["change_pct"] == -15


def test_rule_parser_ignores_notes_without_signal():
    assert stakeholder.parse_note_rules({"source": "", "text": "Nothing new for project managers."}) is None


def test_unapproved_inputs_are_ignored(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(
        "policy:\n  events:\n    - {name: x, date: '2024-01', technology: solar, country: DE, role_families: [admin],"
        " lag_months: 0, duration_months: null, expected_change_pct: 50, status: needs_review}\n"
    )
    assert stakeholder.load(p).policy_events == []


# ---- models and backtest ----------------------------------------------------


def test_masking_hides_the_future(ctx):
    o = ctx.month_index(pd.Period("2024-06", "M"))
    seen = ctx.masked(ctx.months[o])
    assert np.isnan(seen.y[:, o + 1 :]).all()
    assert not np.isnan(seen.y[:, : o + 1]).any()


@pytest.mark.parametrize("cls", models.ALL_MODELS)
def test_every_model_forecasts_12_non_negative_months(ctx, cls):
    o = ctx.month_index(pd.Period("2024-06", "M"))
    seen = ctx.masked(ctx.months[o])
    fc = cls().fit(seen, o).predict(seen, o, 12)
    assert len(fc) == 16 * 12
    assert (fc["forecast"] >= 0).all()
    assert fc["month"].min() == pd.Period("2024-07", "M")


def test_training_never_uses_targets_after_the_origin(ctx):
    o = ctx.month_index(pd.Period("2024-01", "M"))
    m = models.Hybrid().fit(ctx.masked(ctx.months[o]), o)
    assert m.temporal.model is not None
    rows = features.make_rows(ctx.masked(ctx.months[o]), [(oo, h) for h in range(1, 13) for oo in range(o - h + 1)])
    assert rows.dropna(subset=["actual"])["month"].max() <= ctx.months[o]


def test_stakeholder_inputs_improve_on_history_only(ctx):
    bt = evaluation.backtest([models.TemporalML, models.Hybrid], ctx, [pd.Period("2024-01", "M"), pd.Period("2024-07", "M")], 12)
    s = evaluation.summary(bt).set_index("model")
    assert s.loc["hybrid", "quarterly_accuracy_pct"] > s.loc["temporal_ml", "quarterly_accuracy_pct"]


def test_wape():
    assert evaluation.wape([10, 10], [10, 10]) == 0
    assert evaluation.wape([10, 10], [5, 15]) == pytest.approx(0.5)


# ---- output and feedback ----------------------------------------------------


def _toy_backtest():
    months = pd.period_range("2024-01", "2024-06", freq="M")
    rows = []
    for model, f in [("seasonal_naive", 12.0), ("hybrid", 10.0)]:
        for m in months:
            rows.append({"model": model, "origin": pd.Period("2023-12", "M"), "technology": "solar", "country": "DE",
                         "role_family": "technician", "month": m, "horizon": 1, "forecast": f, "actual": 10.0})
    return pd.DataFrame(rows)


def test_bias_audit_flags_over_forecast():
    b = fairness.forecast_bias(_toy_backtest(), 5).set_index("model")
    assert b.loc["seasonal_naive", "bias_pct"] == 20
    assert bool(b.loc["seasonal_naive", "flag"]) and not bool(b.loc["hybrid", "flag"])


def test_cost_counts_over_hiring():
    c = cost.cost_impact(_toy_backtest(), {"cost_per_hire": 1000, "urgent_hire_premium": 500, "overhire_cost": 300}).set_index("model")
    assert c.loc["seasonal_naive", "extra_roles"] == 12  # 2 extra per month for 6 months
    assert c.loc["hybrid", "error_cost_eur"] == 0
    assert c.loc["hybrid", "saving_vs_baseline_pct"] > 0


def test_scenarios_leave_the_base_forecast_alone():
    fc = pd.DataFrame({"technology": ["solar"] * 2, "country": ["DE"] * 2, "role_family": ["technician"] * 2,
                       "month": [pd.Period("2025-12", "M"), pd.Period("2026-01", "M")], "forecast": [10.0, 10.0]})
    out = scenarios.apply_scenarios(fc, [{"name": "up", "technology": "solar", "country": "DE",
                                           "role_families": ["technician"], "start": "2026-01", "change_pct": 20}])
    assert out["forecast"].tolist() == [10.0, 10.0]
    assert out["up"].tolist() == [10.0, 12.0]


def test_intervals_contain_the_forecast():
    out = scenarios.add_intervals(pd.DataFrame({"forecast": [0.5, 5.0, 50.0]}), 0.8)
    assert (out["low"] <= out["forecast"]).all() and (out["high"] >= out["forecast"]).all()
