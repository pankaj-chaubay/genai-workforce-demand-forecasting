"""Runs the whole framework: input -> processing -> output -> feedback."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from . import cost, evaluation, explain, fairness, features, models, privacy, report, scenarios, stakeholder, synthetic
from .report import LABELS, md_table


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_data(cfg: dict, data_dir: Path) -> None:
    synthetic.write(synthetic.generate(cfg["seed"]), data_dir)


def run(config_path: Path, stakeholder_path: Path, notes_path: Path, data_dir: Path, out_dir: Path, report_dir: Path) -> dict:
    cfg = load_config(config_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / "hr_hiring_extract.csv").exists():
        generate_data(cfg, data_dir)
    raw = pd.read_csv(data_dir / "hr_hiring_extract.csv", dtype=str, keep_default_na=False)
    projects = pd.read_csv(data_dir / "project_pipeline.csv")

    # ---- Input layer -------------------------------------------------------
    start = pd.Period(cfg["data"]["start"], "M")
    end = pd.Period(cfg["data"]["end"], "M")
    prep = privacy.prepare_hr_extract(raw, end, cfg["privacy"]["retention_months"], start, cfg["privacy"]["pseudonym_salt_env"])
    (out_dir / "privacy_log.txt").write_text("\n".join(prep.log) + "\n", encoding="utf-8")

    d = cfg["data"]
    months = pd.period_range(start, end, freq="M")
    panel = privacy.monthly_panel(prep.model_records, months, d["technologies"], d["countries"], d["role_families"])
    inputs = stakeholder.load(stakeholder_path)
    fc_origin = pd.Period(cfg["forecast"]["origin"], "M")
    ctx = features.build_context(panel, projects, inputs, fc_origin + cfg["forecast"]["horizon"] - 1)

    drafts = stakeholder.draft_inputs_from_notes(notes_path.read_text(encoding="utf-8"))
    drafts.to_csv(out_dir / "stakeholder_note_drafts.csv", index=False)

    # ---- Processing layer: backtest ---------------------------------------
    b = cfg["backtest"]
    bt = evaluation.backtest(models.ALL_MODELS, ctx, [pd.Period(o, "M") for o in b["origins"]], b["horizon"])
    bt.to_csv(out_dir / "backtest_predictions.csv", index=False)
    summary = evaluation.summary(bt)
    summary.insert(1, "description", summary["model"].map(LABELS))
    summary.to_csv(out_dir / "model_comparison.csv", index=False)
    by_h = evaluation.accuracy_table(bt, b["horizon_buckets"]).pivot(index="model", columns="horizon_bucket", values="accuracy_pct")
    by_h = by_h[list(b["horizon_buckets"])].reset_index()
    by_h.to_csv(out_dir / "accuracy_by_horizon.csv", index=False)

    # ---- Feedback layer: audits and cost ----------------------------------
    f = cfg["fairness"]
    bias = fairness.forecast_bias(bt, f["max_abs_bias_pct"])
    bias.to_csv(out_dir / "bias_audit.csv", index=False)
    rep = fairness.representation(prep.audit_records, f["diversity_targets"], cfg["privacy"]["min_cell_size"])
    rep.to_csv(out_dir / "representation_audit.csv", index=False)
    costs = cost.cost_impact(bt, cfg["cost"])
    costs.to_csv(out_dir / "cost_impact.csv", index=False)

    # ---- Output layer: forward forecast -----------------------------------
    o = ctx.month_index(fc_origin) - 1
    hybrid = models.Hybrid(cfg["forecast"]["horizon"]).fit(ctx, o)
    fc = hybrid.predict(ctx, o, cfg["forecast"]["horizon"])
    fc = scenarios.add_intervals(fc, cfg["forecast"]["interval"])
    fc = scenarios.apply_scenarios(fc, inputs.scenarios)
    fc.round({c: 2 for c in fc.select_dtypes("number").columns}).to_csv(out_dir / "forecast_next_12_months.csv", index=False)
    gap = scenarios.recruiting_gap(fc, cfg["recruiting_capacity"])
    gap.to_csv(out_dir / "recruiting_gap.csv", index=False)

    X = hybrid.temporal.feature_frame(ctx, o, cfg["forecast"]["horizon"])
    sv = explain.shap_values(hybrid.temporal, X)
    imp = explain.global_importance(sv)
    imp.to_csv(out_dir / "shap_drivers.csv", index=False)
    rows = features.make_rows(ctx, [(o, h) for h in range(1, cfg["forecast"]["horizon"] + 1)])
    examples = []
    for seg in [("solar", "DE", "technician"), ("wind", "ES", "admin")]:
        idx = rows[(rows.technology == seg[0]) & (rows.country == seg[1]) & (rows.role_family == seg[2]) & (rows.horizon == 3)].index[0]
        examples.append((seg, rows.loc[idx, "month"], explain.explain_row(sv.loc[idx])))

    # ---- Charts and report -------------------------------------------------
    report.accuracy_chart(summary, fig_dir / "accuracy_by_model.png")
    report.shap_chart(imp, fig_dir / "shap_drivers.png")
    hist = panel[(panel.technology == "solar") & (panel.country == "DE")].groupby("month")["hires"].sum().reset_index()
    sub = fc[(fc.technology == "solar") & (fc.country == "DE")].groupby("month").agg(forecast=("forecast", "sum")).reset_index()
    sub = scenarios.add_intervals(sub, cfg["forecast"]["interval"])
    report.forecast_chart(hist, sub, "Solar DE, all roles: hires and 12-month forecast", fig_dir / "forecast_solar_de.png")

    write_report(report_dir / "forecast_report.md", cfg, prep.log, summary, by_h, bias, rep, costs, fc, gap, imp, examples, drafts, inputs)
    return {"summary": summary, "bias": bias, "costs": costs, "forecast": fc}


def write_report(path, cfg, privacy_log, summary, by_h, bias, rep, costs, fc, gap, imp, examples, drafts, inputs) -> None:
    by_h = by_h.copy()
    by_h["model"] = by_h["model"].map(LABELS)
    flagged = bias[bias["flag"] & (bias["model"] == "hybrid")]
    bias_wide = bias.pivot(index="model", columns="role_family", values="bias_pct").reset_index()
    bias_wide["model"] = bias_wide["model"].map(LABELS)
    costs_view = costs[["model", "missed_roles", "extra_roles", "error_cost_eur", "cost_per_hire_eur", "saving_vs_baseline_pct"]].copy()
    costs_view["model"] = costs_view["model"].map(LABELS)
    rep_view = rep[rep["year"] == rep["year"].max() - 1][["role_family", "year", "hires", "women_share", "target", "gap_to_target"]]
    totals = scenarios.add_intervals(fc.groupby("role_family")["forecast"].sum().reset_index(), cfg["forecast"]["interval"])
    totals["forecast"] = totals["forecast"].round(0).astype(int)
    scen_cols = [s["name"] for s in inputs.scenarios]
    scen = pd.DataFrame(
        {"scenario": ["Base"] + scen_cols, "hires_next_12_months": [int(round(fc["forecast"].sum()))] + [int(round(fc[c].sum())) for c in scen_cols]}
    )
    gap_view = gap[gap["gap"] > 0]

    lines = [
        "# Workforce demand forecast report (synthetic data)",
        "",
        "Generated by `python -m workforce_forecast run`. All figures come from synthetic data and the assumptions in `config/`.",
        "",
        "## 1. Input layer: privacy checks",
        "",
        *[f"- {x}" for x in privacy_log],
        "",
        "## 2. Processing layer: backtest",
        "",
        f"Rolling-origin backtest, origins {', '.join(cfg['backtest']['origins'])}, 12 months ahead each. Models only see data before each origin.",
        "",
        md_table(summary[["description", "accuracy_pct", "quarterly_accuracy_pct", "division_quarterly_accuracy_pct", "mae_roles_per_quarter", "bias_pct"]]),
        "",
        "Accuracy is 1 - WAPE. Monthly and quarterly figures are per segment (technology x country x role family). "
        "Division figures add up all role families per technology and country, which is the level most hiring plans are made at.",
        "",
        "### Accuracy by horizon (monthly, %)",
        "",
        md_table(by_h),
        "",
        "## 3. Feedback layer: bias audit",
        "",
        f"Mean signed error by role family, %. Flagged above {cfg['fairness']['max_abs_bias_pct']}%.",
        "",
        md_table(bias_wide),
        "",
    ]
    if len(flagged):
        lines += [
            "Flags for the hybrid model: " + ", ".join(f"{r.role_family} ({r.bias_pct:+.1f}%)" for r in flagged.itertuples()) + ". "
            "These go to the quarterly review with HR before the next planning round.",
            "",
        ]
    lines += [
        "### Representation in past hiring (aggregated, small groups suppressed)",
        "",
        md_table(rep_view),
        "",
        "Gender is never a model feature. This table only shows whether a plan copied from history would repeat its gaps.",
        "",
        "## 4. Cost impact of forecast errors",
        "",
        "Missed roles cost an urgent-hire premium, extra roles cost an over-hire penalty (assumptions in `config/default.yaml`).",
        "",
        md_table(costs_view),
        "",
        "## 5. Output layer: next 12 months",
        "",
        f"Hybrid model, first forecast month {cfg['forecast']['origin']}. Low and high give the {int(cfg['forecast']['interval'] * 100)}% interval for the 12-month total "
        "from the Poisson count distribution. It covers random variation in hiring, not model error, so treat it as the narrowest likely range.",
        "",
        md_table(totals),
        "",
        "### Scenarios",
        "",
        md_table(scen),
        "",
        "### Quarters where forecast hires exceed recruiting capacity",
        "",
        md_table(gap_view) if len(gap_view) else "None.",
        "",
        "## 6. Explainability",
        "",
        md_table(imp.head(8)[["driver", "mean_abs_shap"]]),
        "",
        *[f"- {s[0]} {s[1]} {s[2]}, {m}: {text}" for s, m, text in examples],
        "",
        "## 7. Stakeholder notes waiting for review",
        "",
        md_table(drafts.fillna("")),
        "",
        "Drafts are not used by the model until someone copies them into `config/stakeholder_inputs.yaml` with status `approved`.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
