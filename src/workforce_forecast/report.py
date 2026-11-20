"""Charts and the markdown report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BLUE = "#2a78d6"
GREY = "#b9b8b2"
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"

LABELS = {
    "seasonal_naive": "Same month last year",
    "poisson_rate": "Poisson rate",
    "additive_seasonal": "Trend + seasonality",
    "temporal_ml": "Temporal ML",
    "stakeholder_temporal": "Temporal ML + stakeholders",
    "stakeholder_poisson": "Poisson + stakeholders",
    "hybrid": "Hybrid (framework)",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def accuracy_chart(summary: pd.DataFrame, path: Path) -> None:
    df = summary.sort_values("quarterly_accuracy_pct")
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    colors = [BLUE if m == "hybrid" else GREY for m in df["model"]]
    ax.barh([LABELS[m] for m in df["model"]], df["quarterly_accuracy_pct"], color=colors, height=0.6)
    for y, v in enumerate(df["quarterly_accuracy_pct"]):
        ax.text(v + 0.8, y, f"{v:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Quarterly accuracy, segment level (1 - WAPE)", color=INK2, fontsize=9)
    ax.set_title("Backtest accuracy on synthetic data, 3 origins x 12 months", loc="left", fontsize=11, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def forecast_chart(history: pd.DataFrame, fc: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor=SURFACE)
    hx = [m.to_timestamp() for m in history["month"]]
    fx = [m.to_timestamp() for m in fc["month"]]
    ax.fill_between(fx, fc["low"], fc["high"], color=BLUE, alpha=0.15, linewidth=0, label="80% interval")
    ax.plot(hx, history["hires"], color=INK2, linewidth=2, label="Actual hires")
    ax.plot(fx, fc["forecast"], color=BLUE, linewidth=2, label="Forecast")
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel("Hires per month", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def shap_chart(importance: pd.DataFrame, path: Path, top: int = 10) -> None:
    df = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=SURFACE)
    ax.barh(df["driver"], df["mean_abs_shap"], color=BLUE, height=0.6)
    ax.set_xlabel("Mean |SHAP value| (log scale of the forecast)", color=INK2, fontsize=9)
    ax.set_title("What drives the temporal part of the forecast", loc="left", fontsize=11, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)
