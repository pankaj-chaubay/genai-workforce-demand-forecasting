"""Command line entry point.

    python -m workforce_forecast generate-data
    python -m workforce_forecast run
    python -m workforce_forecast parse-notes --llm
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import pipeline, stakeholder


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="workforce_forecast")
    p.add_argument("command", choices=["generate-data", "run", "parse-notes"])
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--stakeholders", default="config/stakeholder_inputs.yaml")
    p.add_argument("--notes", default="config/stakeholder_notes.txt")
    p.add_argument("--data", default="data/synthetic")
    p.add_argument("--out", default="outputs")
    p.add_argument("--reports", default="reports")
    p.add_argument("--llm", action="store_true", help="parse notes with a local Ollama model instead of the rules")
    a = p.parse_args(argv)

    if a.command == "generate-data":
        pipeline.generate_data(pipeline.load_config(Path(a.config)), Path(a.data))
        print(f"Synthetic data written to {a.data}")
    elif a.command == "parse-notes":
        print(stakeholder.draft_inputs_from_notes(Path(a.notes).read_text(encoding="utf-8"), use_llm=a.llm).to_string())
    else:
        res = pipeline.run(Path(a.config), Path(a.stakeholders), Path(a.notes), Path(a.data), Path(a.out), Path(a.reports))
        print(res["summary"].to_string(index=False))
        print(f"\nReport written to {a.reports}/forecast_report.md")


if __name__ == "__main__":
    main()
