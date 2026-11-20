"""Stakeholder inputs: the human part of the framework.

Structured inputs (YAML) become model features. Free-text notes from
stakeholders are turned into draft inputs by a parser. The default parser is
rule-based and runs offline. An optional local LLM (through Ollama) can be used
instead, which is the GenAI step the thesis suggests for scaling stakeholder
feedback. Either way, parsed items stay "needs_review" until a person approves
them, so nothing written in a note changes a forecast on its own.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

ROLE_WORDS = {
    "engineer": "engineer",
    "engineers": "engineer",
    "technician": "technician",
    "technicians": "technician",
    "project manager": "project_manager",
    "project managers": "project_manager",
    "admin": "admin",
    "administration": "admin",
}


@dataclass
class StakeholderInputs:
    seasonality: list[dict] = field(default_factory=list)
    one_off_periods: list[dict] = field(default_factory=list)
    policy_events: list[dict] = field(default_factory=list)
    scenarios: list[dict] = field(default_factory=list)
    engineering: dict = field(default_factory=dict)


def load(path) -> StakeholderInputs:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    def approved(items):
        return [i for i in items or [] if i.get("status", "approved") == "approved"]

    return StakeholderInputs(
        seasonality=approved(raw.get("hr", {}).get("seasonality")),
        one_off_periods=approved(raw.get("hr", {}).get("one_off_periods")),
        policy_events=approved(raw.get("policy", {}).get("events")),
        scenarios=raw.get("scenarios") or [],
        engineering=raw.get("engineering") or {},
    )


def _event_active(event: dict, month: pd.Period) -> bool:
    start = pd.Period(event["date"], "M") + int(event.get("lag_months") or 0)
    if month < start:
        return False
    dur = event.get("duration_months")
    return dur is None or month < start + int(dur)


def features(inputs: StakeholderInputs, panel_keys: pd.DataFrame) -> pd.DataFrame:
    """Add stakeholder features to rows with technology, country, role_family, month."""
    out = panel_keys[["technology", "country", "role_family", "month"]].copy()
    season = np.zeros(len(out))
    one_off = np.zeros(len(out))
    policy = np.zeros(len(out))
    for i, r in enumerate(out.itertuples(index=False)):
        for s in inputs.seasonality:
            if s["technology"] == r.technology and s["role_family"] == r.role_family and r.month.month in s["months"]:
                season[i] = 1.0
        for p in inputs.one_off_periods:
            if r.role_family in p["role_families"] and pd.Period(p["start"], "M") <= r.month <= pd.Period(p["end"], "M"):
                one_off[i] = 1.0
        for e in inputs.policy_events:
            if (
                e["technology"] == r.technology
                and e["country"] == r.country
                and r.role_family in e["role_families"]
                and _event_active(e, r.month)
            ):
                policy[i] += float(e["expected_change_pct"]) / 100.0
    out["hr_peak_season"] = season
    out["hr_one_off_period"] = one_off
    out["policy_expected_change"] = policy
    return out


# ---------------------------------------------------------------------------
# Free-text notes -> draft inputs
# ---------------------------------------------------------------------------

NOTE_SCHEMA_KEYS = {"technology", "country", "role_families", "start", "change_pct", "duration_months", "source", "text"}


def split_notes(text: str) -> list[dict]:
    notes = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        source = ""
        if lines[0].lower().startswith("from:"):
            source = lines[0].split(":", 1)[1].strip()
            lines = lines[1:]
        notes.append({"source": source, "text": " ".join(lines)})
    return notes


def parse_note_rules(note: dict) -> dict | None:
    """Pull a draft adjustment out of one note. Returns None if the note has no usable signal."""
    t = note["text"].lower()
    tech = "solar" if "solar" in t else "wind" if "wind" in t else None
    country = next((c for c in ("DE", "ES") if re.search(rf"\b{c.lower()}\b", t)), None)
    roles = sorted({v for k, v in ROLE_WORDS.items() if re.search(rf"\b{k}\b", t)})
    date = re.search(r"\b(20\d{2}-\d{2})\b", t)
    pct = re.search(r"(\d+(?:\.\d+)?)\s*%", t)
    if not (tech and date and pct and roles):
        return None
    down = re.search(r"\b(less|fewer|down|drop|cut|lower|reduce)\b", t[: pct.start() + 20])
    months = re.search(r"\b(\d+|one|two|three|four|five|six|nine|twelve)\s+months?\b", t)
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "nine": 9, "twelve": 12}
    duration = None
    if months:
        v = months.group(1)
        duration = int(v) if v.isdigit() else words[v]
    return {
        "technology": tech,
        "country": country,
        "role_families": roles,
        "start": date.group(1),
        "change_pct": -float(pct.group(1)) if down else float(pct.group(1)),
        "duration_months": duration,
        "source": note["source"],
        "text": note["text"],
        "parser": "rules",
        "status": "needs_review",
    }


LLM_PROMPT = """You turn workforce planning notes into JSON. Return one JSON object with keys:
technology ("solar" or "wind"), country ("DE" or "ES" or null), role_families (list from
engineer, technician, project_manager, admin), start ("YYYY-MM"), change_pct (number, negative
for fewer hires), duration_months (integer or null). If the note has no hiring signal, return {{}}.
Note: {text}"""


def parse_note_llm(note: dict, model: str = "llama3.1", url: str = "http://localhost:11434/api/generate") -> dict | None:
    """Same job as parse_note_rules, done by a local LLM. No data leaves the machine."""
    body = json.dumps({"model": model, "prompt": LLM_PROMPT.format(text=note["text"]), "format": "json", "stream": False})
    req = urllib.request.Request(url, data=body.encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(json.loads(resp.read())["response"])
    if not out:
        return None
    if not {"technology", "role_families", "start", "change_pct"} <= out.keys():
        return None
    if out["technology"] not in ("solar", "wind") or not re.fullmatch(r"20\d{2}-\d{2}", str(out["start"])):
        return None
    out["role_families"] = [r for r in out["role_families"] if r in set(ROLE_WORDS.values())]
    out.update({"source": note["source"], "text": note["text"], "parser": f"llm:{model}", "status": "needs_review"})
    return out


def draft_inputs_from_notes(text: str, use_llm: bool = False) -> pd.DataFrame:
    parse = parse_note_llm if use_llm else parse_note_rules
    rows = []
    for note in split_notes(text):
        parsed = parse(note)
        if parsed is None:
            rows.append({"source": note["source"], "text": note["text"], "status": "no_signal"})
        else:
            rows.append(parsed)
    return pd.DataFrame(rows)
