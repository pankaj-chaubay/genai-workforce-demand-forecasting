# The framework

The framework has four layers: input, processing, output and feedback. This page describes each layer, the theory it rests on, how it maps to the code, and the risks it has to manage.

## Theory

Two theories from the thesis shape the design.

**Resource-Based View (Barney, 1991).** A company gains an advantage from resources others cannot copy. For workforce forecasting the two resources are the company's own hiring history and what its people know about upcoming projects, policy and seasons. Generic forecasting tools use neither. The framework treats both as inputs of equal standing.

**Task-Technology Fit (Goodhue and Thompson, 1995).** A tool helps only when it fits the task. The task here is forecasting project-driven hiring in a regulated sector with small data. That rules out models that need large, stable data sets and favours models that can take outside information and can be explained to HR.

## 1. Input layer

| Input | Owner | In the code |
|---|---|---|
| Hiring history (hires per segment and month) | HR / people analytics | `data/synthetic/hr_hiring_extract.csv` → `privacy.monthly_panel` |
| Project pipeline (development, construction, MW) | Engineering | `data/synthetic/project_pipeline.csv` → `synthetic.pipeline_drivers` |
| Peak hiring seasons, one-off periods | HR | `config/stakeholder_inputs.yaml` → `hr` |
| Announced policy changes with expected effect | Public affairs / policy | `config/stakeholder_inputs.yaml` → `policy` |
| Free-text notes | Anyone | `config/stakeholder_notes.txt` → `stakeholder.draft_inputs_from_notes` |
| Diversity targets | HR / leadership | `config/default.yaml` → `fairness` |

Before anything reaches a model, the privacy controls run (see [governance.md](governance.md)). Segments are technology x country x role family. Each role family has its own pipeline driver: engineers follow projects in development and construction, technicians follow MW under construction, project managers follow construction starts in the next three months, admin follows the number of active projects.

## 2. Processing layer

The thesis compared its model with two existing approaches: Greykite (Hosseini et al., 2022) and inhomogeneous Poisson processes (Shen and Huang, 2008). Both are here as baselines, together with a "same month last year" baseline that is close to how manual planning often works.

| Model | Idea | Sees stakeholder inputs |
|---|---|---|
| `seasonal_naive` | Same month last year | No |
| `poisson_rate` | Poisson regression on segment, month and trend | No |
| `additive_seasonal` | Trend with changepoints plus Fourier seasonality, per segment (the idea behind Greykite) | No |
| `temporal_ml` | Gradient-boosted direct multi-horizon model on hiring history | No |
| `stakeholder_temporal` | Same, plus pipeline, HR and policy inputs | Yes |
| `stakeholder_poisson` | Poisson regression on pipeline, seasons, one-off periods, policy | Yes |
| `hybrid` | Average of the two stakeholder models | Yes |

Two design choices matter:

* **Direct multi-horizon.** One model predicts all horizons from 1 to 12 months, with the horizon as a feature. History features stop at the forecast origin. Only things known in advance (calendar, planned projects, announced policy) are taken at the target month. This avoids feeding forecasts back in as if they were actuals.
* **Policy as an offset.** Every policy change is new, so no model can learn its effect from history. The policy team's expected change scales the forecast directly, and training targets are divided by the same factor so the model learns the rest.

## 3. Output layer

* Forecast per segment and month for the next 12 months, with an 80% interval from the Poisson count distribution.
* What-if scenarios, for example "subsidy extended" or "auction delayed". They are shown next to the base forecast and never change it.
* Recruiting gap: forecast hires per quarter and role family against the capacity of the recruiting team.

### Use cases from the thesis

| Use case | What the framework gives |
|---|---|
| Project-based recruitment | When a new wind or solar project is planned, the pipeline driver raises the forecast for the right roles and quarter, so hiring can start before construction does. |
| Skill gaps | Quarters where forecast hires exceed recruiting capacity, per role family. |
| Seasonal workforce | HR peak seasons for field crews are part of the model, so the Q2 technician peak is planned rather than discovered. |

## 4. Feedback layer

* **Bias audit** of forecast error by role family, every run, with a flag above 5%.
* **Representation audit** of past hiring against diversity targets, aggregated and with small groups suppressed.
* **Explanations**: SHAP values for the temporal model, written as short sentences.
* **Cost of errors**: missed roles and extra roles from the backtest, priced with assumptions from the config.
* **Review loop**: flags, explanations and new note drafts go to a quarterly review. Approved changes go back into `config/stakeholder_inputs.yaml`.

## Process view (SIPOC)

| Suppliers | Inputs | Process | Outputs | Customers |
|---|---|---|---|---|
| HR, engineering, policy team, HRIS | Hiring history, project pipeline, seasons, policy calendar, notes | Privacy controls → features → models → backtest → forecast → audits | Forecast with intervals, scenarios, recruiting gap, bias flags, explanations | Talent acquisition, HR business partners, project managers, finance |

## Risk matrix

From the methodology chapter of the thesis, with where each mitigation lives in the code.

| Risk | Likelihood | Impact | Mitigation | In the code |
|---|---|---|---|---|
| Bias: history skewed towards some groups or periods | High | High | Fairness checks, one-off periods marked by HR, diverse data | `fairness.py`, `hr.one_off_periods` |
| Privacy: misuse of employee data | Medium | High | Pseudonymised, minimised, aggregated pipeline | `privacy.py` |
| Regulation: sudden policy change | High | Medium | Policy calendar with expected effect, scenarios | `policy.events`, `scenarios.py` |
| Stakeholders left out | Medium | Medium | Structured inputs per team, notes parser, review loop | `stakeholder.py` |
| Low trust because output is hard to read | Medium | Medium | SHAP in plain sentences, simple report | `explain.py`, `reports/` |
