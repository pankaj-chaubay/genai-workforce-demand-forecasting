# Governance, privacy and fairness

## GDPR / BDSG controls

Workforce data is personal data. Under Art. 88 GDPR and § 26 BDSG, employee data may only be processed as far as needed for the employment relationship, and in Germany the works council has a say in technical systems that could monitor staff (§ 87(1) no. 6 BetrVG). The framework is built so that the forecast never needs data about individuals.

| Principle | What the code does |
|---|---|
| Data minimisation | Only six columns reach the model: pseudonymous key, hire month, technology, country, role family, contract type. |
| Pseudonymisation | Employee IDs are replaced with salted SHA-256 hashes. The salt comes from an environment variable and is never stored in the repository. |
| Direct identifiers | Email and employee ID are dropped before any processing. |
| Special and sensitive data | Gender is kept apart and used only in an aggregated audit. It is never a model feature. |
| Storage limitation | Records older than the retention period (48 months by default) are removed. |
| Small groups | Audit tables blank out any group with fewer than 10 people. |
| Purpose | The output is hires per segment and month. Nothing scores, ranks or selects a person. |
| Transparency | Every forecast can be explained with SHAP, and every stakeholder input carries a source and an approval status. |

Before using this with real data, involve the data protection officer and the works council, and record the processing in the record of processing activities.

## Bias

Two different checks run on every backtest:

1. **Forecast bias by role family.** If a model keeps over-forecasting a role, recruiting budgets and job adverts move in the wrong direction. The thesis found technical roles over-forecast because history contained a hiring wave. The fix there was to mark the period with HR and re-weight. Here HR marks it as a one-off period in the stakeholder file.
2. **Representation in past hiring.** If past hiring was skewed, a plan copied from it carries the skew forward. This audit compares the share of women hired per role family with the target. It does not change the forecast. It goes to HR as context for how the forecast hires are filled.

## Human in the loop

* Free-text notes become drafts with status `needs_review`. They only affect the forecast after a person copies them into the stakeholder file with status `approved`.
* What-if scenarios sit next to the base forecast and never replace it.
* Bias flags trigger a review, not an automatic correction.

## Proposed operating model (from the thesis)

* **AI ethics board** with HR, technical and policy representatives. It owns the bias thresholds, approves new input sources and reviews flags.
* **Quarterly bias audit**, proposed by the engineering stakeholders in the thesis interviews. The `bias_audit.csv` output is the starting point.
* **AI literacy for HR.** Most HR stakeholders in the thesis found model output hard to follow. Short training on how to read the forecast, intervals and explanations is part of the rollout.
* **Retraining** when a new policy, technology (for example hydrogen) or country is added, with a fresh backtest before the new model replaces the old one.
