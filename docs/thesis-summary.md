# Thesis summary

**Title:** Developing a conceptual framework for leveraging GenAI in workforce demand forecasting: a case study in the renewable energy sector
**Degree:** M.Sc. Data Analytics, Berlin School of Business and Innovation, 2025
**Type:** corporate-sponsored thesis, case study with a renewable energy company in Germany

## Research question

How can a conceptual framework using GenAI be effectively and ethically adapted to forecast workforce demand in the renewable energy sector, considering its dynamic and multidisciplinary challenges?

## Objectives

1. Test how accurate a GenAI-based forecasting model can be for project-driven hiring.
2. Examine the ethical issues: bias, transparency, GDPR/BDSG.
3. Find out how much input from HR, engineering and policy experts adds.
4. Build a mixed approach that combines the technical model with that input.

## Literature

Two rounds of literature review covered:

* forecasting methods, from Poisson processes for call centres (Shen and Huang, 2008; Koole and Li, 2023) to temporal representation learning (Garcia de Macedo et al., 2022) and Greykite at LinkedIn (Hosseini et al., 2022),
* GenAI and the future of work (Chui et al., 2023; Cazzaniga et al., 2024),
* AI in HR and its ethics (Budhwar et al., 2023; Hunkenschroer and Luetge, 2022),
* GenAI and EU law (Novelli et al., 2024),
* human-AI collaboration (Przegalinska et al., 2024; Fui-Hoon Nah et al., 2023; Rane, 2023; Russo, 2024).

The gaps it found: existing models were built for stable settings like call centres and tech platforms, ethics and GDPR were rarely built into technical work, and there was no clear way to bring domain experts into model building.

## Method

Mixed methods:

* **Qualitative:** semi-structured interviews with stakeholders from HR, engineering and policy, analysed thematically.
* **Quantitative:** hiring data for the company's wind and solar business from 2022 onwards (2020 and 2021 left out because of COVID effects), and a comparison with Greykite and Poisson-based forecasting.

## Findings, in short

* The hybrid approach, combining a temporal model with stakeholder input, forecast better than Greykite and Poisson processes on the company's data. Short-term forecasts were clearly better than 6 to 12 month forecasts, which suffered from policy changes.
* Stakeholder input made a measurable difference: HR on seasonality, policy experts on subsidies, engineers on contract types.
* Bias audits found technical roles over-forecast because history was skewed. Re-weighting reduced it.
* Most HR stakeholders found the output hard to interpret. SHAP-based explanations were proposed.
* Recommendations: AI ethics board, quarterly bias audits, explainable output, NLP to process stakeholder feedback, and a live policy feed for long-term forecasts.

## How this repository relates to the thesis

| In the thesis | In this repository |
|---|---|
| Company hiring data | Synthetic data with the same kinds of problems (`synthetic.py`) |
| Stakeholder interviews | A structured stakeholder file and example notes |
| Transformer-based temporal model | Gradient-boosted temporal model, chosen for small data and exact SHAP explanations |
| Comparison with Greykite and Poisson processes | Greykite-style trend + seasonality model and a Poisson rate model as baselines |
| Fairness through re-weighting | One-off periods from HR, bias audit, representation audit |
| GDPR measures (masking, limited retention) | `privacy.py` |
| Recommended NLP for stakeholder feedback | Note parser, rule-based or local LLM, with a review step |
| Recommended explainability (SHAP) | `explain.py` |
| Theoretical cost estimate | Cost of forecast errors from the backtest (`cost.py`) |

The thesis results were measured on the company's own data. Those data, the interview material and the figures stay with the company and are not reproduced here. The numbers in this repository come only from synthetic data.
