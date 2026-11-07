# Experiment Log — RWB Hackathon

## EXP-002 — Baseline Ridge
- Model: Ridge
- Alpha: 4.0
- Features: route_id + status_1..status_6
- Validation: 5 rolling origins
- WAPE: 0.333411
- Relative Bias: 0.019374
- Score: 0.352785
- Result: Baseline control.

## EXP-003A — All Engineered Features
- Features: base statuses + aggregates + lags + deltas + temporal features
- WAPE: 0.333498
- Relative Bias: 0.025519
- Score: 0.359017
- Reproducibility: Confirmed. All metrics reproduced.
- Result: Worse than baseline. Not adopted.

## EXP-003B — Status Lags and Deltas
- Features: base statuses + lags (1, 2, 4) + delta (1)
- WAPE: 0.331830
- Relative Bias: 0.011023
- Score: 0.342853
- Result: Current best experiment.

## Current Leader
EXP-003B — Ridge with status lags and deltas.

## EXP-003C — Aggregated Status Features Only
- Features: base statuses + status_sum_1_3 + status_sum_4_6 + status_sum_all
- Excluded: lags, deltas, temporal features
- Model: Ridge
- Alpha: 4.0
- Validation: Same 5 rolling origins
- Status: Planned.