# USD calibration results (article data store)

*Derived / model-output numbers only — raw Bloomberg vol quotes are not stored here.*

## Nested-model ladder — vol-RMSE (bp)

| model | sim | RMSE | 0-1Y | 1-3Y | 3-7Y | 7Y+ | ATM | OTM | rel% |
|---|---|---|---|---|---|---|---|---|---|
| HWH | qe | 14.2 | 32.6 | 14.7 | 9.6 | 11.2 | 6.7 | 14.4 | 12.0 |

## Calibrated parameters — HWH (qe, best fit 14.155355453491211 bp)

*√θ is the instantaneous OU-driver scale (bp/√yr); the caplet implied vol is lower after mean-reversion (λ) and accrual averaging.*

| θ node (yr) | 0.25 | 0.5 | 1.0 | 2.0 | 3.0 | 5.0 | 7.0 | 10.0 |
|---|---|---|---|---|---|---|---|---|
| √θ₁ (bp/√yr) | 183.4 | 160.0 | 195.9 | 162.0 | 203.8 | 195.3 | 226.3 | 160.9 |
| √θ₂ (bp/√yr) | 174.9 | 107.3 | 141.2 | 154.7 | 105.8 | 100.1 | 123.1 | 55.4 |

| κ_fast | κ_slow | w_fast | ε (vol-of-vol) | λ_x | γ_HW | ξ_HW | v0 |
|---|---|---|---|---|---|---|---|
| 2.03192 | 0.2744 | 0.65091 | 0.08366 | 0.25373 | 0.71375 | 0.00043 | 1e-05 |

| ρ(1Y) | ρ(5Y) | ρ(10Y) | (rate–vol correlation, drives skew) |
|---|---|---|---|
| 0.2032 | 0.31297 | 0.34821 |  |

## Figures (in results/figures/, regenerable)

- `fig_usd_market_surface.png`
- `fig_usd_smile.png`