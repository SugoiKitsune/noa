# GBP calibration results (article data store)

*Derived / model-output numbers only — raw Bloomberg vol quotes are not stored here.*

## Nested-model ladder — vol-RMSE (bp)

| model | sim | RMSE | 0-1Y | 1-3Y | 3-7Y | 7Y+ | ATM | OTM | rel% |
|---|---|---|---|---|---|---|---|---|---|
| HWH | qe | 16.8 | 45.5 | 16.1 | 10.8 | 9.1 | 7.1 | 17.1 | 14.4 |

## Calibrated parameters — HWH (qe, best fit 16.771387100219727 bp)

*√θ is the instantaneous OU-driver scale (bp/√yr); the caplet implied vol is lower after mean-reversion (λ) and accrual averaging.*

| θ node (yr) | 0.25 | 0.5 | 1.0 | 2.0 | 3.0 | 5.0 | 7.0 | 10.0 |
|---|---|---|---|---|---|---|---|---|
| √θ₁ (bp/√yr) | 102.4 | 121.3 | 155.5 | 177.5 | 190.7 | 180.1 | 211.1 | 136.3 |
| √θ₂ (bp/√yr) | 48.8 | 88.0 | 75.1 | 129.6 | 137.6 | 148.8 | 170.9 | 105.5 |

| κ_fast | κ_slow | w_fast | ε (vol-of-vol) | λ_x | γ_HW | ξ_HW | v0 |
|---|---|---|---|---|---|---|---|
| 2.29523 | 0.21839 | 0.56008 | 0.08 | 0.12599 | 0.65137 | 0.00036 | 0.0 |

| ρ(1Y) | ρ(5Y) | ρ(10Y) | (rate–vol correlation, drives skew) |
|---|---|---|---|
| 0.19481 | 0.29553 | 0.23643 |  |

## Figures (in results/figures/, regenerable)

- `fig_gbp_market_surface.png`
- `fig_gbp_smile.png`