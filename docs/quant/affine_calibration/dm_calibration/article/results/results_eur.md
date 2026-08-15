# EUR calibration results (article data store)

*Derived / model-output numbers only — raw Bloomberg vol quotes are not stored here.*

## Nested-model ladder — vol-RMSE (bp)

| model | sim | RMSE | 0-1Y | 1-3Y | 3-7Y | 7Y+ | ATM | OTM | rel% |
|---|---|---|---|---|---|---|---|---|---|
| 1F_Gauss | None | 46.1 | 94.0 | 69.7 | 37.3 | 21.4 | 44.0 | 46.3 | 42.3 |
| 2F_Gauss | None | 47.0 | 92.7 | 72.4 | 37.6 | 21.6 | 44.9 | 47.1 | 43.0 |
| HW_detvol | None | 45.3 | 98.6 | 65.1 | 37.1 | 21.8 | 45.5 | 45.3 | 43.3 |
| HWH | None | 30.6 | 83.1 | 37.2 | 23.8 | 15.9 | 24.1 | 31.0 | 28.0 |
| HWH | qe | 9.4 | 24.3 | 10.3 | 7.7 | 6.3 | 4.2 | 9.6 | 8.0 |

## Calibrated parameters — HWH (qe, best fit 9.401049613952637 bp)

*√θ is the instantaneous OU-driver scale (bp/√yr); the caplet implied vol is lower after mean-reversion (λ) and accrual averaging.*

| θ node (yr) | 0.25 | 0.5 | 1.0 | 2.0 | 3.0 | 5.0 | 7.0 | 10.0 |
|---|---|---|---|---|---|---|---|---|
| √θ₁ (bp/√yr) | 85.7 | 140.4 | 152.6 | 169.3 | 179.8 | 158.3 | 212.0 | 116.6 |
| √θ₂ (bp/√yr) | 106.1 | 157.2 | 107.9 | 112.5 | 110.6 | 135.2 | 64.4 | 100.6 |

| κ_fast | κ_slow | w_fast | ε (vol-of-vol) | λ_x | γ_HW | ξ_HW | v0 |
|---|---|---|---|---|---|---|---|
| 1.50864 | 0.10483 | 0.69765 | 0.10716 | 0.18896 | 0.43254 | 0.00095 | 0.0 |

| ρ(1Y) | ρ(5Y) | ρ(10Y) | (rate–vol correlation, drives skew) |
|---|---|---|---|
| 0.44283 | 0.36521 | 0.28625 |  |

## Figures (in results/figures/, regenerable)

- `fig_eur_HWH_loss.png`
- `fig_eur_HWH_qe_loss.png`
- `fig_eur_convergence.png`
- `fig_eur_fit.png`
- `fig_eur_ladder.png`
- `fig_eur_market_surface.png`
- `fig_eur_overlay.png`
- `fig_eur_review.png`
- `fig_eur_smile.png`
- `fig_eur_surface3d.png`