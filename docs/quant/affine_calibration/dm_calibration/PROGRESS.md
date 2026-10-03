# Progress summary — EUR/USD/GBP caplet calibration (for review / Roland)

*Branch `ap_calib` on the fork (SugoiKitsune/noa). Paper: `article/dm_writeup.tex` (→ 15-page PDF).
Model/data reference: `MODEL_SPEC.md`. Swaption plan: `article/SWAPTION_OOS_PLAN.md`.
Last updated 3 Oct 2026.*

## Model
Two-factor **lifted-CIR Hull–White–Heston**, genuinely **multi-curve** (OIS/€STR discount + 6M
Euribor projection). 27 parameters. A low-order Markovian lift in the sense of Abi Jaber–El Euch
(2019) and Cuchiero–Teichmann (2019). Priced by O(N)-memory GPU Monte Carlo on a daily grid,
Andersen (2008) QE variance scheme, calibrated with gradient-free Adam–SPSA under a vega-weighted
normal-vol objective.

## Fits (3 seeds × 200k paths, `article/eval_daily.py` → `article/results/eval_*.json`)
| market | payoff | caplets | RMSE | ATM MAE | liquid core \|K−F\|<2% | full surface |
|---|---|---|---|---|---|---|
| **EUR** | 6M Euribor, term | 340 | **6.29 ± 0.03** | 2.42 | 3.73 | 6.29 |
| USD | 3M SOFR, in-arrears | 697 | 7.61 ± 0.05 | 2.73 | 3.79 | 16.96 |
| GBP | 3M SONIA, in-arrears | 697 | 7.77 ± 0.85 | 4.68 | 4.81 | 19.30 |

USD/GBP headline over t_fix ≥ 0.5Y; the first RFR caplet (t_fix=0, effective expiry ≈1M) has
near-intrinsic wings and is reported separately. Runs: tags `term` (EUR) and `rfr_lm` (USD/GBP),
800 iters warm-started from the Aug daily (`d252`) fits.

Nested ladder (Roland rec 2, dt=0.02, earlier in-arrears payoff): 1F/2F Gaussian ~46–47,
HW+det-vol 45.3 → **HWH 9.4** — SV is worth ~36 bp.

## Key findings
1. **Instrument conventions (Oct 2026, after Henrard 2nd ed., Roland's literature request).** EUR was
   priced as an in-arrears average while the market quotes forward-looking Euribor caplets. Pricing
   the term rate at t_fix (exact conditional mean of the OU factors) takes EUR from **8.57 → 6.29 bp**
   (same evaluation script); ATM/core unchanged, gain is off-ATM at every maturity 1–10Y. RFR caps
   now include the first live caplet, compound daily, use effective expiry t_fix + τ/3, and USD
   projects on SOFR (was Fed Funds). GBP from 6M: 9.6 → 7.8 bp.
2. **Time step.** With the in-arrears payoff the accrual window must be a whole number of steps;
   daily dt (126/63 steps for 6M/3M) beat dt=0.02 on core/ATM by 4–10× the seed scatter.

## What was done vs Roland's review
- Reworked entirely to developed markets (EUR primary, USD/GBP robustness); no RF content.
- Andersen QE (fixed Euler short-end bias: 0.5Y 106→40 bp); vega-weighted objective (ATM 8→3 bp).
- Nested-model ladder (rec 2) + bucketed/relative metrics (rec 4).
- Cross-market EUR/USD/GBP (one parameter set per market), all on the daily grid.
- Literature pass: Related-literature section, 53 verified references with DOIs, response to
  Henrard §5.3 ("short-rate models on forward curves"), naming clash with equity HHW flagged.

## Open items (priority order)
1. **Initial-curve fit** — the model forward drifts below the market forward (≈10 bp at 5Y,
   22–37 bp at 10Y): factors are added to f(t) with no convexity correction. Fix with a
   deterministic shift fitted to the discount curves under the calibrated dynamics (CIR++-style).
2. **RFR expiry convention** — confirm Bloomberg VCUB SOFR/SONIA flat-vol convention; alternative
   surface `vol_surface_end.csv` (expiry = accrual end) is already stripped.
3. **κ_f at its lower bound (1.5)** — relax the bound and refit.
4. **Ladder rerun** with the term payoff and daily step (current ladder is dt=0.02, old payoff).
5. **Swaption out-of-sample validation** — plan written; the strongest remaining addition.
6. **Short-end ATM undershoot** (EUR 0.5–1Y, RFR 0–1Y) — jumps (Roland rec 1; Andersen–Bang).
