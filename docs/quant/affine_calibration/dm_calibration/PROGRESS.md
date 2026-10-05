# Progress summary — EUR/USD/GBP caplet calibration (for review / Roland)

*Branch `ap_calib` on the fork (SugoiKitsune/noa). Paper: `article/dm_writeup.tex` (→ 15-page PDF).
Model/data reference: `MODEL_SPEC.md`. Swaption plan: `article/SWAPTION_OOS_PLAN.md`.
Last updated 5 Oct 2026.*

## Model
Two-factor **lifted-CIR Hull–White–Heston**, genuinely **multi-curve** (OIS/€STR discount + 6M
Euribor projection). 27 parameters. A low-order Markovian lift in the sense of Abi Jaber–El Euch
(2019) and Cuchiero–Teichmann (2019). Priced by O(N)-memory GPU Monte Carlo on a daily grid,
Andersen (2008) QE variance scheme, calibrated with gradient-free Adam–SPSA under a vega-weighted
normal-vol objective.

## Fits (3 seeds × 200k paths, `article/eval_daily.py` → `article/results/eval_*.json`)
| market | payoff | caplets | RMSE | ATM MAE | liquid core \|K−F\|<2% | full surface |
|---|---|---|---|---|---|---|
| **EUR** | 6M Euribor, term | 340 | **5.90 ± 0.02** | 1.98 | 2.81 | 5.90 |
| USD | 3M SOFR, in-arrears | 697 | 7.98 ± 0.08 | 2.17 | 3.96 | 16.04 |
| GBP | 3M SONIA, in-arrears | 697 | 6.72 ± 0.27 | 4.21 | 4.52 | 18.25 |

USD/GBP headline over t_fix ≥ 0.5Y; the first RFR caplet (t_fix=0, effective expiry ≈1M) has
near-intrinsic wings and is reported separately. Runs: tags `term_fc` (EUR) and `rfr_lm_fc`
(USD/GBP), all with `--fit_curve`, 800 iters warm-started from `term` / `rfr_lm`.

Nested ladder (Roland rec 2; rerun 5 Oct on the headline set-up — daily, term payoff, curve fit,
3-seed eval): 1F Gaussian 46.7 / 2F Gaussian 47.1 / HW+det-vol 45.3 → **HWH 5.9 bp**; ATM and
liquid-core errors ~18–21 bp for every Gaussian rung vs 2.0 / 2.8 for HWH.

## Key findings
1. **Instrument conventions (Oct 2026, after Henrard 2nd ed., Roland's literature request).** EUR was
   priced as an in-arrears average while the market quotes forward-looking Euribor caplets. Pricing
   the term rate at t_fix (exact conditional mean of the OU factors) takes EUR from **8.57 → 6.29 bp**
   (same evaluation script); ATM/core unchanged, gain is off-ATM at every maturity 1–10Y. RFR caps
   now include the first live caplet, compound daily, use effective expiry t_fix + τ/3, and USD
   projects on SOFR (was Fed Funds). GBP from 6M: 9.6 → 7.8 bp.
2. **Initial-curve fit (4 Oct).** Without a convexity shift the model forward sat 22–37 bp
   below the market at 10Y. `--fit_curve` adds a deterministic φ(t) (computed on the path set):
   forwards now match to <0.2 bp; EUR liquid core 3.73 → 2.81, ATM 2.42 → 1.98.
3. **Bloomberg conventions (checked 4 Oct, SWPM).** SOFR caps: SOFR projects and discounts,
   compounded in arrears, first period is a live caplet, priced in HW1F off the RFR BVOL cube
   (so effective expiry ≈ t_fix + τ/3 → `lm`). 1-day lookback, 1 cut-off day, 2BD pay delay.
4. **Time step.** With the in-arrears payoff the accrual window must be a whole number of steps;
   daily dt (126/63 steps for 6M/3M) beat dt=0.02 on core/ATM by 4–10× the seed scatter.

## What was done vs Roland's review
- Reworked entirely to developed markets (EUR primary, USD/GBP robustness); no RF content.
- Andersen QE (fixed Euler short-end bias: 0.5Y 106→40 bp); vega-weighted objective (ATM 8→3 bp).
- Nested-model ladder (rec 2) + bucketed/relative metrics (rec 4).
- Cross-market EUR/USD/GBP (one parameter set per market), all on the daily grid.
- Literature pass: Related-literature section, 53 verified references with DOIs, response to
  Henrard §5.3 ("short-rate models on forward curves"), naming clash with equity HHW flagged.

## Open items (priority order)
1. **Swaption out-of-sample validation** — plan written; build on exact Riccati bond prices
   (also gives φ(t) exactly instead of on the path set). SWPM 1Y×5Y ATM = 111.52 bp as a check.
2. **Short-end ATM undershoot** (EUR 0.5–1Y, GBP 0–1Y, USD 1–3Y) — jumps (Roland rec 1; Andersen–Bang).
3. **κ_f bound** (1.56 vs bound 1.5) — relax and refit as a robustness check.
4. SONIA cap SWPM check (only SOFR was checked; same convention assumed).
