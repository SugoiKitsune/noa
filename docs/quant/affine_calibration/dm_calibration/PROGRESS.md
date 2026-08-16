# Progress summary — EUR/USD/GBP caplet calibration (for review / Roland)

*Branch `ap_calib` on the fork (SugoiKitsune/noa). Paper: `article/dm_writeup.tex` (→ 9-page PDF).
Model/data reference: `MODEL_SPEC.md`. Swaption plan: `article/SWAPTION_OOS_PLAN.md`.*

## Model
Two-factor **lifted-CIR Hull–White–Heston**, genuinely **multi-curve** (OIS/€STR discount + 6M
Euribor projection). 27 parameters. A low-order Markovian lift in the sense of Abi Jaber–El Euch
(2019) and Cuchiero–Teichmann (2019). Priced by O(N)-memory GPU Monte Carlo, Andersen (2008) QE
variance scheme, calibrated with gradient-free Adam–SPSA under a vega-weighted normal-vol objective.

## Fits
| market | RMSE | liquid core \|K−F\|<2% | ATM MAE | dt |
|---|---|---|---|---|
| **EUR** | **9.4 bp** | **4.2 bp** | **2.9 bp** | 0.02 (weekly) |
| USD | 14.2 bp (best) | 8.0 | 5.6 | 0.03 |
| GBP | 16.8 bp | 7.8 | 5.2 | 0.03 |

Nested ladder (Roland rec 2): 1F/2F Gaussian ~46–47, HW+det-vol 45.3 → **HWH 9.4** — SV is worth ~36 bp.

## Key finding
**The Monte-Carlo time step was the bottleneck, not the model.** dt=0.03 (7.6 business days)
under-resolves the accrual-average integral and the variance path; dt=0.02 (5 days) drops EUR from
15.1 → 9.4 bp. Daily (dt≈1/252) is the natural target and would likely tighten further, at ~5× cost.

## What was done vs Roland's review
- Reworked entirely to developed markets (EUR primary, USD/GBP robustness); no RF content.
- Andersen QE (fixed Euler short-end bias: 0.5Y 106→40 bp); vega-weighted objective (ATM 8→3 bp).
- Nested-model ladder (rec 2) + bucketed/relative metrics (rec 4).
- Cross-market EUR/USD/GBP (one parameter set per market).
- Journal formatting + lifted-Heston literature positioning; full 9-page write-up that compiles.
- Patched the `signum` charting library (renders in VSCode, smooth surfaces, market/model overlay).

## Open items (priority order)
1. **USD/GBP at dt=0.02** — memory-blocked (GPU shared with other apps); needs freeing GPU or a
   path-chunking tweak. USD's dt=0.02 attempt at reduced paths regressed to 16.7 bp — under-resourced,
   not a model limit.
2. **dt convergence check** (dt=0.01, then daily) to confirm the EUR fit has plateaued.
3. **Swaption out-of-sample validation** — plan written; the single strongest remaining addition.
4. **Jumps** for the far-OTM wall (Roland rec 1) — the one structural diffusion limit.
5. Cross-market table is mixed-dt (EUR 0.02, USD/GBP 0.03) with an honest note until (1) is done.
