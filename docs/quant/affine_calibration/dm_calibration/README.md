# DM calibration & Risk article

Self-contained workspace for the developed-markets (EUR/GBP/USD) rework of the
affine multi-curve stochastic-volatility calibration, and the *Risk* (Cutting Edge)
article that comes out of it. Separated from the Russian thesis work in
`../multi_theta/` (which stays as historical record). Reuses the market-agnostic
engine in `../scripts/` (simulation, pricing, param_space, calib_objective,
monotone_postprocess) — nothing is duplicated.

## Why this exists
Per the supervisor review (`…/Final paper/Results/Grinis_review_Popov_thesis.pdf`) and
Roland's directive: de-RF the paper, target *Risk*, rework thesis + repo on DM data.

## Layout
```
dm_calibration/
  README.md              this file
  dm_data.py             ingest data/Data_DM_markets.xlsx (VCUB vols + ICVS curves,
                         30-Jun-26) -> bootstrap curves -> pipeline CSVs in data/<ccy>/
  article/
    workplan.md          full thesis + repo + data rework plan
    article_plan.md      Risk article plan, built around Roland's 6 review recommendations
  notebooks/             per-market calibration notebooks (EUR primary, GBP/USD robustness)
  runs/                  calibration configs + logs + checkpoints
    calibration_weights/ checkpoints (gitignored — local only)
  figures/               generated article figures (gitignored; `git add -f` finals)
```

## Data (local only — never committed)
Market data lives in the repo-root `data/` (gitignored): `Data_DM_markets.xlsx` (raw
Bloomberg transcription, all 30-Jun-26 + a 23-Jul-26 curve snapshot) and the generated
per-currency pipeline files `data/<ccy>/{forward_ois,forward_key_rate,cap_vol_surface,
swaption_atm}.csv`. **Bloomberg licence + compliance: raw/derived market tables must not
be committed.** The public repo ships code, calibrated parameters, and result figures only;
`data/README.md` documents how to reproduce the dataset from a terminal.

## Reproduce the data pipeline
```
python dm_calibration/dm_data.py      # reads data/Data_DM_markets.xlsx, writes data/<ccy>/*.csv
```

## Status
- [x] Data collected (EUR/GBP/USD, 30-Jun-26) + ingestion/bootstrap (`dm_data.py`)
- [x] Caplet stripper (`caplet_stripper.py`) — flat VCUB cap vols -> per-period caplet
      surface `data/<ccy>/vol_surface.csv`; validated (flat-cap reprice err = 0.000 bp)
- [x] Per-period caplet pricer — numpy reference (`dm_pricer.py`) validated: ATM ≈ √θ,
      Gaussian limit -> flat smile (~1-2 bp), eps>0 -> convex smile (Prop. 1 demonstrated)
- [x] DM calibration driver `dm_calibrate.py` (torch, run on GPU): per-period pricer +
      vol-RMSE objective + self-contained Adam-SPSA, reusing scripts/ simulator + param_space
- [x] Nested-model ladder (review rec 2) built in: 1F_Gauss / 2F_Gauss / HW_detvol / HWH
      via constraint masks (`python dm_calibrate.py --ccy eur` runs the full ladder)
- [x] Article first draft `article/article_draft.tex` (compiles, 12pp; §5 = \TODO skeleton)
- [ ] RUN the EUR calibration on the GPU box -> fill §5 numbers; then GBP/USD + swaption OOS
- [ ] DM figures (adapt `../scripts/deck_charts.py`/`generate_figures.py` to the results)
- [ ] Confirm caplet tenor (EUR 6M vs 3M) vs the VCUB underlying
See `article/article_plan.md` for the full plan and the review-driven task list.
