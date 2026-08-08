# DM rework — thesis + repo + Risk article (workplan)

Scope per Roland (23-Jul-26): rework the thesis on DM data, rework the GitHub repo,
target a Risk (Cutting Edge) submission. All RF-specific content out.

**As-of date for everything: 2026-06-30 (month-end).** Second snapshot (23-Jul curves)
kept for the stability section.

---

## 0. Non-negotiable guardrail — data licensing

The public repo (github.com/SugoiKitsune/noa fork) gets **code, methodology, calibrated
parameters, and result figures ONLY**. Raw Bloomberg-derived tables (VCUB surfaces, ICVS
curves — even transcribed) must **never** be committed: Bloomberg licence + bank compliance.
`data/` stays gitignored (already is). The repo ships a `data/README.md` describing exactly
which Bloomberg screens/tickers reproduce the dataset (VCUB caplet view Normal, ICVS 514/45/
141/490/85, PCS BGN) — standard practice for published quant work. Charts in the paper:
"Source: Bloomberg" attribution. Confirm bank policy for external publication.

## 1. Data — DONE / remaining

DONE (today):
- `data/Data_DM_markets.xlsx`: VCUB cap vols + swaption ATM (USD/EUR/GBP) + ICVS curves,
  all 30-Jun-26; second curve snapshot 23-Jul-26.
- `scripts/dm_data.py`: parses workbook, bootstraps curves (OIS annual-fixed bootstrap,
  EURIBOR-6M pseudo-DF, FF = SOFR + quoted basis), emits per-ccy pipeline CSVs:
  `forward_ois.csv`, `forward_key_rate.csv`, `cap_vol_surface.csv`, `swaption_atm.csv`.

REMAINING:
- [ ] **Caplet stripping** (`scripts/caplet_stripper.py`): VCUB CAPV are flat cap vols →
      bootstrap per-caplet normal vols per strike (piecewise vol between cap maturities,
      Bachelier per-period caplets on the bootstrapped curves; ATM column placed at the
      par cap rate from the curves). Output: `vol_surface.csv` = (T_fix, strike, sigma_N)
      — the calibration target. Validate: repriced flat cap vol == input to <0.5bp.
- [ ] EURIBOR multi-curve refinement (ESTR-discounted 6M bootstrap) — v2, low priority.
- [ ] Optional: VCUB caplet-view export from terminal to cross-check the stripper.

## 2. Model / pricer changes (the real quant work)

- [ ] **Per-period caplets** (replaces whole-period [0,T] average): payoff on the accrual
      [T−tau, T] per caplet, tau = 6M (EUR) / 3M-6M per market convention. This
      simultaneously fixes the thesis' eq-8-vs-code mismatch — in the DM version the
      instrument and the code will agree by construction.
- [ ] Dual-curve payoff: project EURIBOR-6M (key curve), discount on ESTR (ois curve).
      USD: SOFR single-curve caplets + FF as the "policy" curve for the k_t basis story.
      GBP: SONIA single-curve (k_t degenerate) — simplest market, good robustness case.
- [ ] Bachelier per-period vol inversion (sqrt(T)-type scaling, not sqrt(T/3)); the MC
      snapshot logic changes from one running sum to per-accrual sums (small change to
      the cumulative-sum design — snapshot at accrual start AND end).
- [ ] Keep: 2-factor lifted CIR-HW dynamics, PCHIP term structures, param_space (27p),
      Adam-SPSA, QE evaluation, PAVA post-processing — the machinery is market-agnostic.

## 3. Calibration campaign

- [ ] EUR primary (genuine dual curve, deepest cap market, Cuchiero benchmark comparable —
      target: beat/match ~14% relative RMSE on a DM surface).
- [ ] GBP + USD robustness; USD adds the market-quoted FF-SOFR basis as an observable
      benchmark for k_t (RU had to infer the basis — nice article point).
- [ ] Stability check: recalibrate EUR on the 23-Jul snapshot (curves done; needs VCUB
      23-Jul export — 10 min terminal task).
- [ ] Swaption ATM matrices: out-of-sample validation — price swaptions from the
      caplet-calibrated model; addresses the factor-identifiability critique.
- Discipline: warm-start chains, one change per run, VERSION_HISTORY-style log (same
  protocol as v22–v35).

## 4. Repo rework (public)

- [ ] New top-level structure under `docs/quant/affine_calibration/`: keep `scripts/`
      (market-agnostic engine + `dm_data.py`), add `dm_runs/` for configs+logs.
- [ ] `data/README.md`: how to reproduce the dataset from a Bloomberg terminal (no data).
- [ ] Strip/relocate RF-specific notebooks & docs from the *new* branch (do NOT rewrite
      git history — old thesis work stays in history; the article branch simply doesn't
      feature it).
- [ ] Commit hygiene per house rules: no outputs in notebooks, no binaries, no data.

## 5. Thesis rework (диплом)

- Keeps: Proposition 1 (Gaussian ⇒ flat smile — market-independent), model section,
  affine/multi-curve framework, calibration methodology, PAVA, GPU implementation.
- Replaces: market intro (EUR caplet market, RFR reform framing), data section (VCUB/ICVS),
  instrument definition (per-period RFR/IBOR caplets — now honest to the code), all
  results/figures (rerun `deck_charts.py`/`generate_figures.py` on DM runs).
- Drops: CBR/RUONIA specifics, RF market narrative, CBR-meeting jump conjecture (replace
  with ECB/BoE/Fed meeting-date jumps as future work — same idea, DM framing).

## 6. Risk Cutting Edge article

- ~3.5–4.5k words: intro (RFR options, multi-curve SV gap) → model → calibration method
  (SPSA+GPU+PAVA as the methodological contribution) → EUR results + GBP/USD robustness +
  swaption validation → conclusions. Compress from thesis; Roland on framing/lit; Andrey
  on runs/figures.
- Working title: "Calibrating a two-factor lifted CIR Hull-White-Heston model to RFR
  caplet surfaces: a multi-curve affine framework with hard no-arbitrage projection".

## Suggested order of execution

1. Caplet stripper (blocks everything downstream).
2. Per-period pricer change + unit test vs Bachelier closed form on flat vol.
3. First EUR calibration (coarse, v35-style warm protocol) → sanity → full run.
4. GBP, USD runs; swaption validation; 23-Jul stability run.
5. Thesis text swap + figures; repo restructure; article draft.
