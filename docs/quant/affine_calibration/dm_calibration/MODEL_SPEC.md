# EUR calibration — what the model is and how the data is fed

*Grounded in the code (`dm_calibrate.py`, `dm_sim_qe.py`) and the data in `data/eur/`.
This is the "what exactly is going on" reference for validation and the article.*

## 1. What we feed in (three market inputs, all Bloomberg-derived)

| input | file | what it is |
|---|---|---|
| **OIS forward curve** (discounting) | `forward_ois.csv` | 360 monthly forwards, 0.08→30Y. €STR/OIS. |
| **Key-rate forward curve** (projection) | `forward_key_rate.csv` | 360 monthly forwards. Euribor-style projection curve. |
| **Caplet vol surface** | `vol_surface.csv` | per-period **normal (Bachelier) vols** in bp, stripped from the flat cap-vol surface (`cap_vol_surface.csv`). t_fix × strike. |

This is a genuine **dual-curve / multi-curve** feed. The key-rate curve sits **above** OIS by a basis that we can read directly:

| maturity | OIS fwd | key fwd | **basis** |
|---|---|---|---|
| 1Y | 2.46% | 2.67% | **21 bp** |
| 5Y | 2.65% | 2.83% | **18 bp** |
| 10Y | 3.21% | 3.30% | **9 bp** |

That basis is exactly what the `k` factor below is responsible for.

## 2. The stochastic process (joint curves)

Both curves are driven by shared + idiosyncratic factors:

```
OIS short rate :  r_ois(t) = f_ois(t) + x(t)
key   rate     :  r_key(t) = f_key(t) + x(t) + k(t)
```

- **x — shared level factor** (OU with *stochastic* vol):  `dx = -λ x dt + √V dW^x`
  Drives BOTH curves → moves the whole rate level. This is where the smile lives.
- **k — basis factor** (Hull–White, constant vol):  `dk = -γ k dt + ξ dW^k`
  Only on the key curve → models the OIS-vs-projection basis (the 9–21 bp above).
- **V — stochastic variance** = 2-factor lifted CIR (Heston):
  `V = w·V₁ + (1-w)·V₂`,  `dVᵢ = κᵢ(θᵢ(t) - Vᵢ) dt + ε√Vᵢ dW^{vᵢ}`
  Two factors (fast κ_f, slow κ_s) give a **term structure of vol-of-vol**; ε is the Heston vol-of-vol that creates smile *curvature*.
- **ρ(t) — leverage**: correlation between `dW^x` and `dW^{v₁}`. Positive here (payer skew) → tilts the smile so OTM caps are dear (the "wall").

θ₁(t), θ₂(t) and ρ(t) are **term structures** (PCHIP splines through 8 / 3 nodes) — not constants. That's how one parameter set bends across all maturities.

**Nested ladder** (`constrain()`): set ε=0 → flat smile (Gaussian HW); w=1, flat θ → 1F/2F Gaussian; ε=0, term-structure θ → HW+det-vol. So the ladder is the *same* model with parameters switched off — a clean apples-to-apples test that the smile needs stochastic vol.

## 3. The instrument and the partial-year inversion

Per-period (RFR / backward-looking) caplet — `price_perperiod()`:

- The underlying is the **average key rate over the accrual** `[t_fix, t_fix+τ]`:
  `A = (1/τ) ∫ r_key dt`  (τ = 0.5 for EUR 6M).
- Payoff `(A − K)⁺ · τ`, **discounted at OIS**: `D = exp(−∫ r_ois dt)` to pay date.
- Monte-Carlo: forward `F = E[A·D]/E[D]`, price `pv = E[(A−K)⁺ · τ · D]`.

**Inversion** (`implied_vol_bisect` on `bachelier_price`): invert `pv` to a **normal/Bachelier** vol with **expiry = t_fix** (vol enters as `σ√t_fix`). This "partial-year" convention (√t_fix, not √(t_fix+τ)) matches how the cap surface is quoted, so model vol and market vol are compared on the same footing.

## 4. The calibration — "wide testing"

One 27-parameter vector is fit to the **whole surface at once**: 20 maturities (0.5–10Y) × 17 strikes (−1.5% to +10%) = **340 caplets** jointly. Not ATM-only, not per-slice — the *entire* wide surface with a single parameter set.

- **Simulation**: GPU Monte Carlo, Andersen (2008) QE scheme for the CIR variance (unbiased in the Feller-violated DM regime; Euler+reflection over-inflated short-end vols).
- **Optimizer**: Adam-SPSA (gradient-free — the MC objective is noisy/non-differentiable).
- **Objective**: vega-weighted normal-vol RMSE — fits the liquid ATM tightly, down-weights the ~zero-vega deep-ITM/-OTM wings whose quoted vols are numerically ill-defined.

**Current fit**: 15.2 bp overall, **ATM mean error 2.9 bp**, liquid core (|K−F|<2%) **7.7 bp**. Nested ladder: HW+det-vol 45.3 → **HWH 15.2 bp** (stochastic vol worth ~30 bp).

## 5. Where it could feed better (open items)
- **Extend the surface**: `vol_surface.csv` actually carries maturities to **29.5Y** (1003 quotes); we currently cap at 10Y (`--max_mat`). Calibrating the full 30Y is a wider test.
- **Validate the √t_fix convention** against the exact VCUB quoting basis (confirm expiry treatment for the backward-looking period).
- **Out-of-sample**: `swaption_atm.csv` is on hand — reprice ATM swaptions with the calibrated model as an independent check (not used in calibration).
- **1Y / 10Y ATM** still ~10 bp soft — a long-end θ node or a maturity-weight tweak could tighten it.
