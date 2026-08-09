# Swaption out-of-sample validation — plan (not yet implemented)

**Status:** scoped, deliberately *not* executed overnight. Repricing swaptions under the
stochastic-volatility model is a real modelling task; a rushed/approximate pricer could produce a
mismatch that reflects the *pricer*, not the model, and would undermine the whole point of an
independent check. Flagged for review before implementing.

## Goal
The calibration uses only **caplets**. `data/<ccy>/swaption_atm.csv` holds **ATM swaption normal
vols** on an `(expiry × tenor)` grid (228 points/market; e.g. EUR 1M×1Y = 47.9 bp). Repricing these
with the *already-calibrated* parameters — instruments that never entered the fit — is a genuine
out-of-sample test of the model's rate dynamics and correlation structure.

## What a swaption needs (and why it's non-trivial here)
A payer swaption expiring at `Te` on a swap with payment dates `T_1..T_n` pays
`(S(Te) - K)^+ · A(Te)` at `Te`, where the forward swap rate and annuity are
```
S(Te) = [ float leg PV ] / A(Te),    A(Te) = Σ_i τ_i · P_ois(Te, T_i)
```
Both require the **OIS discount curve at Te**, i.e. bond prices `P_ois(Te, T_i)` conditional on the
model state `(x, k, V)` at `Te`. In a pure Gaussian Hull–White model these are affine:
`P(Te,T) = exp(a(Te,T) − b(Te,T)·x(Te))`. **With stochastic variance `V`, the bond price is no
longer simply affine in the state** — that is the crux.

## Three viable methods (in order of rigour)
1. **Conditional-Gaussian (Heston–Hull–White) bond formula.** Conditional on the variance path,
   `x` is Gaussian and `P(Te,T)` is affine in `x` with a path-dependent integrated variance. Gives
   an (essentially) exact MC swaption price. Most correct; most work.
2. **Nested Monte Carlo.** Outer paths to `Te`; inner MC from `Te` for each `P_ois(Te,T_i)`.
   Correct but expensive; fine for a one-off validation on the ATM grid.
3. **Frozen-variance affine approximation.** Use the HW bond formula with `V` frozen at its
   expected level at `Te`. Cheap, ballpark only — acceptable as a first sanity check but *not* as
   the headline validation (its error is confounded with the model's).

## Recommended path
Implement **method 1** (conditional-Gaussian) for the ATM grid, sanity-check it against **method 2**
(nested MC) on a few nodes, then report model-vs-market swaption normal vols as the OOS table/figure
and add it to §7 of `dm_writeup.tex`. Expect this to be a focused half-day, done with the user in
the loop — not an unattended overnight run.

## Quick lower-bound check available now (optional)
As an immediate, cheap consistency check (not a substitute for the above): a swaption vol is
approximately a correlation-weighted average of the constituent forward-rate (caplet) vols. One can
compare the market ATM swaption vol against a Rebonato-style aggregate of the *calibrated* caplet
vols to confirm the order of magnitude and term-structure shape line up. This uses only the existing
caplet fit and no new pricer.
