# -*- coding: utf-8 -*-
"""Caplet stripper: flat VCUB cap vols -> per-period caplet normal-vol surface.

VCUB quotes *flat* (term) cap normal vols sigma_flat(T_cap, K): the single vol that
reprices the whole cap of maturity T_cap at strike K.  A cap is a strip of caplets, so
we bootstrap the per-period caplet vols sigma_cpl(t_fix, K) maturity-bucket by bucket.

Per-period, forward-looking Bachelier caplet (this is the DM fix for the RU
whole-period-average issue: expiry = fixing time t_fix, accrual = tau):

    Caplet = D(t_pay) * tau * [ (F - K) * Phi(d) + v * phi(d) ],   v = sigma * sqrt(t_fix),
             d = (F - K) / v

with F the simple forward key rate over [t_fix, t_fix+tau], D the OIS discount to payment.

Output: data/<ccy>/vol_surface.csv  (t_fix_yrs, strike, sigma_N_bp)  -- the calibration target.
Also derives the ATM caplet vols by interpolating the stripped surface at each caplet's forward.

RFR caps (SOFR/SONIA, --expiry end|lm). The IBOR schedule above is wrong for an
in-arrears cap in two ways (Henrard 2026, s.5.5 and 7.12): the first period [0,tau]
is not fixed at trade date, so it is a live caplet and belongs in the cap; and the
rate keeps setting until t_fix+tau, so the option expiry is not t_fix. With
--expiry the first caplet is included and each caplet carries an explicit expiry
texp_yrs used in place of t_fix in the Bachelier formula:
    end : texp = t_fix + tau            (variance to the end of accrual)
    lm  : texp = t_fix + tau/3          (Lyashenko-Mercurio linear decay; Henrard eq. 5.16)
Written to data/<ccy>/vol_surface_<expiry>.csv; vol_surface.csv is left alone.
Which one matches the Bloomberg VCUB flat-vol convention must be confirmed.

Run:  python caplet_stripper.py                         (legacy: all three, IBOR schedule)
      python caplet_stripper.py --ccy usd gbp --expiry lm
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import interp1d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DATA = os.path.join(ROOT, "data")

# caplet accrual tenor per market (verify vs VCUB underlying convention)
TENOR = {"eur": 0.5, "gbp": 0.25, "usd": 0.25}   # EUR 6M EURIBOR; GBP/USD 3M RFR


# ------------------------------------------------------------------ curves
def load_df(ccy, role):
    """Reconstruct discount factors from instantaneous fwd CSV -> interpolator DF(t)."""
    fname = {"ois": "forward_ois", "key": "forward_key_rate"}[role]
    df = pd.read_csv(os.path.join(DATA, ccy, f"{fname}.csv"))
    t = df["time_to_maturity"].values
    f = df["forward_rate"].values
    t_ext = np.concatenate([[0.0], t])
    f_ext = np.concatenate([[f[0]], f])
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (f_ext[1:] + f_ext[:-1]) * np.diff(t_ext))])
    lnDF = -integral                                   # DF = exp(-∫f)
    return interp1d(t_ext, lnDF, kind="linear", fill_value="extrapolate")


def simple_forward(ln_df_key, t0, t1):
    P0 = np.exp(ln_df_key(t0)); P1 = np.exp(ln_df_key(t1))
    return (P0 / P1 - 1.0) / (t1 - t0)


# ------------------------------------------------------------------ pricing
EXPIRY = {"start": lambda tf, tau: tf,
          "end": lambda tf, tau: tf + tau,
          "lm": lambda tf, tau: tf + tau / 3.0}


def bachelier_caplet(F, K, sigma, t_fix, tau, disc):
    if sigma <= 0 or t_fix <= 0:
        return disc * tau * max(F - K, 0.0)
    v = sigma * np.sqrt(t_fix)
    d = (F - K) / v
    return disc * tau * ((F - K) * norm.cdf(d) + v * norm.pdf(d))


def caplet_schedule(T_cap, tau, first=False):
    """Fixings tau, 2tau, ..., T-tau; pay at fix+tau. The first period [0,tau] is
    excluded (IBOR: fixed at trade date) unless first=True (in-arrears RFR)."""
    n = int(round(T_cap / tau))
    return [(j * tau, (j + 1) * tau) for j in range(0 if first else 1, n)]   # (t_fix, t_pay)


def cap_price(T_cap, K, sigma_flat, tau, fwd, disc, expiry="start"):
    """Price a flat cap: all its caplets at the single vol sigma_flat."""
    px = 0.0
    for t_fix, t_pay in caplet_schedule(T_cap, tau, first=expiry != "start"):
        px += bachelier_caplet(fwd[t_fix], K, sigma_flat, EXPIRY[expiry](t_fix, tau),
                               tau, disc[t_pay])
    return px


# ------------------------------------------------------------------ stripping
def strip_currency(ccy, verbose=True, expiry="start"):
    tau = TENOR[ccy]
    first = expiry != "start"
    tex = lambda tf: EXPIRY[expiry](tf, tau)
    ln_ois = load_df(ccy, "ois")
    ln_key = load_df(ccy, "key")

    surf = pd.read_csv(os.path.join(DATA, ccy, "cap_vol_surface.csv"))
    abs_rows = surf[surf.strike != "ATM"].copy()
    abs_rows["strike"] = abs_rows["strike"].astype(float)
    mats = sorted(abs_rows.maturity_yrs.unique())
    strikes = sorted(abs_rows.strike.unique())

    # precompute forwards / discounts on the full fixing grid
    all_fix = sorted({tf for T in mats for tf, _ in caplet_schedule(T, tau, first)})
    fwd = {tf: simple_forward(ln_key, tf, tf + tau) for tf in all_fix}
    disc = {tf + tau: float(np.exp(ln_ois(tf + tau))) for tf in all_fix}

    out = []                                       # (t_fix, strike, sigma_cpl_bp)
    worst = 0.0
    for K in strikes:
        vols = {float(T): float(abs_rows[(abs_rows.maturity_yrs == T) &
                (abs_rows.strike == K)].vol_bp.iloc[0]) / 1e4
                for T in mats if not abs_rows[(abs_rows.maturity_yrs == T) &
                (abs_rows.strike == K)].empty}
        # market cap prices from flat vols
        capP = {T: cap_price(T, K, vols[T], tau, fwd, disc, expiry) for T in vols}
        prev_T, prev_P = 0.0, 0.0
        sigma_prev = vols[min(vols)]
        for T in sorted(vols):
            bucket = [(tf, tp) for tf, tp in caplet_schedule(T, tau, first) if tp > prev_T + 1e-9]
            target = capP[T] - prev_P
            if not bucket or target <= 1e-14:
                sigma_i = sigma_prev
            else:
                def resid(s):
                    return sum(bachelier_caplet(fwd[tf], K, s, tex(tf), tau, disc[tp])
                               for tf, tp in bucket) - target
                try:
                    sigma_i = brentq(resid, 1e-6, 0.20, xtol=1e-10, maxiter=200)
                except ValueError:
                    sigma_i = sigma_prev                 # target unreachable -> carry
            for tf, tp in bucket:
                out.append((tf, K, sigma_i * 1e4))
            prev_T, prev_P, sigma_prev = T, capP[T], sigma_i

        # validation: reprice each flat cap from stripped caplet vols
        strp = {tf: s / 1e4 for tf, kk, s in out if kk == K}
        for T in vols:
            rp = sum(bachelier_caplet(fwd[tf], K, strp[tf], tex(tf), tau, disc[tp])
                     for tf, tp in caplet_schedule(T, tau, first))
            # back out flat vol of the repriced cap and compare
            def rv(s):
                return cap_price(T, K, s, tau, fwd, disc, expiry) - rp
            try:
                flat_rp = brentq(rv, 1e-6, 0.20) * 1e4
                worst = max(worst, abs(flat_rp - vols[T] * 1e4))
            except ValueError:
                pass

    strip_df = pd.DataFrame(out, columns=["t_fix_yrs", "strike", "sigma_N_bp"])

    # ATM caplet vols: interp stripped surface across strike at each caplet's forward
    atm_rows = []
    for tf in sorted(strip_df.t_fix_yrs.unique()):
        sl = strip_df[strip_df.t_fix_yrs == tf].sort_values("strike")
        if len(sl) >= 2:
            Fj = fwd[tf]
            g = interp1d(sl.strike, sl.sigma_N_bp, bounds_error=False,
                         fill_value=(sl.sigma_N_bp.iloc[0], sl.sigma_N_bp.iloc[-1]))
            atm_rows.append((tf, "ATM", float(g(Fj)), Fj))

    final = pd.concat([
        strip_df.assign(fwd=strip_df.t_fix_yrs.map(fwd)),
        pd.DataFrame(atm_rows, columns=["t_fix_yrs", "strike", "sigma_N_bp", "fwd"]),
    ], ignore_index=True)
    if first:
        final["texp_yrs"] = final.t_fix_yrs.map(tex)
    name = "vol_surface.csv" if expiry == "start" else f"vol_surface_{expiry}.csv"
    final.to_csv(os.path.join(DATA, ccy, name), index=False)

    if verbose:
        n_abs = (final.strike != "ATM").sum()
        print(f"[{ccy}] expiry={expiry}  tau={tau}  fixings={len(all_fix)}  strikes={len(strikes)}  "
              f"caplet pts={n_abs} (+{len(atm_rows)} ATM)  "
              f"flat-cap reprice max err={worst:.3f} bp")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ccy", nargs="+", default=["eur", "gbp", "usd"])
    ap.add_argument("--expiry", default="start", choices=list(EXPIRY),
                    help="start = IBOR schedule (legacy); end/lm = in-arrears RFR")
    a = ap.parse_args()
    for ccy in a.ccy:
        strip_currency(ccy, expiry=a.expiry)


if __name__ == "__main__":
    main()
