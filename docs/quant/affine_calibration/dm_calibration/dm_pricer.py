# -*- coding: utf-8 -*-
"""Per-period caplet Monte-Carlo pricer for the 2-factor lifted CIR-HWH model — NUMPY
reference implementation for validation.

Purpose: validate the *per-period* caplet payoff + Bachelier inversion and the
stochastic-volatility -> smile behaviour, independently of the (torch/GPU) calibration
engine. The production calibration reuses scripts/simulation.py (torch, drift/measure
consistent). This module is a light CPU reference: it checks that
  (a) the per-period pricing + Newton normal-vol inversion is correct, and
  (b) eps=0 (Gaussian HW) -> ~flat smile, eps>0 -> a genuine smile (the thesis' Prop.).

Per-period caplet: underlying A = (1/tau) ∫_{t_fix}^{t_fix+tau} r_key du (accrual-average
key rate); payoff tau*(A-K)^+ paid at t_fix+tau, discounted on OIS. Expiry = t_fix
(sqrt(t_fix) Bachelier scaling) — the DM fix for the RU whole-period-average issue.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DATA = os.path.join(ROOT, "data")


def load_fwd(ccy, role):
    fname = {"ois": "forward_ois", "key": "forward_key_rate"}[role]
    d = pd.read_csv(os.path.join(DATA, ccy, f"{fname}.csv"))
    return d["time_to_maturity"].values, d["forward_rate"].values


def bachelier_caplet(F, K, sigma, t_fix, tau, disc):
    if sigma <= 0 or t_fix <= 0:
        return disc * tau * max(F - K, 0.0)
    v = sigma * np.sqrt(t_fix)
    d = (F - K) / v
    return disc * tau * ((F - K) * norm.cdf(d) + v * norm.pdf(d))


def invert_normal_vol(pv, F, K, t_fix, tau, disc):
    intr = disc * tau * max(F - K, 0.0)
    if pv <= intr + 1e-15:
        return 0.0
    try:
        return brentq(lambda s: bachelier_caplet(F, K, s, t_fix, tau, disc) - pv,
                      1e-6, 0.30, xtol=1e-10, maxiter=200)
    except ValueError:
        return np.nan


def simulate(ccy, params, snap_times, dt=1/52, n_paths=40000, seed=0):
    """Simulate (I_key, I_ois) running integrals; snapshot at nearest grid index to
    each requested time in snap_times. Antithetic. Returns dict time -> integral array."""
    rng = np.random.default_rng(seed)
    p = params
    t_grid, f_key = load_fwd(ccy, "key")
    _, f_ois = load_fwd(ccy, "ois")
    T_max = max(snap_times)
    n = int(round(T_max / dt)) + 1
    tl = np.linspace(0.0, T_max, n)
    fk = np.interp(tl, t_grid, f_key)
    fo = np.interp(tl, t_grid, f_ois)
    # map each requested time to its nearest grid index
    idx_for = {t: int(np.argmin(np.abs(tl - t))) for t in snap_times}

    half = n_paths // 2
    N = 2 * half
    sdt = np.sqrt(dt)

    V1 = np.full(N, p["v0"]); V2 = np.full(N, p["v0"])
    x = np.zeros(N); k = np.zeros(N)
    Ikey = np.zeros(N); Iois = np.zeros(N)
    snap_key, snap_ois = {}, {}
    idx_to_times = {}
    for t, ix in idx_for.items():
        idx_to_times.setdefault(ix, []).append(t)
    rho = p.get("rho", 0.0)
    rho_c = np.sqrt(max(1 - rho * rho, 0.0))
    for i in range(n):
        if i in idx_to_times:
            for t in idx_to_times[i]:
                snap_key[t] = Ikey.copy(); snap_ois[t] = Iois.copy()
        if i == n - 1:
            break
        zh = rng.standard_normal((half, 4))             # per-step draw + antithetic
        Z = np.concatenate([zh, -zh], axis=0)
        V = p["w"] * V1 + (1 - p["w"]) * V2
        Ikey += (fk[i] + x + p.get("s_adj", 0.0) + k) * dt
        Iois += (fo[i] + x) * dt
        V1 = np.abs(V1 + p["kf"] * (p["th1"] - V1) * dt + p["eps"] * np.sqrt(np.maximum(V1, 0)) * sdt * Z[:, 0])
        V2 = np.abs(V2 + p["ks"] * (p["th2"] - V2) * dt + p["eps"] * np.sqrt(np.maximum(V2, 0)) * sdt * Z[:, 1])
        dWx = rho * Z[:, 0] + rho_c * Z[:, 2]
        x = x - p["lam"] * x * dt + np.sqrt(np.maximum(V, 0)) * sdt * dWx
        k = k - p["gam"] * k * dt + p["xi"] * sdt * Z[:, 3]
    return snap_key, snap_ois


def price_surface(ccy, params, caplets, **kw):
    """caplets: list of (t_fix, tau). Returns DataFrame (t_fix, K, model_vol_bp) for strikes."""
    snap_times = sorted({t for tf, ta in caplets for t in (tf, tf + ta)})
    snap_key, snap_ois = simulate(ccy, params, snap_times, **kw)

    rows = []
    for t_fix, tau in caplets:
        t_pay = t_fix + tau
        A = (snap_key[t_pay] - snap_key[t_fix]) / tau
        disc = np.exp(-snap_ois[t_pay])
        D = disc.mean()
        F = (A * disc).mean() / D                       # T-forward expectation of A
        for K in params["strikes"]:
            pv = (np.maximum(A - K, 0.0) * tau * disc).mean()
            vol = invert_normal_vol(pv, F, K, t_fix, tau, D)
            rows.append((t_fix, F, K, vol * 1e4))
    return pd.DataFrame(rows, columns=["t_fix", "F", "K", "model_vol_bp"])


def self_test():
    ccy = "eur"
    F0 = 0.027
    # near-ATM strikes only (~+/-1.5 sigma) where MC is reliable
    strikes = F0 + np.array([-0.010, -0.005, 0.0, 0.005, 0.010])
    base = dict(v0=3.6e-5, th1=3.6e-5, th2=3.6e-5, kf=1.5, ks=0.2, w=0.5,
                lam=0.4, gam=0.5, xi=0.0, rho=-0.3, s_adj=0.0, strikes=strikes)
    caplets = [(1.0, 0.5), (5.0, 0.5)]

    print("=== per-period pricer self-test (EUR curves) ===")
    # what matters is the RELATIVE vol-of-vol eps/sqrt(theta): ~0.3 is near-Gaussian,
    # ~7 is strong SV. (Large eps badly violates Feller; Euler+reflection then biases
    # the ATM level upward — a known CIR-scheme artifact handled by QE in production.)
    for label, eps in (("Gaussian limit (eps=0.002)", 0.002), ("Stochastic vol (eps=0.04)", 0.04)):
        pr = price_surface(ccy, {**base, "eps": eps}, caplets, n_paths=150000, dt=1/104, seed=1)
        print(f"\n{label}:")
        for tf in sorted(pr.t_fix.unique()):
            sl = pr[pr.t_fix == tf].sort_values("K")
            atm = sl.iloc[(sl.K - sl.F.iloc[0]).abs().argmin()].model_vol_bp
            spread = sl.model_vol_bp.max() - sl.model_vol_bp.min()
            wings = "/".join(f"{v:.1f}" for v in sl.model_vol_bp.values)
            print(f"  T={tf:.1f}  F={sl.F.iloc[0]*100:.2f}%  ATM={atm:.1f}bp  "
                  f"vols(K low->high)={wings}bp  smile-spread={spread:.1f}bp")
    print("\nExpect: Gaussian limit -> ~flat (spread ~ MC noise, few bp);")
    print("        eps>0 -> convex smile (higher vols in the wings, larger spread).")


if __name__ == "__main__":
    self_test()
