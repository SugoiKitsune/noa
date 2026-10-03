# -*- coding: utf-8 -*-
"""DM market data ingestion for the Risk article (EUR/GBP/USD, as-of 2026-06-30).

Parses data/Data_DM_markets.xlsx (VCUB cap vols + swaption ATM + ICVS curves,
transcribed from Bloomberg) and emits pipeline-format CSVs into data/<ccy>/:

    forward_ois.csv       time_to_maturity, forward_rate   (instantaneous fwd, decimal)
    forward_key_rate.csv  time_to_maturity, forward_rate   (projection / "key" curve)
    cap_vol_surface.csv   maturity_yrs, strike, vol_bp     (strike='ATM' rows included)
    swaption_atm.csv      expiry_yrs, tenor_yrs, vol_bp

Curve bootstrap (v1, documented approximations):
  * mid = (bid+ask)/2 of the SWAP RATES block (+ cash pillar)
  * T<=1Y: single-payment OIS,   DF = 1/(1 + r*tau(T))
  * T> 1Y: annual fixed leg; payment schedule 1..floor(T) plus stub T;
           gap pillars solved with log-linear DF interpolation + brentq
  * day-count: ACT/360 for EUR/USD (tau = T*365/360), ACT/365 for GBP (tau = T)
  * EURIBOR-6M projection: classic single-curve pseudo-discount bootstrap from
    cash + FRA chain + par swaps (fixed 30/360 annual).  Proper multi-curve
    (ESTR-discounted) bootstrap is a listed refinement.
  * USD FF curve: composite par set = outright FF OIS (<1Y) plus
    SOFR_par(T) + basis_mid(T)/100 for 1Y+ (ICVS 85 quotes basis in bp).
  * instantaneous forwards: PCHIP through (t, -ln DF), analytic derivative.
"""
import os
import numpy as np
import openpyxl
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
XLSX = os.path.join(ROOT, "data", "Data_DM_markets.xlsx")
GRID = np.round(np.arange(1, 361) / 12.0, 6)          # monthly grid to 30Y

# ---------------------------------------------------------------- parsing
def term_to_years(term):
    n, unit = term.split()
    n = float(n)
    unit = unit.upper()
    return {"DY": n / 365.0, "WK": n * 7 / 365.0, "MO": n / 12.0, "YR": n}[unit]


def parse_curve_sheet(ws):
    """Return dict(cash=[(T, mid)], swaps=[(T, mid)]) from a CRV sheet."""
    rows = list(ws.values)
    out = {"cash": [], "swaps": []}
    section = None
    for r in rows:
        c0 = r[0] if r else None
        if isinstance(c0, str):
            if c0.startswith("CASH"):
                section = "cash"; continue
            if c0.startswith(("FUTURES", "SERIAL")):
                section = None; continue          # futures not used in v1 bootstrap
            if c0.startswith("SWAP"):
                section = "swaps"; continue
            if c0 == "Term":
                continue
        if section and isinstance(c0, str) and len(r) >= 3:
            bid, ask = r[1], r[2]
            if bid is None and ask is None:
                continue
            mid = np.mean([v for v in (bid, ask) if v is not None])
            try:
                out[section].append((term_to_years(c0), float(mid)))
            except (ValueError, KeyError):
                pass
    return out


def parse_ff_basis_sheet(ws):
    """ICVS 85: outright FF OIS short end + FF-SOFR basis (bp) from 1Y."""
    rows = list(ws.values)
    swaps, basis = [], []
    for r in rows:
        if not r or not isinstance(r[0], str) or len(r) < 6:
            continue
        try:
            T = term_to_years(r[0])
        except (ValueError, KeyError):
            continue
        mid, rtype = r[4], r[5]
        if rtype == "Swap":
            swaps.append((T, float(mid)))
        elif rtype == "Basis":
            basis.append((T, float(mid)))
    return swaps, basis


def parse_fras(ws):
    """FRA nXm rows from the EURIBOR sheet -> [(start_yr, end_yr, mid)]."""
    fras = []
    for r in ws.values:
        c0 = r[0] if r else None
        if isinstance(c0, str) and c0.startswith("FRA ") and r[1] is not None:
            n, m = c0.replace("FRA ", "").split("X")
            fras.append((float(n) / 12.0, float(m) / 12.0, (r[1] + r[2]) / 2.0))
    return fras


# ---------------------------------------------------------------- bootstrap
def bootstrap_ois(pillars, act360=True):
    """pillars: [(T, par_rate_pct)] -> dict T -> DF. Annual fixed leg past 1Y."""
    ay = (365.0 / 360.0) if act360 else 1.0     # accrual per year of fixed leg
    dfs = {0.0: 1.0}
    known_t = [0.0]

    def df_interp(t, extra_t=None, extra_df=None):
        ts = sorted(known_t + ([extra_t] if extra_t else []))
        ln = {**{k: np.log(dfs[k]) for k in known_t},
              **({extra_t: np.log(extra_df)} if extra_t else {})}
        xs = np.array(ts); ys = np.array([ln[x] for x in xs])
        return float(np.exp(np.interp(t, xs, ys)))

    for T, r in sorted(pillars):
        r = r / 100.0
        if T <= 1.0 + 1e-9:
            dfs[T] = 1.0 / (1.0 + r * T * ay)
        else:
            pay_ts = [float(k) for k in range(1, int(np.floor(T - 1e-9)) + 1)]
            if T - pay_ts[-1] > 1e-6:
                pay_ts.append(T)
            accr = np.diff([0.0] + pay_ts) * ay

            def gap(df_T):
                pv_fix = sum(a * (df_interp(t, T, df_T) if t > known_t[-1] else df_interp(t))
                             for a, t in zip(accr, pay_ts[:-1]))
                pv_fix += accr[-1] * df_T
                return r * pv_fix + df_T - 1.0

            dfs[T] = brentq(gap, 1e-8, 1.5)
        known_t = sorted(dfs)
    return dfs


def bootstrap_euribor6m(cash, fras, swaps):
    """Classic single-curve pseudo-DF bootstrap: cash + FRA chain + 30/360 par swaps."""
    tau = 365.0 / 360.0
    dfs = {0.0: 1.0}
    for T, r in cash:                              # simple deposits
        dfs[T] = 1.0 / (1.0 + (r / 100.0) * T * tau)

    def interp(t):
        xs = np.array(sorted(dfs)); ys = np.log([dfs[x] for x in xs])
        return float(np.exp(np.interp(t, xs, ys)))

    for s, e, f in sorted(fras):                   # PD(e) = PD(s)/(1 + F*tau6)
        dfs[e] = interp(s) / (1.0 + (f / 100.0) * (e - s) * tau)

    swap_pillars = [(T, r) for T, r in swaps if T > max(dfs) + 1e-6]
    known = bootstrap_from_swaps_seeded(dfs, swap_pillars, ay=1.0)   # 30/360 fixed
    return known


def bootstrap_from_swaps_seeded(dfs, pillars, ay):
    dfs = dict(dfs)
    for T, r in sorted(pillars):
        r = r / 100.0
        pay_ts = [float(k) for k in range(1, int(np.floor(T - 1e-9)) + 1)]
        if T - pay_ts[-1] > 1e-6:
            pay_ts.append(T)
        accr = np.diff([0.0] + pay_ts) * ay
        kmax = max(dfs)

        def interp2(t, df_T):
            xs = sorted(list(dfs) + [T]); ln = {**{x: np.log(dfs[x]) for x in dfs}, T: np.log(df_T)}
            return float(np.exp(np.interp(t, np.array(xs), np.array([ln[x] for x in xs]))))

        def gap(df_T):
            pv = sum(a * interp2(t, df_T) for a, t in zip(accr[:-1], pay_ts[:-1]))
            pv += accr[-1] * df_T
            return r * pv + df_T - 1.0

        dfs[T] = brentq(gap, 1e-8, 1.5)
    return dfs


def forwards_from_dfs(dfs, grid=GRID):
    ts = np.array(sorted(dfs))
    lnp = np.array([-np.log(dfs[t]) for t in ts])
    spline = PchipInterpolator(ts, lnp)
    f = spline.derivative()(grid)
    return np.clip(f, -0.02, 0.30)


# ---------------------------------------------------------------- vol sheets
def _label_to_years(s):
    s = str(s).strip()
    for suf, mult in (("Yr", 1.0), ("Mo", 1 / 12.0)):
        if s.endswith(suf):
            return float(s[: -len(suf)]) * mult
    raise ValueError(s)


def parse_capv(ws):
    rows = list(ws.values)
    hdr_i = next(i for i, r in enumerate(rows) if r and r[0] == "Tenor")
    strikes = rows[hdr_i][1:]
    out = []
    for r in rows[hdr_i + 1:]:
        if not r or r[0] is None:
            continue
        T = _label_to_years(r[0])
        for k, v in zip(strikes, r[1:]):
            if v is None or k is None:
                continue
            key = "ATM" if str(k).strip().upper() == "ATM" else float(k)
            out.append((T, key, float(v)))
    return out


def parse_swpt(ws):
    rows = list(ws.values)
    hdr_i = next(i for i, r in enumerate(rows) if r and r[0] == "Expiry")
    tenors = [_label_to_years(x) for x in rows[hdr_i][1:] if x is not None]
    out = []
    for r in rows[hdr_i + 1:]:
        if not r or r[0] is None:
            continue
        e = _label_to_years(r[0])
        for tn, v in zip(tenors, r[1:]):
            if v is not None:
                out.append((e, tn, float(v)))
    return out


# ---------------------------------------------------------------- main
def main():
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)

    # --- curves
    estr  = parse_curve_sheet(wb["EUR_OIS_ESTR_CRV_300626"])
    eur6m_ws = wb["EUR_EURIBOR6M_CRV_300626"]
    eur6m = parse_curve_sheet(eur6m_ws)
    fras  = parse_fras(eur6m_ws)
    sonia = parse_curve_sheet(wb["GBP_OIS_SONIA_CRV_300626"])
    sofr  = parse_curve_sheet(wb["USD_SOFR_CRV_300626"])
    ff_sw, ff_basis = parse_ff_basis_sheet(wb["USD_FF_BASIS_CRV_300626"])

    def ois_curve(parsed, act360=True):
        pillars = parsed["cash"] + parsed["swaps"]
        # de-dup by tenor (cash 1DY + swaps)
        seen = {}
        for T, r in pillars:
            seen[T] = r
        return bootstrap_ois(sorted(seen.items()), act360=act360)

    curves = {}
    curves[("eur", "ois")] = ois_curve(estr, act360=True)
    curves[("gbp", "ois")] = ois_curve(sonia, act360=False)
    curves[("usd", "ois")] = ois_curve(sofr, act360=True)

    # EURIBOR 6M projection (pseudo-DF single-curve v1)
    e6_dfs = bootstrap_euribor6m(eur6m["cash"], fras, eur6m["swaps"])
    curves[("eur", "key")] = e6_dfs
    curves[("gbp", "key")] = curves[("gbp", "ois")]          # SONIA single-curve

    # USD: the caplets are on SOFR, so they project on SOFR (single-curve, like GBP).
    # The FF composite (outright FF OIS + FF-SOFR basis) is kept as forward_ff.csv for
    # reference only -- a SOFR caplet has no exposure to the FF-SOFR basis.
    curves[("usd", "key")] = curves[("usd", "ois")]
    sofr_par = dict(sofr["swaps"])
    ff_pillars = list(ff_sw)
    for T, b in ff_basis:
        if T in sofr_par:
            ff_pillars.append((T, sofr_par[T] + b / 100.0))
    ff_curve = bootstrap_ois(sorted(set(ff_pillars)), act360=True)

    # --- emit
    for ccy in ("eur", "gbp", "usd"):
        d = os.path.join(ROOT, "data", ccy)
        os.makedirs(d, exist_ok=True)
        for role, fname in (("ois", "forward_ois.csv"), ("key", "forward_key_rate.csv")):
            f = forwards_from_dfs(curves[(ccy, role)])
            with open(os.path.join(d, fname), "w") as fh:
                fh.write("time_to_maturity,forward_rate\n")
                for t, v in zip(GRID, f):
                    fh.write(f"{t:.6f},{v:.8f}\n")     # already decimal
        if ccy == "usd":                               # pre-Oct-2026 USD projection curve
            with open(os.path.join(d, "forward_ff.csv"), "w") as fh:
                fh.write("time_to_maturity,forward_rate\n")
                for t, v in zip(GRID, forwards_from_dfs(ff_curve)):
                    fh.write(f"{t:.6f},{v:.8f}\n")

        capv = parse_capv(wb[f"{ccy.upper()}_VCUB_CAPV_300626"])
        with open(os.path.join(d, "cap_vol_surface.csv"), "w") as fh:
            fh.write("maturity_yrs,strike,vol_bp\n")
            for T, k, v in capv:
                fh.write(f"{T:.4f},{k if k == 'ATM' else f'{k:.4f}'},{v:.4f}\n")

        swpt = parse_swpt(wb[f"{ccy.upper()}_VCUB_STRIKE_300626"])
        with open(os.path.join(d, "swaption_atm.csv"), "w") as fh:
            fh.write("expiry_yrs,tenor_yrs,vol_bp\n")
            for e, tn, v in swpt:
                fh.write(f"{e:.4f},{tn:.4f},{v:.4f}\n")

        print(f"[{ccy}] wrote forward_ois / forward_key_rate / cap_vol_surface "
              f"({len(capv)} pts) / swaption_atm ({len(swpt)} pts)")

    # quick sanity prints
    for (ccy, role), dfs in sorted(curves.items()):
        f = forwards_from_dfs(dfs)
        print(f"  {ccy}/{role}: f(3M)={f[2]*100:.3f}%  f(2Y)={f[23]*100:.3f}%  "
              f"f(10Y)={f[119]*100:.3f}%  f(30Y)={f[-1]*100:.3f}%")


if __name__ == "__main__":
    main()
