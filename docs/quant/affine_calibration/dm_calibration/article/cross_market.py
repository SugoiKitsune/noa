# -*- coding: utf-8 -*-
"""Cross-market robustness: compare the calibrated HWH (QE, vega) fit across
EUR / USD / GBP. Reads runs/results_<ccy>_HWH_qe.csv for each available market
and writes:
  - a markdown fit table (article/results/cross_market.md)
  - a comparison figure (figures/fig_cross_market.png): ATM term structure and
    the 5Y smile, market vs model, for each market.
Only markets whose results exist are included.

Run:  <python> cross_market.py --tag qe
      <python> cross_market.py --tags eur=term_eval usd=rfr_lm_eval gbp=rfr_lm_eval
The headline metrics for the paper come from eval_daily.py; this script is for the
figure and a quick table. ATM/liquid use the market forward (F_mkt) when the results
file carries it (eval_daily output), otherwise the model forward.
"""
import os, sys, argparse
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); DM = os.path.dirname(HERE)
RUNS = os.path.join(DM, "runs"); FIG = os.path.join(DM, "figures")
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
MKTS = [("eur", "EUR", "6M Euribor (IBOR)"), ("usd", "USD", "3M SOFR (RFR)"),
        ("gbp", "GBP", "3M SONIA (RFR)")]

_ap = argparse.ArgumentParser()
_ap.add_argument("--tag", default="qe", help="results_<ccy>_HWH_<tag>.csv to read")
_ap.add_argument("--tags", nargs="*", default=[], help="per-market overrides, e.g. eur=term_eval")
_A = _ap.parse_args()
_TAGS = dict(t.split("=", 1) for t in _A.tags)

rows, present = [], []
for ccy, LBL, underlying in MKTS:
    p = os.path.join(RUNS, f"results_{ccy}_HWH_{_TAGS.get(ccy, _A.tag)}.csv")
    if not os.path.exists(p):
        continue
    r = pd.read_csv(p); r["strike"] = r.strike.astype(float)
    if "F_mkt" in r:
        r["F"] = r["F_mkt"]
    err = r.model_vol_bp - r.market_vol_bp
    # ATM = the single strike nearest the forward at each maturity. A fixed band
    # (e.g. |K-F|<30bp) can catch several strikes or none depending on the ladder,
    # which made GBP disagree with the per-caplet metric.
    atm = r.loc[r.groupby("t_fix").apply(
        lambda g: (g.strike - g.F).abs().idxmin(), include_groups=False)]
    atm = atm.model_vol_bp - atm.market_vol_bp
    liq = r[(r.strike - r.F).abs() < 0.02]
    rmse = lambda s: float(np.sqrt((s ** 2).mean()))
    rows.append({
        "ccy": LBL, "underlying": underlying, "n": len(r),
        "rmse": round(rmse(err), 1),
        "atm_mae": round(float(atm.abs().mean()), 1),
        "liquid": round(rmse(liq.model_vol_bp - liq.market_vol_bp), 1),
        "rel_pct": round(float(np.sqrt(((err / r.market_vol_bp) ** 2).mean()) * 100), 1),
    })
    present.append((ccy, LBL, r))

# ---- markdown table
lines = ["# Cross-market robustness — HWH (QE, vega-weighted)", "",
         "| market | underlying | caplets | RMSE (bp) | ATM MAE (bp) | liquid RMSE (bp) | rel % |",
         "|---|---|---|---|---|---|---|"]
for x in rows:
    lines.append(f"| {x['ccy']} | {x['underlying']} | {x['n']} | {x['rmse']} | "
                 f"{x['atm_mae']} | {x['liquid']} | {x['rel_pct']} |")
open(os.path.join(OUT, "cross_market.md"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))

# ---- comparison figure: ATM term structure + 5Y smile per market
if present:
    n = len(present)
    fig, axs = plt.subplots(2, n, figsize=(5 * n, 8), squeeze=False)
    for j, (ccy, LBL, r) in enumerate(present):
        a = r.loc[r.groupby("t_fix").apply(
            lambda g: (g.strike - g.F).abs().idxmin(), include_groups=False)].set_index("t_fix")
        a = a.rename(columns={"market_vol_bp": "mkt", "model_vol_bp": "mdl"})
        axs[0][j].plot(a.index, a.mkt, "k-o", ms=3, label="market")
        axs[0][j].plot(a.index, a.mdl, "r--s", ms=3, label="model")
        axs[0][j].set_title(f"{LBL} — ATM term structure"); axs[0][j].set_xlabel("maturity (Y)")
        axs[0][j].set_ylabel("vol (bp)"); axs[0][j].legend(fontsize=8); axs[0][j].grid(alpha=.3)
        av = sorted(r.t_fix.unique()); T = min(av, key=lambda x: abs(x - 5.0))
        s = r[np.isclose(r.t_fix, T)].sort_values("strike"); F = float(s.F.iloc[0]) * 100
        axs[1][j].plot(s.strike * 100, s.market_vol_bp, "k-o", ms=3, label="market")
        axs[1][j].plot(s.strike * 100, s.model_vol_bp, "r--s", ms=3, label="model")
        axs[1][j].axvline(F, color="grey", ls=":")
        axs[1][j].set_title(f"{LBL} — smile @ {T:.0f}Y (F~{F:.1f}%)")
        axs[1][j].set_xlabel("strike (%)"); axs[1][j].set_ylabel("vol (bp)")
        axs[1][j].legend(fontsize=8); axs[1][j].grid(alpha=.3)
    fig.suptitle("Cross-market HWH calibration (daily grid) — market vs model", fontsize=13, fontweight="bold")
    plt.tight_layout(); p = os.path.join(FIG, "fig_cross_market.png")
    plt.savefig(p, dpi=110); print("\nwrote", os.path.basename(p))
