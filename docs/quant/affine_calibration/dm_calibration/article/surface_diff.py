# -*- coding: utf-8 -*-
"""Full caplet-surface diff: where does the calibrated model fit / miss?
Reads runs/results_<ccy>_<label>.csv (model_vol_bp vs market_vol_bp over the WHOLE
surface) and prints: overall RMSE, RMSE per maturity, RMSE per strike, a text
residual heatmap (model - market, bp), and the worst cells.

Run:  <python> surface_diff.py [ccy=eur] [label=HWH_qe]
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), "runs")
CCY = sys.argv[1] if len(sys.argv) > 1 else "eur"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "HWH_qe"

df = pd.read_csv(os.path.join(RUNS, f"results_{CCY}_{LABEL}.csv"))
df["strike"] = df.strike.astype(float)
df["err"] = df.model_vol_bp - df.market_vol_bp
rmse = lambda s: float(np.sqrt((s ** 2).mean())) if len(s) else float("nan")

print(f"### {CCY.upper()} / {LABEL} : {len(df)} caplets")
print(f"overall  RMSE={rmse(df.err):5.1f}  MAE={df.err.abs().mean():5.1f}  "
      f"bias(mean err)={df.err.mean():+5.1f}  rel-RMSE={np.sqrt(((df.err/df.market_vol_bp)**2).mean())*100:4.1f}%")

print("\n-- RMSE per maturity (bias in parens) --")
for T, g in df.groupby("t_fix"):
    print(f"  {T:5.2f}Y : RMSE {rmse(g.err):5.1f}  bias {g.err.mean():+6.1f}  "
          f"(mkt {g.market_vol_bp.min():.0f}-{g.market_vol_bp.max():.0f})")

print("\n-- RMSE per strike (bias in parens) --")
for K, g in df.groupby("strike"):
    print(f"  K={K*100:6.2f}% : RMSE {rmse(g.err):5.1f}  bias {g.err.mean():+6.1f}")

# text heatmap: rows=strike, cols=maturity, cell=model-market (bp)
piv = df.pivot_table(index="strike", columns="t_fix", values="err")
mats = piv.columns.values
sub = [m for m in mats if m in (0.5, 1, 2, 3, 5, 7, 10)] or list(mats[::3])
print("\n-- residual heatmap  model-market (bp)   [+ model too high, - too low] --")
print("  strike\\T  " + "".join(f"{m:6.1f}" for m in sub))
for K in piv.index:
    row = "".join(f"{piv.loc[K, m]:6.0f}" if not np.isnan(piv.loc[K, m]) else "     ." for m in sub)
    print(f"  {K*100:7.2f}%  {row}")

print("\n-- 10 worst cells (|error|) --")
w = df.reindex(df.err.abs().sort_values(ascending=False).index).head(10)
for _, r in w.iterrows():
    print(f"  T={r.t_fix:5.2f}Y K={r.strike*100:6.2f}%  mkt {r.market_vol_bp:6.1f} "
          f"model {r.model_vol_bp:6.1f}  ({r.err:+.0f})")

# where the wall lives: OTM (K > F+1%) high-strike residual
otm = df[df.strike * 100 > 100 * df.F + 1.0]
print(f"\n-- high-strike wall (K > F+1%): {len(otm)} pts, "
      f"RMSE {rmse(otm.err):.1f}, bias {otm.err.mean():+.1f} (negative = model under-reaches wall)")
