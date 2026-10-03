# -*- coding: utf-8 -*-
"""Regenerate the EUR paper figures from a calibration run, scripted.

Replaces the interactive `signum` notebook route for fig_eur_surface3d.png and
fig_eur_review.png so the figures can be rebuilt from a results CSV without a
browser. Both are labelled with the Monte-Carlo time step they came from.

Run:  <python> daily_figures.py --tag d252
"""
import os, argparse
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__)); DM = os.path.dirname(HERE)
RUNS, FIG = os.path.join(DM, "runs"), os.path.join(DM, "figures")

ap = argparse.ArgumentParser()
ap.add_argument("--ccy", default="eur")
ap.add_argument("--tag", default="d252")
ap.add_argument("--step", default="dt=1/252 (daily)")
A = ap.parse_args()

res = pd.read_csv(os.path.join(RUNS, f"results_{A.ccy}_HWH_{A.tag}.csv"))
res["strike"] = res.strike.astype(float)
if "F_mkt" in res:                                         # eval_daily output: ATM/core on the market forward
    res["F"] = res["F_mkt"]
err = (res.model_vol_bp - res.market_vol_bp).values
core = (res.strike - res.F).abs().values < 0.02
rmse = float(np.sqrt((err ** 2).mean()))
crmse = float(np.sqrt((err[core] ** 2).mean()))
U = A.ccy.upper()

pm = res.pivot_table(index="strike", columns="t_fix", values="market_vol_bp")
po = res.pivot_table(index="strike", columns="t_fix", values="model_vol_bp")
mats = pm.columns.values.astype(float)
strikes = pm.index.values.astype(float)
X, Y = np.meshgrid(mats, np.arange(len(strikes)))          # uniform strike index

# ------------------------------------------------------------------ 3D surface
fig = plt.figure(figsize=(11, 7.5))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, po.values, cmap="viridis", alpha=0.9, linewidth=0,
                antialiased=True, rstride=1, cstride=1)
ax.plot_wireframe(X, Y, pm.values, color="k", linewidth=0.6, alpha=0.65,
                  rstride=1, cstride=1)
ax.set_xlabel("maturity $t_{fix}$ (Y)")
ax.set_ylabel("strike index")
ax.set_zlabel("normal vol (bp)")
step = max(1, len(strikes) // 8)
ax.set_yticks(np.arange(len(strikes))[::step])
ax.set_yticklabels([f"{s*100:.1f}%" for s in strikes[::step]], fontsize=7)
ax.set_title(f"{U} caplet normal-volatility surface — model (coloured) vs market (black mesh)\n"
             f"{A.step};  RMSE {rmse:.1f} bp, liquid core {crmse:.1f} bp", fontsize=11)
ax.view_init(elev=24, azim=-58)
fig.tight_layout()
p3 = os.path.join(FIG, f"fig_{A.ccy}_surface3d.png")
fig.savefig(p3, dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote", os.path.basename(p3))

# ------------------------------------------------------------------ review panel
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

atm_i = [np.where(res.t_fix.values == t)[0][
             np.argmin(np.abs(res.strike.values[res.t_fix.values == t]
                              - res.F.values[res.t_fix.values == t]))]
         for t in np.unique(res.t_fix.values)]
a = axes[0, 0]
a.plot(res.t_fix.values[atm_i], res.market_vol_bp.values[atm_i], "ko-", ms=4, label="market")
a.plot(res.t_fix.values[atm_i], res.model_vol_bp.values[atm_i], "r.--", ms=8, label="model")
a.set_xlabel("$t_{fix}$ (Y)"); a.set_ylabel("ATM normal vol (bp)")
a.set_title(f"ATM term structure (MAE {np.abs(err[atm_i]).mean():.2f} bp)")
a.legend(); a.grid(alpha=0.3)

a = axes[0, 1]
for T in [t for t in (1.0, 2.0, 5.0, 10.0) if t in set(res.t_fix.values)]:
    g = res[res.t_fix == T].sort_values("strike")
    ln, = a.plot(g.strike * 100, g.market_vol_bp, "o-", ms=3, lw=1, label=f"{T:.0f}Y mkt")
    a.plot(g.strike * 100, g.model_vol_bp, "--", lw=1.2, color=ln.get_color())
a.set_xlabel("strike (%)"); a.set_ylabel("normal vol (bp)")
a.set_title("smile slices (solid = market, dashed = model)")
a.legend(fontsize=7); a.grid(alpha=0.3)

a = axes[1, 0]
pe = res.pivot_table(index="strike", columns="t_fix", values="model_vol_bp") - pm
v = float(np.nanmax(np.abs(pe.values)))
im = a.pcolormesh(mats, np.arange(len(strikes)), pe.values, cmap="RdBu_r", vmin=-v, vmax=v)
a.set_yticks(np.arange(len(strikes))[::step])
a.set_yticklabels([f"{s*100:.1f}%" for s in strikes[::step]], fontsize=7)
a.set_xlabel("$t_{fix}$ (Y)"); a.set_ylabel("strike")
a.set_title("signed residual, model - market (bp)")
fig.colorbar(im, ax=a)

a = axes[1, 1]
a.hist(err[core], bins=30, color="tab:green", alpha=0.75, label=f"liquid core (n={core.sum()})")
a.hist(err[~core], bins=30, color="tab:red", alpha=0.45, label=f"wings (n={(~core).sum()})")
a.axvline(0, color="k", lw=0.8)
a.set_xlabel("model - market (bp)"); a.set_ylabel("count")
a.set_title(f"residuals: overall RMSE {rmse:.2f} bp, core {crmse:.2f} bp")
a.legend(fontsize=8); a.grid(alpha=0.3)

fig.suptitle(f"{U} HWH calibration — {A.step}, {len(res)} caplets", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
pr = os.path.join(FIG, f"fig_{A.ccy}_review.png")
fig.savefig(pr, dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote", os.path.basename(pr))
print(f"{U}: RMSE {rmse:.2f}  core {crmse:.2f}  ATM MAE {np.abs(err[atm_i]).mean():.2f}")
