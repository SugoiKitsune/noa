# -*- coding: utf-8 -*-
"""DM article figures + metrics.

Market figures (no GPU, from the stripped surfaces) can be generated now:
  - fig_market_surface(ccy): 3-D caplet normal-vol surface
  - fig_smile(ccy): smile-by-maturity slices
Model-vs-market figures + the relative/bucketed/ATM-OTM metrics (Roland rec 4) run
once a calibration result CSV exists (columns: t_fix, strike, market_vol_bp, model_vol_bp).

Run:  python dm_figures.py --ccy eur --market   (market-only figures, works today)
"""
import os, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

CREAM = "#F1E8CF"; GOLD = "#F0A22E"; TERRA = "#D9785A"; GREY = "#B8AE9C"
plt.rcParams.update({"figure.dpi": 150, "font.family": "serif",
                     "font.serif": ["Georgia", "DejaVu Serif"]})


def _abs_surface(ccy):
    d = pd.read_csv(os.path.join(DATA, ccy, "vol_surface.csv"))
    d = d[d.strike != "ATM"].copy()
    d["strike"] = d["strike"].astype(float)
    d["t_fix_yrs"] = d["t_fix_yrs"].astype(float)
    return d


def fig_market_surface(ccy):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    d = _abs_surface(ccy)
    piv = d.pivot_table(index="t_fix_yrs", columns="strike", values="sigma_N_bp")
    T = piv.index.values; K = piv.columns.values * 100
    K2, T2 = np.meshgrid(K, T)
    fig = plt.figure(figsize=(9, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(K2, T2, piv.values, cmap="viridis", alpha=0.9, linewidth=0, antialiased=True)
    ax.set_xlabel("Strike (%)"); ax.set_ylabel("Maturity (Y)"); ax.set_zlabel("Normal vol (bp)")
    ax.set_title(f"{ccy.upper()} stripped caplet normal-vol surface (30-Jun-2026)", fontweight="bold")
    ax.view_init(elev=22, azim=-60)
    p = os.path.join(FIG, f"fig_{ccy}_market_surface.png")
    fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_smile(ccy, mats=(1, 2, 5, 10)):
    d = _abs_surface(ccy)
    fig, ax = plt.subplots(figsize=(8, 5))
    avail = sorted(d.t_fix_yrs.unique())
    for T in mats:
        Tn = min(avail, key=lambda a: abs(a - T))
        sl = d[np.isclose(d.t_fix_yrs, Tn)].sort_values("strike")
        ax.plot(sl.strike * 100, sl.sigma_N_bp, "-o", ms=3, label=f"T≈{Tn:.1f}Y")
    ax.set_xlabel("Strike (%)"); ax.set_ylabel("Caplet normal vol (bp)")
    ax.set_title(f"{ccy.upper()} caplet smile by maturity"); ax.legend()
    p = os.path.join(FIG, f"fig_{ccy}_smile.png")
    fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# ---------------------------------------------------------- metrics (rec 4)
def error_metrics(res):
    """res: DataFrame with t_fix, strike, market_vol_bp, model_vol_bp, and F (forward).
    Returns dict of overall / bucketed / ATM-OTM absolute & relative errors."""
    e = res.model_vol_bp - res.market_vol_bp
    rel = 100 * e / res.market_vol_bp
    out = {"rmse_bp": float(np.sqrt((e ** 2).mean())),
           "rel_rmse_pct": float(np.sqrt((rel ** 2).mean()))}
    # maturity buckets
    buckets = [(0, 1), (1, 5), (5, 30)]
    for lo, hi in buckets:
        m = (res.t_fix >= lo) & (res.t_fix < hi)
        if m.any():
            out[f"rmse_bp_{lo}-{hi}Y"] = float(np.sqrt((e[m] ** 2).mean()))
    # ATM vs OTM (needs forward F)
    if "F" in res:
        atm = (res.strike.astype(float) - res.F).abs() < 0.0025
        out["rmse_bp_ATM"] = float(np.sqrt((e[atm] ** 2).mean())) if atm.any() else None
        out["rmse_bp_OTM"] = float(np.sqrt((e[~atm] ** 2).mean())) if (~atm).any() else None
    return out


def fig_overlay(res, ccy):
    """3-D market (mesh) vs model (surface) overlay — once results exist."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    piv_m = res.pivot_table(index="t_fix", columns="strike", values="market_vol_bp")
    piv_o = res.pivot_table(index="t_fix", columns="strike", values="model_vol_bp")
    K2, T2 = np.meshgrid(piv_m.columns.values * 100, piv_m.index.values)
    fig = plt.figure(figsize=(9, 6.2)); ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(K2, T2, piv_o.values, color=GOLD, alpha=0.65, linewidth=0)
    ax.plot_wireframe(K2, T2, piv_m.values, color="#333333", linewidth=0.7)
    rmse = np.sqrt(((res.model_vol_bp - res.market_vol_bp) ** 2).mean())
    ax.set_xlabel("Strike (%)"); ax.set_ylabel("Maturity (Y)"); ax.set_zlabel("Normal vol (bp)")
    ax.set_title(f"{ccy.upper()} market (mesh) vs HWH model — RMSE {rmse:.0f} bp", fontweight="bold")
    ax.view_init(elev=22, azim=-60)
    p = os.path.join(FIG, f"fig_{ccy}_overlay.png")
    fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_fit(res, ccy, mats=(1, 2, 5, 9)):
    """2-panel model-vs-market: ATM term structure + smile slices."""
    res = res.copy(); res["strike"] = res.strike.astype(float)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    atm = res[(res.strike - res.F).abs() < 0.003]
    g = atm.groupby("t_fix").agg(mkt=("market_vol_bp", "mean"),
                                 mdl=("model_vol_bp", "mean")).reset_index()
    a1.plot(g.t_fix, g.mkt, "-o", ms=4, color="#333333", label="market")
    a1.plot(g.t_fix, g.mdl, "-s", ms=4, color=GOLD, label="model")
    a1.set_xlabel("Maturity (Y)"); a1.set_ylabel("ATM normal vol (bp)")
    a1.set_title(f"{ccy.upper()} ATM term structure"); a1.legend(); a1.grid(alpha=.3)
    avail = sorted(res.t_fix.unique())
    for T in mats:
        Tn = min(avail, key=lambda x: abs(x - T))
        s = res[np.isclose(res.t_fix, Tn)].sort_values("strike")
        ln, = a2.plot(s.strike * 100, s.market_vol_bp, "-", lw=1.6, label=f"{Tn:.0f}Y")
        a2.plot(s.strike * 100, s.model_vol_bp, "--", color=ln.get_color(), lw=1.4)
    a2.set_xlabel("Strike (%)"); a2.set_ylabel("Normal vol (bp)")
    a2.set_title(f"{ccy.upper()} smile: market (—) vs model (- -)")
    a2.legend(fontsize=8, title="maturity"); a2.grid(alpha=.3)
    p = os.path.join(FIG, f"fig_{ccy}_fit.png")
    fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_loss(ccy, model="HWH"):
    """Calibration loss vs SPSA iteration (from runs/history_{ccy}_{model}.csv)."""
    h = pd.read_csv(os.path.join(HERE, "runs", f"history_{ccy}_{model}.csv"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(h["iter"], h["loss"], "-o", ms=3, color=TERRA)
    ax.set_xlabel("SPSA iteration"); ax.set_ylabel("loss (weighted vol-RMSE, bp)")
    ax.set_title(f"{ccy.upper()} {model} — calibration loss"); ax.grid(alpha=.3)
    p = os.path.join(FIG, f"fig_{ccy}_{model}_loss.png")
    fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


LADDER_MODELS = ["1F_Gauss", "2F_Gauss", "HW_detvol", "HWH_qe"]   # HWH_qe = final QE fit (16.1bp)
_DISP = {"HWH_qe": "HWH"}                                          # display label


def ladder_frame(ccy):
    """RMSE per ladder model from whatever results CSVs exist (Roland rec 2 table)."""
    rows = []
    for m in LADDER_MODELS:
        p = os.path.join(HERE, "runs", f"results_{ccy}_{m}.csv")
        if os.path.exists(p):
            r = pd.read_csv(p)
            rmse = float(np.sqrt(((r.model_vol_bp - r.market_vol_bp) ** 2).mean()))
            rows.append({"model": m, "rmse_bp": round(rmse, 1), "n_caplets": len(r)})
    return pd.DataFrame(rows)


def fig_ladder(ccy, smile_T=5):
    """3-panel nested-model ladder: RMSE bars | ATM term structure | smile — all vs market."""
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    lf = ladder_frame(ccy)
    if len(lf):
        axs[0].bar([_DISP.get(m, m) for m in lf.model], lf.rmse_bp,
                   color=[GOLD if m == "HWH_qe" else GREY for m in lf.model])
        for i, v in enumerate(lf.rmse_bp):
            axs[0].text(i, v, f"{v}", ha="center", va="bottom", fontsize=9)
    axs[0].set_ylabel("vol-RMSE (bp)"); axs[0].set_title(f"{ccy.upper()} nested-model ladder")
    axs[0].tick_params(axis="x", rotation=20)
    first = True
    for m in LADDER_MODELS:
        p = os.path.join(HERE, "runs", f"results_{ccy}_{m}.csv")
        if not os.path.exists(p):
            continue
        r = pd.read_csv(p); r["strike"] = r.strike.astype(float)
        av = sorted(r.t_fix.unique()); Tn = min(av, key=lambda x: abs(x - smile_T))
        if first:
            g0 = r[(r.strike - r.F).abs() < 0.003].groupby("t_fix").market_vol_bp.mean()
            axs[1].plot(g0.index, g0.values, "k-", lw=2, label="market")
            s0 = r[np.isclose(r.t_fix, Tn)].sort_values("strike")
            axs[2].plot(s0.strike * 100, s0.market_vol_bp, "k-", lw=2, label="market")
            first = False
        g = r[(r.strike - r.F).abs() < 0.003].groupby("t_fix").model_vol_bp.mean()
        axs[1].plot(g.index, g.values, "--", label=_DISP.get(m, m))
        s = r[np.isclose(r.t_fix, Tn)].sort_values("strike")
        axs[2].plot(s.strike * 100, s.model_vol_bp, "--", label=_DISP.get(m, m))
    axs[1].set_xlabel("Maturity (Y)"); axs[1].set_ylabel("ATM vol (bp)")
    axs[1].set_title("ATM term structure"); axs[1].legend(fontsize=8)
    axs[2].set_xlabel("Strike (%)"); axs[2].set_ylabel("vol (bp)")
    axs[2].set_title(f"smile @ {smile_T}Y (market solid)"); axs[2].legend(fontsize=8)
    for a in axs:
        a.grid(alpha=.3)
    out = os.path.join(FIG, f"fig_{ccy}_ladder.png")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ccy", default="eur")
    ap.add_argument("--market", action="store_true", help="market-only figures (no results needed)")
    ap.add_argument("--results", default=None, help="path to results CSV for overlay + metrics")
    args = ap.parse_args()
    if args.market or not args.results:
        print("wrote", os.path.relpath(fig_market_surface(args.ccy), HERE))
        print("wrote", os.path.relpath(fig_smile(args.ccy), HERE))
    if args.results:
        res = pd.read_csv(args.results)
        print("wrote", os.path.relpath(fig_overlay(res, args.ccy), HERE))
        import json; print(json.dumps(error_metrics(res), indent=2))


if __name__ == "__main__":
    main()
