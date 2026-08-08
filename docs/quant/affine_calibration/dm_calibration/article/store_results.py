# -*- coding: utf-8 -*-
"""Store article-ready DERIVED results from the DM calibration runs.

Writes committable text artifacts to article/results/:
  - results_<ccy>.md     human-readable ladder RMSE table, error metrics, calibrated params
  - metrics_<ccy>.json   machine-readable version (for building LaTeX tables)
  - calibrated_<ccy>.json calibrated model parameters (model output, publishable)
and copies the figures into article/results/figures/ (gitignored, regenerable).

Deliberately DOES NOT copy the per-caplet market vols: those are Bloomberg-derived
quotes and must not land in committed files. Only aggregate/derived numbers and
model-output parameters (standard to publish) are stored.

Run:  <ml-python> store_results.py [ccy=eur]
"""
import os, sys, json, glob, shutil
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))      # .../dm_calibration/article
DM = os.path.dirname(HERE)                             # .../dm_calibration
RUNS = os.path.join(DM, "runs")
FIGS = os.path.join(DM, "figures")
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
OUTFIG = os.path.join(OUT, "figures"); os.makedirs(OUTFIG, exist_ok=True)

sys.path.insert(0, DM)
sys.path.insert(0, os.path.join(DM, "..", "scripts"))
from dm_calibrate import make_dm_space, constrain, LADDER      # noqa: E402

CCY = sys.argv[1] if len(sys.argv) > 1 else "eur"


def label_to_model(label):
    return label[:-3] if label.endswith("_qe") else label


def decode_params(label):
    """Decode the 0-1 param vector in the .pt into human-readable model parameters."""
    pt = os.path.join(RUNS, "calibration_weights", f"{CCY}_{label}.pt")
    if not os.path.exists(pt):
        return None
    d = torch.load(pt, map_location="cpu")
    space = make_dm_space(torch.device("cpu"))
    u = constrain(label_to_model(label), space["unpack"](d["p"]))
    th1, th2, kf, ks, wf, eps, lam, gam, xi, r1, r5, r10, v0 = u
    nodes = space["theta_nodes"]
    nodes = nodes.tolist() if hasattr(nodes, "tolist") else list(nodes)
    def vol_bp(t):                                     # variance level -> normal vol (bp)
        a = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        return [round(float(x), 1) for x in np.sqrt(np.clip(a, 0, None)) * 1e4]
    f = lambda z: round(float(z), 5)
    return {
        "sim": d.get("sim"), "rmse_bp": d.get("rmse_bp"),
        "theta_nodes_yr": [round(float(x), 2) for x in nodes],
        "theta1_vol_bp": vol_bp(th1), "theta2_vol_bp": vol_bp(th2),
        "kappa_fast": f(kf), "kappa_slow": f(ks), "w_fast": f(wf),
        "eps_volvol": f(eps), "lambda_x": f(lam), "gamma_hw": f(gam), "xi_hw": f(xi),
        "rho_1y": f(r1), "rho_5y": f(r5), "rho_10y": f(r10),
        "v0": f(v0 if not hasattr(v0, "item") else v0.item()),
    }


def metrics(df):
    df = df.copy(); df["err"] = df.model_vol_bp - df.market_vol_bp
    rmse = lambda s: (round(float(np.sqrt((s ** 2).mean())), 1) if len(s) else None)
    buckets = [(0, 1, "0-1Y"), (1, 3, "1-3Y"), (3, 7, "3-7Y"), (7, 99, "7Y+")]
    atm = df[(df.strike - df.F).abs() < 0.0025]
    otm = df[(df.strike - df.F).abs() >= 0.0025]
    return {
        "n_caplets": int(len(df)),
        "rmse_bp": rmse(df.err), "mae_bp": round(float(df.err.abs().mean()), 1),
        "rel_rmse_pct": round(float(np.sqrt(((df.err / df.market_vol_bp) ** 2).mean()) * 100), 1),
        "by_maturity": {lab: rmse(df[(df.t_fix >= a) & (df.t_fix < b)].err)
                        for a, b, lab in buckets if ((df.t_fix >= a) & (df.t_fix < b)).any()},
        "atm_rmse_bp": rmse(atm.err), "otm_rmse_bp": rmse(otm.err),
    }


# ---- collect per-model results
rows, params = {}, {}
for path in sorted(glob.glob(os.path.join(RUNS, f"results_{CCY}_*.csv"))):
    label = os.path.basename(path)[len(f"results_{CCY}_"):-4]
    df = pd.read_csv(path); df["strike"] = df.strike.astype(float)
    rows[label] = metrics(df)
    p = decode_params(label)
    if p:
        params[label] = p

# best model = lowest RMSE (typically HWH_qe)
best = min(rows, key=lambda k: rows[k]["rmse_bp"]) if rows else None

payload = {"ccy": CCY, "best_label": best, "models": rows, "params": params}
json.dump(payload, open(os.path.join(OUT, f"metrics_{CCY}.json"), "w"), indent=2)
if best in params:
    json.dump(params[best], open(os.path.join(OUT, f"calibrated_{CCY}.json"), "w"), indent=2)

# ---- human-readable markdown
def order_key(lab):
    m = label_to_model(lab)
    return (LADDER.index(m) if m in LADDER else 99, lab)

lines = [f"# {CCY.upper()} calibration results (article data store)", "",
         "*Derived / model-output numbers only — raw Bloomberg vol quotes are not stored here.*",
         "", "## Nested-model ladder — vol-RMSE (bp)", "",
         "| model | sim | RMSE | 0-1Y | 1-3Y | 3-7Y | 7Y+ | ATM | OTM | rel% |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for lab in sorted(rows, key=order_key):
    m = rows[lab]; bm = m["by_maturity"]
    sim = params.get(lab, {}).get("sim", "euler")
    g = lambda k: bm.get(k, "-")
    lines.append(f"| {label_to_model(lab)} | {sim} | {m['rmse_bp']} | {g('0-1Y')} | {g('1-3Y')} | "
                 f"{g('3-7Y')} | {g('7Y+')} | {m['atm_rmse_bp']} | {m['otm_rmse_bp']} | {m['rel_rmse_pct']} |")

if best in params:
    p = params[best]
    lines += ["", f"## Calibrated parameters — {label_to_model(best)} ({p['sim']}, best fit {p['rmse_bp']} bp)", "",
              "*√θ is the instantaneous OU-driver scale (bp/√yr); the caplet implied vol is lower "
              "after mean-reversion (λ) and accrual averaging.*", "",
              "| θ node (yr) | " + " | ".join(str(x) for x in p["theta_nodes_yr"]) + " |",
              "|---|" + "---|" * len(p["theta_nodes_yr"]),
              "| √θ₁ (bp/√yr) | " + " | ".join(str(x) for x in p["theta1_vol_bp"]) + " |",
              "| √θ₂ (bp/√yr) | " + " | ".join(str(x) for x in p["theta2_vol_bp"]) + " |",
              "",
              "| κ_fast | κ_slow | w_fast | ε (vol-of-vol) | λ_x | γ_HW | ξ_HW | v0 |",
              "|---|---|---|---|---|---|---|---|",
              f"| {p['kappa_fast']} | {p['kappa_slow']} | {p['w_fast']} | {p['eps_volvol']} | "
              f"{p['lambda_x']} | {p['gamma_hw']} | {p['xi_hw']} | {p['v0']} |",
              "",
              "| ρ(1Y) | ρ(5Y) | ρ(10Y) | (rate–vol correlation, drives skew) |",
              "|---|---|---|---|",
              f"| {p['rho_1y']} | {p['rho_5y']} | {p['rho_10y']} |  |"]

# ---- copy figures (local, gitignored)
copied = []
for f in sorted(glob.glob(os.path.join(FIGS, f"*{CCY}*"))):
    shutil.copy2(f, os.path.join(OUTFIG, os.path.basename(f)))
    copied.append(os.path.basename(f))
lines += ["", "## Figures (in results/figures/, regenerable)", ""] + [f"- `{c}`" for c in copied]

open(os.path.join(OUT, f"results_{CCY}.md"), "w", encoding="utf-8").write("\n".join(lines))
print(f"stored: results_{CCY}.md, metrics_{CCY}.json, calibrated_{CCY}.json ; {len(copied)} figures")
print(f"best = {best}  ({rows[best]['rmse_bp']} bp)" if best else "no results found")
