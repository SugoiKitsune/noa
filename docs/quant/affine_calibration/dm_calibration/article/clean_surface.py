# -*- coding: utf-8 -*-
"""Re-price the saved calibration at a HIGH path count and rewrite the results CSV +
figures, so the plotted surface is free of MC noise at the extreme corners (short-TTM
/ far-OTM), where too-few-paths otherwise 'cuts down' cells. Params are unchanged;
only the plotting eval uses more paths.

Run:  <ml-python> clean_surface.py [ccy=eur] [label=HWH_qe] [paths=262144]
"""
import os, sys
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__)); DM = os.path.dirname(HERE)
sys.path.insert(0, DM); sys.path.insert(0, os.path.join(DM, "..", "scripts"))
from dm_calibrate import (make_dm_space, make_eval_loss, load_market,
                          load_curve_vec, TENOR, RUNS)                       # noqa: E402
import dm_figures as Fg                                                      # noqa: E402

CCY = sys.argv[1] if len(sys.argv) > 1 else "eur"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "HWH_qe"
NPATHS = int(sys.argv[3]) if len(sys.argv) > 3 else 262144
MODEL = LABEL[:-3] if LABEL.endswith("_qe") else LABEL
SIM = "qe" if LABEL.endswith("_qe") else "euler"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tau = TENOR[CCY]; dt = 0.03; max_mat = 10.0

caplets = load_market(CCY, dev, max_mat=max_mat)
T_max = float(caplets.t_fix.max()) + tau + 0.05
timeline = torch.linspace(0, T_max, int(round(T_max / dt)) + 1, device=dev)
f_key = torch.tensor(load_curve_vec(CCY, "key", timeline), dtype=torch.float32, device=dev)
f_ois = torch.tensor(load_curve_vec(CCY, "ois", timeline), dtype=torch.float32, device=dev)
space = make_dm_space(dev)
_, predict, mkt = make_eval_loss(space, timeline, f_key, f_ois, caplets, dev, tau, model=MODEL, sim=SIM)

d = torch.load(os.path.join(RUNS, "calibration_weights", f"{CCY}_{LABEL}.pt"), map_location=dev)
torch.manual_seed(0)
mv, F = predict(d["p"].to(dev), NPATHS, 7)
mv = mv.cpu().numpy()
res = caplets[["t_fix", "strike"]].copy()
res["market_vol_bp"] = mkt.cpu().numpy(); res["model_vol_bp"] = mv; res["F"] = F.cpu().numpy()
rmse = float(np.sqrt(((res.model_vol_bp - res.market_vol_bp) ** 2).mean()))
out = os.path.join(RUNS, f"results_{CCY}_{LABEL}.csv")
res.to_csv(out, index=False)
print(f"rewrote {os.path.basename(out)} @ {NPATHS} paths : RMSE {rmse:.1f} bp")

# regenerate figures from the clean surface
print("fit    ->", os.path.basename(Fg.fig_fit(res, CCY)))
print("overlay->", os.path.basename(Fg.fig_overlay(res, CCY)))
print("smile  ->", os.path.basename(Fg.fig_smile(CCY)))
print("mktsurf->", os.path.basename(Fg.fig_market_surface(CCY)))
