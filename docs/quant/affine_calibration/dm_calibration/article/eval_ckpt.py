# -*- coding: utf-8 -*-
"""Diagnostic: evaluate a saved calibration at increasing MC path counts to separate
implied-vol INVERSION NOISE (deep-ITM/short-T, near-zero vega) from real model misfit.
If the short-end model vol converges toward market as paths rise -> it was noise.

Run:  <ml-python> eval_ckpt.py [ccy=eur] [label=HWH_qe]
"""
import os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
DM = os.path.dirname(HERE)
sys.path.insert(0, DM); sys.path.insert(0, os.path.join(DM, "..", "scripts"))
from dm_calibrate import (make_dm_space, make_eval_loss, load_market,
                          load_curve_vec, TENOR, RUNS)                       # noqa: E402

CCY = sys.argv[1] if len(sys.argv) > 1 else "eur"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "HWH_qe"
MODEL = LABEL[:-3] if LABEL.endswith("_qe") else LABEL
SIM = "qe" if LABEL.endswith("_qe") else "euler"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tau = TENOR[CCY]; dt = 0.03; max_mat = 10.0

caplets = load_market(CCY, dev, max_mat=max_mat)
T_max = float(caplets.t_fix.max()) + tau + 0.05
n = int(round(T_max / dt)) + 1
timeline = torch.linspace(0, T_max, n, device=dev)
f_key = torch.tensor(load_curve_vec(CCY, "key", timeline), dtype=torch.float32, device=dev)
f_ois = torch.tensor(load_curve_vec(CCY, "ois", timeline), dtype=torch.float32, device=dev)
space = make_dm_space(dev)
_, predict, mkt = make_eval_loss(space, timeline, f_key, f_ois, caplets, dev, tau, model=MODEL, sim=SIM)

d = torch.load(os.path.join(RUNS, "calibration_weights", f"{CCY}_{LABEL}.pt"), map_location=dev)
p = d["p"].to(dev)
tf = caplets.t_fix.values; K = caplets.strike.values.astype(float); mk = mkt.cpu().numpy()

print(f"{CCY}/{LABEL}  saved RMSE={d.get('rmse_bp'):.1f}bp   (dt={dt}, max_mat={max_mat})")
rows_short = [(i, tf[i], K[i]) for i in range(len(tf)) if abs(tf[i] - 0.5) < 1e-6]
rows_short.sort(key=lambda r: r[2])
print("\nShort end (0.5Y): model vol (bp) at rising path counts vs market")
hdr = "  K%     mkt " + " ".join(f"{npc//1000}k".rjust(7) for npc in [16384, 65536, 262144])
print(hdr)
mvs = {}
for npaths in [16384, 65536, 262144]:
    torch.manual_seed(0)
    mv, F = predict(p, npaths, 7)
    mvs[npaths] = mv.cpu().numpy()
for i, t, k in rows_short:
    line = f"  {k*100:6.2f} {mk[i]:5.0f} " + " ".join(f"{mvs[npc][i]:7.0f}" for npc in [16384, 65536, 262144])
    print(line)

# overall + short-end RMSE at each path count (does more MC reduce the short-end error?)
print("\nRMSE (bp): overall  |  0.5Y-only  |  0.5Y with |K-F|<2% (well-vega'd)")
for npaths in [16384, 65536, 262144]:
    e = mvs[npaths] - mk
    sh = np.array([abs(tf[i] - 0.5) < 1e-6 for i in range(len(tf))])
    Fnow = predict(p, npaths, 7)[1].cpu().numpy()
    liq = sh & (np.abs(K - Fnow) < 0.02)
    r = lambda m: np.sqrt((e[m]**2).mean()) if m.sum() else float("nan")
    print(f"  {npaths//1000:4d}k :  {np.sqrt((e**2).mean()):5.1f}   |   {r(sh):5.1f}    |   {r(liq):5.1f}")
