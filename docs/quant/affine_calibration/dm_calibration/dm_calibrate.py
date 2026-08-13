# -*- coding: utf-8 -*-
"""DM calibration driver — per-period caplet calibration of the 2-factor lifted
CIR Hull-White-Heston model, with the nested-model ladder (Roland review rec 2).

Reuses the tested engine (scripts/simulation.py path simulator, param_space, PAVA)
but implements the DM-specific per-period caplet pricing, a direct vol-RMSE objective,
and a self-contained Adam-SPSA loop (no coupling to the RU whole-period/PV optimizer).

  NOTE: requires torch + a GPU (run in your calibration env, not miniconda base).
  The per-period pricing logic is the torch mirror of the validated numpy dm_pricer.py.

Nested-model ladder (all as restrictions of the full HWH, calibrated on the same surface):
  1F Gaussian HW   : eps=0, flat theta, single factor (k off)      -> flat smile (Prop. 1)
  2F Gaussian HW   : eps=0, flat theta, x + k Gaussian factors     -> still ~flat smile
  HW + det. vol    : eps=0, theta(t) term structure (no stoch var) -> term structure, ~flat in K
  HWH (full)       : eps>0, 2-factor lifted CIR stochastic variance -> genuine smile

Run:  python dm_calibrate.py --ccy eur            (full ladder)
      python dm_calibrate.py --ccy eur --model HWH
"""
import os, sys, json, argparse, math
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DATA = os.path.join(ROOT, "data")
SCRIPTS = os.path.join(ROOT, "docs", "quant", "affine_calibration", "scripts")
RUNS = os.path.join(HERE, "runs")
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "docs", "quant"))   # for `import pyquant` (torch_spline)

import torch
from simulation import fast_simulate_2factor, theta_to_vec, rho_to_vec
from dm_sim_qe import simulate_2factor_qe          # Andersen QE variance sampler
from param_space import make_param_space
try:
    from monotone_postprocess import apply_monotone_postprocess   # PAVA
except Exception:
    apply_monotone_postprocess = None

TENOR = {"eur": 0.5, "gbp": 0.25, "usd": 0.25}


# ----------------------------------------------------------------- data
def load_curve_vec(ccy, role, timeline):
    fname = {"ois": "forward_ois", "key": "forward_key_rate"}[role]
    d = pd.read_csv(os.path.join(DATA, ccy, f"{fname}.csv"))
    return np.interp(timeline.cpu().numpy(), d.time_to_maturity.values, d.forward_rate.values)


def load_market(ccy, device, atm_only=False, kmin=None, kmax=None, max_mat=None):
    """Return caplets DataFrame (t_fix, strike, sigma_mkt_bp) from the stripped surface.
    max_mat caps the caplet maturity (shortens the MC timeline -> much faster runs)."""
    d = pd.read_csv(os.path.join(DATA, ccy, "vol_surface.csv"))
    d = d[d.strike != "ATM"].copy() if not atm_only else d[d.strike == "ATM"].copy()
    if not atm_only:
        d["strike"] = d["strike"].astype(float)
        if kmin is not None:
            d = d[(d.strike >= kmin) & (d.strike <= kmax)]
    d = d.rename(columns={"t_fix_yrs": "t_fix", "sigma_N_bp": "sigma_mkt_bp"})
    if max_mat is not None:
        d = d[d.t_fix <= max_mat]
    return d.reset_index(drop=True)


# ----------------------------------------------------------------- pricing (torch)
def bachelier_price(F, K, sigma, texp, tau, disc):
    v = sigma * torch.sqrt(torch.clamp(texp, min=1e-8))
    d = (F - K) / torch.clamp(v, min=1e-12)
    Phi = 0.5 * (1 + torch.erf(d / math.sqrt(2)))
    phi = torch.exp(-0.5 * d * d) / math.sqrt(2 * math.pi)
    return disc * tau * ((F - K) * Phi + v * phi)


def implied_vol_bisect(pv, F, K, texp, tau, disc, lo=1e-6, hi=0.06, iters=50):
    lo = torch.full_like(pv, lo); hi = torch.full_like(pv, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pm = bachelier_price(F, K, mid, texp, tau, disc)
        hi = torch.where(pm > pv, mid, hi)
        lo = torch.where(pm <= pv, mid, lo)
    return 0.5 * (lo + hi)


def price_perperiod(key_paths, ois_paths, dt, i_fix, i_pay, tau_t, K_t):
    """Per-period caplet prices for a batch of caplets.
    key_paths/ois_paths: (n_eff, n_steps+1). i_fix/i_pay: (n_caplets,) long tensors.
    Returns model_vols_bp, F, needs texp=t_fix. Vectorised over caplets & paths.
    """
    Ikey = torch.cumsum(key_paths[:, :-1] * dt, dim=1)      # integral up to each step
    Ikey = torch.cat([torch.zeros_like(Ikey[:, :1]), Ikey], dim=1)
    Iois = torch.cumsum(ois_paths[:, :-1] * dt, dim=1)
    Iois = torch.cat([torch.zeros_like(Iois[:, :1]), Iois], dim=1)
    # A_j = (Ikey[:, i_pay]-Ikey[:, i_fix])/tau ; disc_j = exp(-Iois[:, i_pay])
    A = (Ikey[:, i_pay] - Ikey[:, i_fix]) / tau_t                 # (n_eff, n_caplets)
    disc = torch.exp(-Iois[:, i_pay])
    D = disc.mean(0)                                              # (n_caplets,)
    F = (A * disc).mean(0) / D.clamp(min=1e-12)
    pv = (torch.clamp(A - K_t, min=0.0) * tau_t * disc).mean(0)
    return pv, F, D


# ----------------------------------------------------------------- nested-model constraints
def _flat(theta):
    return torch.full_like(theta, float(theta.mean()))

def constrain(model, unpacked):
    """Apply the nested-model restriction to the unpacked 13-tuple:
    (theta1, theta2, kf, ks, wf, eps, lam, gam, xi, rho1, rho5, rho10, v0)."""
    th1, th2, kf, ks, wf, eps, lam, gam, xi, r1, r5, r10, v0 = unpacked
    if model == "HWH":
        return unpacked
    if model == "HW_detvol":          # deterministic vol term structure, single factor
        return (th1, th1, kf, ks, 1.0, 0.0, lam, gam, xi, r1, r5, r10, v0)
    if model == "2F_Gauss":           # flat vol, x + k Gaussian factors
        return (_flat(th1), _flat(th1), kf, ks, 1.0, 0.0, lam, gam, xi, r1, r5, r10, v0)
    if model == "1F_Gauss":           # flat vol, single Gaussian factor (k off)
        return (_flat(th1), _flat(th1), kf, ks, 1.0, 0.0, lam, gam, 0.0, r1, r5, r10, v0)
    raise ValueError(model)


# ----------------------------------------------------------------- objective
def make_eval_loss(space, timeline, f_key_vec, f_ois_vec, caplets, device,
                   tau, model="HWH", short_wt=1.5, sim="euler"):
    unpack = space["unpack"]; theta_nodes = space["theta_nodes"]; rho_nodes_t = space["rho_nodes_t"]
    dt = (timeline[1] - timeline[0]).item()
    t_fix = torch.tensor(caplets.t_fix.values, dtype=torch.float32, device=device)
    i_fix = torch.tensor([int(round(t / dt)) for t in caplets.t_fix.values], device=device)
    i_pay = torch.tensor([int(round((t + tau) / dt)) for t in caplets.t_fix.values], device=device)
    K_t = torch.tensor(caplets.strike.values, dtype=torch.float32, device=device)
    tau_t = torch.tensor(tau, dtype=torch.float32, device=device)
    mkt = torch.tensor(caplets.sigma_mkt_bp.values, dtype=torch.float32, device=device)
    # VEGA (Bachelier) weight: fit the liquid ATM tightly, down-weight the near-intrinsic
    # deep-ITM/-OTM wings. Those have ~0 vega, so their quoted vols are ill-defined and,
    # under equal weight, drag the smile down and sag the ATM (the ATM was ~8bp low).
    _tf = caplets.t_fix.values.astype(float)
    _K = caplets.strike.values.astype(float)
    _sig = caplets.sigma_mkt_bp.values.astype(float) / 1e4
    _Fpx = np.array([_K[_tf == t][int(np.argmin(_sig[_tf == t]))] for t in _tf])  # ATM = min-vol strike
    _d = (_Fpx - _K) / (_sig * np.sqrt(np.maximum(_tf, 0.1)))
    _w = np.maximum(np.exp(-0.5 * _d ** 2), 0.20)        # phi(d) with a 20% floor for the wings
    wt = torch.tensor(_w, dtype=torch.float32, device=device) * torch.where(t_fix < 1.0, short_wt, 1.0)
    wt = wt / wt.mean()

    def predict(p, n_paths, seed):
        u = constrain(model, unpack(p))
        th1, th2, kf, ks, wf, eps, lam, gam, xi, r1, r5, r10, v0 = u
        theta_vec = theta_to_vec(th1, theta_nodes, timeline)
        theta2_vec = theta_to_vec(th2, theta_nodes, timeline)
        rho_vec = rho_to_vec(torch.tensor([r1, r5, r10], device=device), rho_nodes_t, timeline)
        sim_fn = simulate_2factor_qe if sim == "qe" else fast_simulate_2factor
        key_paths, ois_paths, _ = sim_fn(
            n_paths, timeline, theta_vec, eps, v0 if isinstance(v0, float) else v0.item(),
            kf, ks, wf, lam, gam, xi, f_key_vec, f_ois_vec, device,
            seed=seed, rho_vx=rho_vec, antithetic=True, theta2_vec=theta2_vec)
        pv, F, D = price_perperiod(key_paths, ois_paths, dt, i_fix, i_pay, tau_t, K_t)
        model_vol = implied_vol_bisect(pv, F, K_t, t_fix, tau_t, D) * 1e4
        return model_vol, F

    def eval_loss(p, n_paths, seed):
        model_vol, _ = predict(p, n_paths, seed)
        model_vol = torch.nan_to_num(model_vol, nan=1e4, posinf=1e4, neginf=0.0)  # degenerate-price guard
        err = (model_vol - mkt)
        rmse = torch.sqrt((wt * err * err).mean())
        if not torch.isfinite(rmse):
            rmse = torch.tensor(999.0, device=device)     # keep SPSA out of NaN regions
        return rmse, model_vol

    return eval_loss, predict, mkt


# ----------------------------------------------------------------- self-contained Adam-SPSA
def adam_spsa(eval_loss, p_init, *, n_params, max_iter=1500, n_paths=16384,
              lr=3e-3, c=0.02, M=8, seed_base=42, device="cpu", ckpt_path=None):
    p = p_init.clone()
    m = torch.zeros(n_params, device=device); v = torch.zeros(n_params, device=device)
    b1, b2, eps_a = 0.9, 0.999, 1e-8
    best_loss, best_p = float("inf"), p.clone()
    hist = []
    for it in range(max_iter):
        frac = it / max(max_iter - 1, 1)
        lr_t = lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))   # cosine
        c_t = c * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * frac)))
        g = torch.zeros(n_params, device=device)
        for j in range(M):
            delta = (torch.randint(0, 2, (n_params,), device=device).float() * 2 - 1)
            lp, _ = eval_loss(torch.clamp(p + c_t * delta, 0, 1), n_paths, seed_base + it * M + j)
            lm, _ = eval_loss(torch.clamp(p - c_t * delta, 0, 1), n_paths, seed_base + 999 + it * M + j)
            g += (lp - lm) / (2 * c_t) * delta
        g /= M
        g = torch.clamp(g, -10, 10)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** (it + 1)); vh = v / (1 - b2 ** (it + 1))
        p = torch.clamp(p - lr_t * mh / (vh.sqrt() + eps_a), 0, 1)
        cur, _ = eval_loss(p, n_paths, seed_base + 10 ** 6 + it)
        cur = float(cur)
        if cur < best_loss:
            best_loss, best_p = cur, p.clone()
        if it % 10 == 0:
            hist.append((it, cur))
        if it % 50 == 0:
            print(f"    iter {it:4d}  loss={cur:.4f}  best={best_loss:.4f}")
            if ckpt_path is not None:                     # periodic best -> survive a wedge/crash
                torch.save({"p": best_p.cpu(), "loss": best_loss, "iter": it}, ckpt_path)
    return best_p, best_loss, hist


# ----------------------------------------------------------------- driver
def calibrate(ccy, model, device, *, max_iter, n_paths, T_max=None, dt=1/52, max_mat=None, sim="euler"):
    tau = TENOR[ccy]
    label = f"{model}_qe" if sim == "qe" else model      # keep QE outputs separate from Euler
    caplets = load_market(ccy, device, max_mat=max_mat)
    if T_max is None:
        T_max = float(caplets.t_fix.max()) + tau + 0.05
    n = int(round(T_max / dt)) + 1
    timeline = torch.linspace(0, T_max, n, device=device)
    f_key = torch.tensor(load_curve_vec(ccy, "key", timeline), dtype=torch.float32, device=device)
    f_ois = torch.tensor(load_curve_vec(ccy, "ois", timeline), dtype=torch.float32, device=device)

    space = make_dm_space(device)               # DM-scaled theta/v0 bounds (see make_dm_space)
    eval_loss, predict, mkt = make_eval_loss(space, timeline, f_key, f_ois, caplets, device, tau,
                                             model=model, sim=sim)

    # market-informed smart init if available, else midpoint
    if "smart_init" in space:
        p0 = space["smart_init"](mkt / 1e4, torch.tensor(caplets.t_fix.values, device=device))
    else:
        p0 = 0.5 * torch.ones(space["n_params"], device=device)

    print(f"\n=== {ccy.upper()} / {label} (sim={sim}) : {len(caplets)} caplets, {space['n_params']} params ===")
    os.makedirs(os.path.join(RUNS, "calibration_weights"), exist_ok=True)
    ckpt = os.path.join(RUNS, "calibration_weights", f"{ccy}_{label}.ckpt.pt")
    best_p, best_loss, hist = adam_spsa(
        eval_loss, p0, n_params=space["n_params"], max_iter=max_iter,
        n_paths=n_paths, device=device, ckpt_path=ckpt)
    # final eval at HIGH paths -> clean saved surface (few paths leave MC noise at the
    # short-TTM / far-OTM corners, which 'cuts down' cells in the plotted surface).
    # 100k is enough for a clean surface and keeps the fine-dt final eval within GPU memory.
    model_vol, F = predict(best_p, max(n_paths * 2, 100000), 7)
    rmse = float(torch.sqrt(((model_vol - mkt) ** 2).mean()))

    os.makedirs(os.path.join(RUNS, "calibration_weights"), exist_ok=True)
    out = os.path.join(RUNS, "calibration_weights", f"{ccy}_{label}.pt")
    torch.save({"p": best_p.cpu(), "loss": best_loss, "rmse_bp": rmse,
                "model": model, "sim": sim, "ccy": ccy, "history": hist}, out)
    # per-caplet results CSV for figures + metrics (dm_figures.py)
    res = caplets[["t_fix", "strike"]].copy()
    res["market_vol_bp"] = mkt.detach().cpu().numpy()
    res["model_vol_bp"] = model_vol.detach().cpu().numpy()
    res["F"] = F.detach().cpu().numpy()
    res_path = os.path.join(RUNS, f"results_{ccy}_{label}.csv")
    res.to_csv(res_path, index=False)
    pd.DataFrame(hist, columns=["iter", "loss"]).to_csv(
        os.path.join(RUNS, f"history_{ccy}_{label}.csv"), index=False)   # loss curve (no torch to read)
    print(f"  -> {label}: vol-RMSE = {rmse:.1f} bp  (saved {os.path.basename(out)}, "
          f"{os.path.basename(res_path)})")
    return {"model": label, "rmse_bp": round(rmse, 1), "loss": round(best_loss, 4)}


def make_dm_space(device, vol_lo_bp=20.0, vol_hi_bp=400.0, eps0=0.15):
    """v35 param structure but with DM-appropriate theta/v0 bounds. The RU floor
    THETA_MIN=1e-3 => min normal vol ~316bp, far above DM vols (~20-300bp)."""
    base = make_param_space(device, version="v35")
    lb = base["lb"].clone(); ub = base["ub"].clone()
    th_lo = (vol_lo_bp / 1e4) ** 2; th_hi = (vol_hi_bp / 1e4) ** 2
    for i in list(range(16)) + [26]:                     # 8 theta1 + 8 theta2 + v0
        lb[i] = th_lo; ub[i] = th_hi
    lb[19] = 0.08; ub[19] = 0.50                         # eps in [0.08,0.5]: convexity, avoid high-eps CUDA stall
    lb[23] = lb[24] = lb[25] = 0.0                       # rho >= 0: caps show a right (payer) skew
    rng = ub - lb
    nt = 8

    def unpack(p):
        raw = torch.clamp(lb + p * rng, lb, ub)
        return (raw[0:nt], raw[nt:2 * nt], raw[16].item(), raw[17].item(), raw[18].item(),
                raw[19].item(), raw[20].item(), raw[21].item(), raw[22].item(),
                raw[23].item(), raw[24].item(), raw[25].item(), raw[26].item())

    def smart_init(market_vols, T_fixes):
        Tn = T_fixes.detach().cpu().numpy(); Vn = market_vols.detach().cpu().numpy()
        nodes = base["theta_nodes"].detach().cpu().numpy()
        raw = 0.5 * (lb + ub).clone()
        th = torch.tensor([float(np.clip(Vn[int(np.argmin(np.abs(Tn - tn)))] ** 2, th_lo, th_hi))
                           for tn in nodes], dtype=torch.float32, device=device)
        raw[0:nt] = th; raw[nt:2 * nt] = th
        raw[16], raw[17], raw[18] = 1.5, 0.2, 0.5        # kf, ks, wf
        raw[19] = eps0                                    # eps (vol-of-vol)
        raw[20], raw[21], raw[22] = 0.4, 0.5, 0.001       # lam, gam, xi
        raw[23], raw[24], raw[25] = 0.1, 0.2, 0.3         # rho >= 0 start -> right (payer) skew
        raw[26] = th_lo * 4                               # v0
        return torch.clamp((raw - lb) / rng, 0, 1)

    return dict(unpack=unpack, smart_init=smart_init, n_params=base["n_params"],
                theta_nodes=base["theta_nodes"], rho_nodes_t=base["rho_nodes_t"],
                lb=lb, ub=ub, param_range=rng)


LADDER = ["1F_Gauss", "2F_Gauss", "HW_detvol", "HWH"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ccy", default="eur")
    ap.add_argument("--model", default="ladder", help="one of LADDER, or 'ladder' for all")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--paths", type=int, default=16384)
    ap.add_argument("--max_mat", type=float, default=None, help="cap caplet maturity (yrs) — faster")
    ap.add_argument("--dt", type=float, default=1/52, help="MC timeline step (yrs); larger = faster")
    ap.add_argument("--sim", default="euler", choices=["euler", "qe"],
                    help="variance sampler: euler (reflection) or qe (Andersen, unbiased)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    models = LADDER if args.model == "ladder" else [args.model]
    results = [calibrate(args.ccy, m, device, max_iter=args.iters, n_paths=args.paths,
                         dt=args.dt, max_mat=args.max_mat, sim=args.sim)
               for m in models]

    print("\n" + "=" * 44 + f"\nNESTED-MODEL LADDER — {args.ccy.upper()} (Roland rec 2)")
    print(f"{'model':<12}{'vol-RMSE (bp)':>14}")
    for r in results:
        print(f"{r['model']:<12}{r['rmse_bp']:>14}")
    with open(os.path.join(RUNS, f"ladder_{args.ccy}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
