# -*- coding: utf-8 -*-
"""Batched Adam-SPSA driver on the streaming simulator.

Same model, same objective, same optimiser maths as dm_calibrate.py -- the only
difference is that an SPSA iteration's 2M+1 evaluations are stepped through ONE
loop instead of 2M+1 sequential ones. Because the simulation is kernel-launch
bound (17 members measured at 6.73s vs 7.59s for a single one at daily dt), that
turns ~129 s/iter into ~7 s/iter and makes a daily-dt calibration a ~1.5-3 h run
instead of a multi-day one.

dm_calibrate.py is deliberately untouched: import it, don't modify it. Run this
module only when you want the batched path.

  python dm_calibrate_batched.py --ccy eur --dt 0.02  --iters 800 --max_mat 10
  python dm_calibrate_batched.py --ccy eur --dt 0.004 --iters 800 --max_mat 10

Instrument conventions (Oct 2026, after Henrard 2026). --payoff auto picks
  eur -> term          6M Euribor is set at t_fix (forward-looking term rate)
  usd/gbp -> arrears_comp   SOFR/SONIA compound daily through the accrual window
and --surface picks the stripped target (vol_surface_lm.csv / _end.csv carry the
in-arrears expiry texp; see caplet_stripper.py). --payoff arrears_avg with the
default surface reproduces the pre-Oct runs. --init warm-starts from a checkpoint.
--fit_curve adds the deterministic shift that makes the model reprice the initial
discount and projection curves (see dm_sim_stream._price).

Two documented differences from the sequential optimiser, neither affecting the
estimator's form:
  * the members of one sub-batch share a seed (they draw independent slices of
    it) rather than each taking its own seed. The +/- perturbations still see
    different draws, so this is not common-random-numbers.
  * random draws are generated in step-chunks, so the stream cannot be
    reproduced bit-for-bit against the sequential code. Equivalence is pinned
    instead by driving both with explicit draws (see the gate in scratch).
"""
import os, sys, json, argparse, math
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "scripts")))

import dm_calibrate as DC
from dm_calibrate import (TENOR, RUNS, load_market, load_curve_vec, make_dm_space,
                          constrain, implied_vol_bisect, LADDER)
from dm_sim_stream import simulate_price_streaming_batched, PAYOFFS
from simulation import theta_to_vec, rho_to_vec


# ----------------------------------------------------------------- objective
AUTO_PAYOFF = {"eur": "term", "usd": "arrears_comp", "gbp": "arrears_comp"}


def build_objective(ccy, device, *, model="HWH", dt=0.02, max_mat=None, short_wt=1.5,
                    payoff="arrears_avg", surface="vol_surface.csv", fit_curve=False):
    """Market tensors + a batched loss. The vega weighting mirrors
    dm_calibrate.make_eval_loss exactly; the gate checks the two agree."""
    tau = TENOR[ccy]
    payoff = AUTO_PAYOFF[ccy] if payoff == "auto" else payoff
    assert payoff in PAYOFFS, payoff
    caplets = load_market(ccy, device, max_mat=max_mat, surface=surface)
    # Grid snapped so the step is EXACTLY dt. The accrual average runs over a
    # whole number of steps (i_pay - i_fix), so tau/dt must be an integer or the
    # simulated accrual period is the wrong length -- at dt=0.02 a 3M tenor gives
    # 12.5, i.e. windows alternating 0.238/0.258y against a true 0.25. At
    # dt=1/252 it is 63 steps for 3M and 126 for 6M, exactly, and every business
    # day is accrued as a backward-looking RFR caplet requires.
    span = float(caplets.t_fix.max()) + tau + 0.05
    n_steps = int(math.ceil(span / dt - 1e-9))
    T_max = n_steps * dt
    n = n_steps + 1
    timeline = (torch.arange(n, dtype=torch.float64) * dt).to(torch.float32).to(device)
    step = dt
    f_key = torch.tensor(load_curve_vec(ccy, "key", timeline), dtype=torch.float32, device=device)
    f_ois = torch.tensor(load_curve_vec(ccy, "ois", timeline), dtype=torch.float32, device=device)

    t_fix = torch.tensor(caplets.t_fix.values, dtype=torch.float32, device=device)
    texp = torch.tensor(caplets.texp.values, dtype=torch.float32, device=device)
    i_fix = torch.tensor([int(round(t / step)) for t in caplets.t_fix.values], device=device)
    i_pay = torch.tensor([int(round((t + tau) / step)) for t in caplets.t_fix.values], device=device)
    K_t = torch.tensor(caplets.strike.values, dtype=torch.float32, device=device)
    tau_t = torch.tensor(tau, dtype=torch.float32, device=device)
    mkt = torch.tensor(caplets.sigma_mkt_bp.values, dtype=torch.float32, device=device)

    _tf = caplets.t_fix.values.astype(float)
    _K = caplets.strike.values.astype(float)
    _sig = caplets.sigma_mkt_bp.values.astype(float) / 1e4
    _Fpx = np.array([_K[_tf == t][int(np.argmin(_sig[_tf == t]))] for t in _tf])
    _d = (_Fpx - _K) / (_sig * np.sqrt(np.maximum(caplets.texp.values.astype(float), 0.1)))
    _w = np.maximum(np.exp(-0.5 * _d ** 2), 0.20)
    wt = torch.tensor(_w, dtype=torch.float32, device=device) * torch.where(t_fix < 1.0, short_wt, 1.0)
    wt = wt / wt.mean()

    space = make_dm_space(device)
    nodes, rho_nodes = space["theta_nodes"], space["rho_nodes_t"]
    unpack = space["unpack"]

    def _members(P):
        """(K, n_params) in [0,1] -> per-member parameter tensors for the batched sim."""
        rows = [constrain(model, unpack(P[k])) for k in range(P.shape[0])]
        vec = lambda j: torch.tensor([float(r[j]) for r in rows], device=device)
        return dict(
            theta_mat=torch.stack([theta_to_vec(r[0], nodes, timeline) for r in rows]),
            theta2_mat=torch.stack([theta_to_vec(r[1], nodes, timeline) for r in rows]),
            rho_mat=torch.stack([rho_to_vec(torch.tensor([r[9], r[10], r[11]], device=device),
                                            rho_nodes, timeline) for r in rows]),
            kf=vec(2), ks=vec(3), wf=vec(4), eps=vec(5), lam=vec(6), gam=vec(7),
            xi=vec(8), v0=vec(12))

    def predict(P, n_paths, seed):
        m = _members(P)
        pv, F, D = simulate_price_streaming_batched(
            n_paths, timeline, m["theta_mat"], m["theta2_mat"], m["eps"], m["v0"],
            m["kf"], m["ks"], m["wf"], m["lam"], m["gam"], m["xi"], f_key, f_ois,
            m["rho_mat"], i_fix, i_pay, tau_t, K_t, device, seed=seed, payoff=payoff,
            fit_curve=fit_curve)
        return implied_vol_bisect(pv, F, K_t, texp, tau_t, D) * 1e4, F

    def eval_batch(P, n_paths, seed):
        """-> (K,) vega-weighted vol-RMSE, matching dm_calibrate's eval_loss per member."""
        model_vol, _ = predict(P, n_paths, seed)
        model_vol = torch.nan_to_num(model_vol, nan=1e4, posinf=1e4, neginf=0.0)
        err = model_vol - mkt
        rmse = torch.sqrt((wt * err * err).mean(dim=-1))
        return torch.where(torch.isfinite(rmse), rmse, torch.full_like(rmse, 999.0))

    w = (i_pay - i_fix).unique().tolist()
    acc_err = max(abs(x * dt / tau - 1.0) for x in w) * 100
    if len(w) > 1 or acc_err > 0.5:
        print(f"  WARNING accrual window {w} steps x dt={dt:.6f} vs tau={tau}: "
              f"up to {acc_err:.2f}% error -- tau/dt = {tau/dt:.3f} is not an integer",
              flush=True)
    else:
        print(f"  accrual window {w[0]} steps x dt={dt:.6f} = {w[0]*dt:.6f} "
              f"vs tau={tau} ({acc_err:.3f}% error)", flush=True)

    print(f"  payoff={payoff}  surface={surface}  fit_curve={fit_curve}", flush=True)
    return dict(eval_batch=eval_batch, predict=predict, mkt=mkt, caplets=caplets,
                space=space, n=n, timeline=timeline, T_max=T_max, accrual_steps=w,
                payoff=payoff, surface=surface, fit_curve=fit_curve)


# ----------------------------------------------------------------- optimiser
def adam_spsa_batched(eval_batch, p_init, *, n_params, max_iter=800, n_paths=16384,
                      batch=9, lr=3e-3, c=0.02, M=8, seed_base=42, device="cpu",
                      ckpt_path=None):
    """dm_calibrate.adam_spsa with the 2M+1 evaluations batched. The cosine
    schedules on lr and c are identical, so a run at one dt is directly
    comparable with a run at another."""
    p = p_init.clone()
    m = torch.zeros(n_params, device=device); v = torch.zeros(n_params, device=device)
    b1, b2, eps_a = 0.9, 0.999, 1e-8
    best_loss, best_p = float("inf"), p.clone()
    hist = []

    def run_all(P, seed0):
        out = []
        for s in range(0, P.shape[0], batch):
            out.append(eval_batch(P[s:s + batch], n_paths, seed0 + s))
        return torch.cat(out)

    for it in range(max_iter):
        frac = it / max(max_iter - 1, 1)
        lr_t = lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))   # cosine
        c_t = c * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * frac)))
        deltas = [(torch.randint(0, 2, (n_params,), device=device).float() * 2 - 1)
                  for _ in range(M)]
        P = torch.stack([torch.clamp(p + c_t * d, 0, 1) for d in deltas]
                        + [torch.clamp(p - c_t * d, 0, 1) for d in deltas]
                        + [p])
        losses = run_all(P, seed_base + it * 977)
        g = torch.zeros(n_params, device=device)
        for j in range(M):
            g += (losses[j] - losses[M + j]) / (2 * c_t) * deltas[j]
        g = torch.clamp(g / M, -10, 10)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** (it + 1)); vh = v / (1 - b2 ** (it + 1))
        p = torch.clamp(p - lr_t * mh / (vh.sqrt() + eps_a), 0, 1)
        cur = float(losses[2 * M])
        if cur < best_loss:
            best_loss, best_p = cur, p.clone()
        if it % 10 == 0:
            hist.append((it, cur))
        if it % 50 == 0:
            print(f"    iter {it:4d}  loss={cur:.4f}  best={best_loss:.4f}", flush=True)
            if ckpt_path is not None:
                torch.save({"p": best_p.cpu(), "loss": best_loss, "iter": it}, ckpt_path)
    return best_p, best_loss, hist


# ----------------------------------------------------------------- driver
def calibrate(ccy, model, device, *, max_iter, n_paths, dt, max_mat, batch, tag,
              payoff="arrears_avg", surface="vol_surface.csv", init=None, fit_curve=False):
    obj = build_objective(ccy, device, model=model, dt=dt, max_mat=max_mat,
                          payoff=payoff, surface=surface, fit_curve=fit_curve)
    space, mkt = obj["space"], obj["mkt"]
    label = f"{model}_{tag}"
    if init:
        p0 = torch.load(init, map_location=device)["p"].to(device)
        print(f"  warm start from {os.path.basename(init)}", flush=True)
    else:
        p0 = space["smart_init"](mkt / 1e4,
                                 torch.tensor(obj["caplets"].t_fix.values, device=device))
    print(f"\n=== {ccy.upper()} / {label} : {len(obj['caplets'])} caplets, dt={dt}, "
          f"{obj['n']-1} steps, batch={batch}, {max_iter} iters ===", flush=True)
    os.makedirs(os.path.join(RUNS, "calibration_weights"), exist_ok=True)
    ckpt = os.path.join(RUNS, "calibration_weights", f"{ccy}_{label}.ckpt.pt")
    best_p, best_loss, hist = adam_spsa_batched(
        obj["eval_batch"], p0, n_params=space["n_params"], max_iter=max_iter,
        n_paths=n_paths, batch=batch, device=device, ckpt_path=ckpt)

    final_paths = min(100000, int(6.0e7 / obj["n"]))
    model_vol, F = obj["predict"](best_p.unsqueeze(0), final_paths, 7)
    model_vol, F = model_vol[0], F[0]
    rmse = float(torch.sqrt(((model_vol - mkt) ** 2).mean()))

    out = os.path.join(RUNS, "calibration_weights", f"{ccy}_{label}.pt")
    torch.save({"p": best_p.cpu(), "loss": best_loss, "rmse_bp": rmse, "model": model,
                "sim": "stream_batched", "ccy": ccy, "dt": dt, "max_mat": max_mat,
                "n_paths": n_paths, "iters": max_iter, "history": hist,
                "payoff": obj["payoff"], "surface": surface, "init": init, "fit_curve": fit_curve,
                "final_paths": final_paths, "final_seed": 7}, out)
    res = obj["caplets"][["t_fix", "strike"]].copy()
    res["market_vol_bp"] = mkt.detach().cpu().numpy()
    res["model_vol_bp"] = model_vol.detach().cpu().numpy()
    res["F"] = F.detach().cpu().numpy()
    res.to_csv(os.path.join(RUNS, f"results_{ccy}_{label}.csv"), index=False)
    pd.DataFrame(hist, columns=["iter", "loss"]).to_csv(
        os.path.join(RUNS, f"history_{ccy}_{label}.csv"), index=False)
    print(f"  -> {label}: vol-RMSE = {rmse:.2f} bp  (final eval {final_paths} paths)", flush=True)
    return {"model": label, "dt": dt, "rmse_bp": round(rmse, 2), "loss": round(best_loss, 4),
            "payoff": obj["payoff"], "surface": surface, "fit_curve": fit_curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ccy", default="eur")
    ap.add_argument("--model", default="HWH")
    ap.add_argument("--iters", type=int, default=800, help="matches the 9.4bp run's schedule")
    ap.add_argument("--paths", type=int, default=16384)
    ap.add_argument("--max_mat", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--batch", type=int, default=9, help="SPSA members per batched pass")
    ap.add_argument("--tag", default=None, help="output label (default: dt-derived)")
    ap.add_argument("--payoff", default="arrears_avg", choices=("auto",) + PAYOFFS,
                    help="caplet underlying; auto = eur:term, usd/gbp:arrears_comp")
    ap.add_argument("--surface", default="vol_surface.csv",
                    help="stripped target in data/<ccy>/ (e.g. vol_surface_lm.csv)")
    ap.add_argument("--init", default=None, help="checkpoint .pt to warm-start from")
    ap.add_argument("--fit_curve", action="store_true",
                    help="deterministic shift so the model reprices the initial curves")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag or ("daily" if args.dt < 0.01 else "weekly")
    print(f"device={device}  torch={torch.__version__}", flush=True)
    r = calibrate(args.ccy, args.model, device, max_iter=args.iters, n_paths=args.paths,
                  dt=args.dt, max_mat=args.max_mat, batch=args.batch, tag=tag,
                  payoff=args.payoff, surface=args.surface, init=args.init,
                  fit_curve=args.fit_curve)
    with open(os.path.join(RUNS, f"batched_{args.ccy}_{tag}.json"), "w") as f:
        json.dump(r, f, indent=2)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
