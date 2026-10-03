# -*- coding: utf-8 -*-
"""Multi-seed final evaluation of a batched calibration -> the paper's headline numbers.

Re-prices a saved checkpoint at its own dt / payoff / surface with many paths over
several seeds, and reports mean +- std of the overall RMSE, the ATM mean absolute
error and the liquid-core RMSE. This is the single source for Tables 1 and 4 of
dm_writeup.tex: the numbers there should be copied from results/eval_<ccy>_<tag>.md.

  ATM          = at each t_fix, the strike nearest the MARKET forward (surface `fwd`)
  liquid core  = |K - F_mkt| < 2%

Writes (committable, aggregate only -- no per-caplet market vols):
  article/results/eval_<ccy>_<tag>.json / .md
and the seed-averaged per-caplet surface (gitignored) for the figures:
  runs/results_<ccy>_HWH_<tag>_eval.csv

Run:  <ml-python> eval_daily.py --ccy eur --tag term [--paths 200000 --seeds 7 8 9]

Pre-Oct-2026 checkpoints carry no payoff/surface and are evaluated with the legacy
arrears_avg payoff on vol_surface.csv. Note USD now projects on SOFR, so a legacy
USD checkpoint (fitted on the Fed Funds curve) no longer evaluates as it was fitted.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
DM = os.path.dirname(HERE)
sys.path.insert(0, DM)
sys.path.insert(0, os.path.join(DM, "..", "scripts"))
import dm_calibrate_batched as B                                      # noqa: E402
from dm_calibrate import RUNS, DATA                                   # noqa: E402

OUT = os.path.join(HERE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ccy", required=True)
    ap.add_argument("--tag", required=True, help="checkpoint runs/calibration_weights/<ccy>_HWH_<tag>.pt")
    ap.add_argument("--paths", type=int, default=200000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(os.path.join(RUNS, "calibration_weights", f"{a.ccy}_HWH_{a.tag}.pt"),
                    map_location=dev)
    payoff = ck.get("payoff", "arrears_avg")
    surface = ck.get("surface", "vol_surface.csv")
    dt, max_mat = float(ck["dt"]), ck.get("max_mat", 10.0)
    print(f"{a.ccy}/{a.tag}: dt={dt:.6f} payoff={payoff} surface={surface} "
          f"paths={a.paths} seeds={a.seeds}", flush=True)

    obj = B.build_objective(a.ccy, dev, dt=dt, max_mat=max_mat, payoff=payoff, surface=surface)
    c = obj["caplets"]
    mkt = obj["mkt"].cpu().numpy()
    p = ck["p"].to(dev).unsqueeze(0)

    surf = pd.read_csv(os.path.join(DATA, a.ccy, surface))
    fmkt = surf[surf.strike == "ATM"].set_index("t_fix_yrs").fwd
    tf = c.t_fix.values.astype(float)
    K = c.strike.values.astype(float)
    F_mkt = np.array([fmkt[t] for t in tf])
    core = np.abs(K - F_mkt) < 0.02
    atm = np.array([np.where(tf == t)[0][np.argmin(np.abs(K[tf == t] - fmkt[t]))]
                    for t in np.unique(tf)])

    rows, vols, Fs = [], [], []
    for s in a.seeds:
        with torch.no_grad():
            v, F = obj["predict"](p, a.paths, s)
        v = v[0].cpu().numpy(); vols.append(v); Fs.append(F[0].cpu().numpy())
        e = v - mkt
        rows.append(dict(seed=s, rmse_bp=float(np.sqrt((e ** 2).mean())),
                         atm_mae_bp=float(np.abs(e[atm]).mean()),
                         core_rmse_bp=float(np.sqrt((e[core] ** 2).mean()))))
        print(f"  seed {s}: RMSE {rows[-1]['rmse_bp']:.2f}  ATM MAE {rows[-1]['atm_mae_bp']:.2f}  "
              f"core {rows[-1]['core_rmse_bp']:.2f}", flush=True)
        torch.cuda.empty_cache() if dev.type == "cuda" else None

    df = pd.DataFrame(rows)
    summ = {k: {"mean": round(float(df[k].mean()), 2), "std": round(float(df[k].std(ddof=1)), 2)}
            for k in ("rmse_bp", "atm_mae_bp", "core_rmse_bp")}
    meta = dict(ccy=a.ccy, tag=a.tag, dt=dt, payoff=payoff, surface=surface, max_mat=max_mat,
                n_caplets=int(len(c)), paths=a.paths, seeds=a.seeds,
                train_rmse_bp=ck.get("rmse_bp"), iters=ck.get("iters"), init=ck.get("init"))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"eval_{a.ccy}_{a.tag}.json"), "w") as f:
        json.dump(dict(meta=meta, summary=summ, per_seed=rows), f, indent=2)
    md = [f"# {a.ccy.upper()} / {a.tag} -- {len(a.seeds)}-seed evaluation",
          "", f"dt={dt:.6f}, payoff `{payoff}`, surface `{surface}`, {len(c)} caplets, "
              f"{a.paths} paths, seeds {a.seeds}", "",
          "| metric | mean | std |", "|---|---|---|"]
    md += [f"| {k} | {v['mean']:.2f} | {v['std']:.2f} |" for k, v in summ.items()]
    with open(os.path.join(OUT, f"eval_{a.ccy}_{a.tag}.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    res = c[["t_fix", "strike"]].copy()
    res["market_vol_bp"] = mkt
    res["model_vol_bp"] = np.mean(vols, axis=0)
    res["F"] = np.mean(Fs, axis=0)
    res["F_mkt"] = F_mkt
    res.to_csv(os.path.join(RUNS, f"results_{a.ccy}_HWH_{a.tag}_eval.csv"), index=False)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
