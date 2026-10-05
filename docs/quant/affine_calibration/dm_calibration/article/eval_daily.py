# -*- coding: utf-8 -*-
"""Multi-seed final evaluation of a batched calibration -> the paper's headline numbers.

Re-prices a saved checkpoint at its own dt / payoff / surface with many paths over
several seeds, and reports mean +- std of the overall RMSE, the ATM mean absolute
error and the liquid-core RMSE. This is the single source for Tables 1 and 4 of
dm_writeup.tex: the numbers there should be copied from results/eval_<ccy>_<tag>.md.

  ATM          = at each t_fix, the strike nearest the MARKET forward (surface `fwd`)
  liquid core  = |K - F_mkt| < 2%
  *_ge05       = the same restricted to t_fix >= 0.5y. For the RFR markets the first
                 caplet (t_fix=0, effective expiry ~1M) has near-intrinsic wings whose
                 implied vols are numerically ill-defined; the paper's RFR headline uses
                 t_fix >= 0.5y and reports the first caplet separately.

Writes (committable, aggregate only -- no per-caplet market vols):
  article/results/eval_<ccy>_<tag>.json / .md
and the seed-averaged per-caplet surface (gitignored) for the figures:
  runs/results_<ccy>_<model>_<tag>_eval.csv   (eval_* outputs get a _<model> suffix
                                               for the nested-ladder rungs, model != HWH)

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
    ap.add_argument("--tag", required=True, help="checkpoint runs/calibration_weights/<ccy>_<model>_<tag>.pt")
    ap.add_argument("--model", default="HWH", help="HWH or a ladder rung: 1F_Gauss, 2F_Gauss, HW_detvol")
    ap.add_argument("--paths", type=int, default=200000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(os.path.join(RUNS, "calibration_weights", f"{a.ccy}_{a.model}_{a.tag}.pt"),
                    map_location=dev)
    payoff = ck.get("payoff", "arrears_avg")
    surface = ck.get("surface", "vol_surface.csv")
    fit_curve = bool(ck.get("fit_curve", False))
    dt, max_mat = float(ck["dt"]), ck.get("max_mat", 10.0)
    print(f"{a.ccy}/{a.tag}: dt={dt:.6f} payoff={payoff} surface={surface} "
          f"fit_curve={fit_curve} paths={a.paths} seeds={a.seeds}", flush=True)

    obj = B.build_objective(a.ccy, dev, model=a.model, dt=dt, max_mat=max_mat, payoff=payoff,
                            surface=surface,
                            fit_curve=fit_curve)
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
    ge = tf >= 0.5 - 1e-9
    atm_ge = atm[tf[atm] >= 0.5 - 1e-9]
    buckets = {"0-0.25": tf < 0.5 - 1e-9, "0.5-1": (tf >= 0.5 - 1e-9) & (tf <= 1 + 1e-9),
               "1-3": (tf > 1 + 1e-9) & (tf <= 3 + 1e-9), "3-7": (tf > 3 + 1e-9) & (tf <= 7 + 1e-9),
               "7-10": tf > 7 + 1e-9}
    rmse = lambda x: float(np.sqrt((x ** 2).mean())) if x.size else float("nan")

    rows, vols, Fs = [], [], []
    for s in a.seeds:
        with torch.no_grad():
            v, F = obj["predict"](p, a.paths, s)
        v = v[0].cpu().numpy(); vols.append(v); Fs.append(F[0].cpu().numpy())
        e = v - mkt
        row = dict(seed=s, rmse_bp=rmse(e), atm_mae_bp=float(np.abs(e[atm]).mean()),
                   core_rmse_bp=rmse(e[core]), rmse_ge05_bp=rmse(e[ge]),
                   atm_mae_ge05_bp=float(np.abs(e[atm_ge]).mean()),
                   core_rmse_ge05_bp=rmse(e[core & ge]))
        row.update({f"rmse_{k}_bp": rmse(e[m]) for k, m in buckets.items() if m.any()})
        rows.append(row)
        print(f"  seed {s}: RMSE {row['rmse_bp']:.2f}  ATM MAE {row['atm_mae_bp']:.2f}  "
              f"core {row['core_rmse_bp']:.2f}  | t>=0.5: RMSE {row['rmse_ge05_bp']:.2f}  "
              f"ATM {row['atm_mae_ge05_bp']:.2f}  core {row['core_rmse_ge05_bp']:.2f}", flush=True)
        torch.cuda.empty_cache() if dev.type == "cuda" else None

    df = pd.DataFrame(rows)
    summ = {k: {"mean": round(float(df[k].mean()), 2), "std": round(float(df[k].std(ddof=1)), 2)}
            for k in df.columns if k != "seed"}
    meta = dict(ccy=a.ccy, tag=a.tag, model=a.model, dt=dt, payoff=payoff, surface=surface, max_mat=max_mat,
                fit_curve=fit_curve,
                n_caplets=int(len(c)), paths=a.paths, seeds=a.seeds,
                train_rmse_bp=ck.get("rmse_bp"), iters=ck.get("iters"), init=ck.get("init"))
    os.makedirs(OUT, exist_ok=True)
    stem = f"eval_{a.ccy}_{a.tag}" + ("" if a.model == "HWH" else f"_{a.model}")
    with open(os.path.join(OUT, f"{stem}.json"), "w") as f:
        json.dump(dict(meta=meta, summary=summ, per_seed=rows), f, indent=2)
    md = [f"# {a.ccy.upper()} / {a.tag} -- {len(a.seeds)}-seed evaluation",
          "", f"dt={dt:.6f}, payoff `{payoff}`, surface `{surface}`, {len(c)} caplets, "
              f"{a.paths} paths, seeds {a.seeds}", "",
          "| metric | mean | std |", "|---|---|---|"]
    md += [f"| {k} | {v['mean']:.2f} | {v['std']:.2f} |" for k, v in summ.items()]
    with open(os.path.join(OUT, f"{stem}.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    res = c[["t_fix", "strike"]].copy()
    res["market_vol_bp"] = mkt
    res["model_vol_bp"] = np.mean(vols, axis=0)
    res["F"] = np.mean(Fs, axis=0)
    res["F_mkt"] = F_mkt
    res.to_csv(os.path.join(RUNS, f"results_{a.ccy}_{a.model}_{a.tag}_eval.csv"), index=False)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
