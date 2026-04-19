"""Calibration objective helpers: checkpoint I/O, SPSA/Adam optimisers, batched evaluation, experiment log."""

import pickle
import time
import sqlite3
import json
import datetime
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from IPython.display import display, clear_output

from affine_calibration.scripts.simulation import (
    fast_simulate_2factor, batch_price_caplets,
    theta_to_vec, rho_to_vec,
)
from affine_calibration.scripts.caplet_vol_surface import (
    generate_caplet_vol_surface, plot_caplet_vol_surface,
    params_to_dataframe, implied_vol_avg_rate,
)
from affine_calibration.scripts.pricing_models import implied_vol_batch
from affine_calibration.scripts.loss_function import (
    total_loss as loss_total,
    make_strike_weights as make_strike_weights_lf,
)
from affine_calibration.scripts.optimizers import (
    run_spsa, run_adam_spsa,
)

_val = lambda x: x.item() if isinstance(x, torch.Tensor) else float(x)


# ── Error-adaptive strike weights ─────────────────────────────────────────────

def make_error_adaptive_weights(T_fixes, device, csv_search_paths,
                                err_floor=5e-3, max_weight=4.0, verbose=True):
    """Build per-caplet weights proportional to sqrt(|current vol error|).

    Cells where the model is already accurate get weight close to 1; cells
    with large errors get upweighted (up to ``max_weight``), so the optimizer
    focuses gradient budget on the misfitting wings rather than the well-fitted
    ATM region.

    Parameters
    ----------
    T_fixes : Tensor — maturity for each caplet (N,)
    device  : torch.device
    csv_search_paths : list of (Path, label) — checked in order; first existing
        file with a 'vol_error' column is used.
    err_floor : float — minimum absolute error (bp) before sqrt scaling.
        Prevents well-fitted cells from getting zero weight (default 0.5%).
    max_weight : float — cap on any single cell's weight (default 4×).
    verbose : bool — print weight diagnostics per maturity bucket.

    Returns
    -------
    strike_weights : Tensor (N,) normalised per maturity bucket, or None if no
        CSV found (caller should fall back to uniform weights).
    """
    for csv_path, label in csv_search_paths:
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            err_col = next((c for c in df.columns if 'vol_error' in c), None)
            if err_col:
                if verbose:
                    print(f"Error-adaptive weights from: {label}  (col={err_col})")
                err_abs = torch.tensor(
                    np.abs(df[err_col].fillna(0).values),
                    dtype=torch.float32, device=device,
                ).clamp(min=err_floor)
                err_med = err_abs.median().clamp(min=1e-4)
                w = (err_abs / err_med).sqrt().clamp(min=0.2, max=max_weight)
                # Normalize per maturity bucket: sum of weights = n_caplets in bucket
                for T_u in torch.unique(T_fixes):
                    bk = T_fixes == T_u
                    s  = w[bk].sum()
                    if s > 0:
                        w[bk] = w[bk] * bk.sum().float() / s
                if verbose:
                    print(f"  Weight range: [{w.min():.2f}, {w.max():.2f}]  mean={w.mean():.2f}")
                    for T_u in torch.unique(T_fixes):
                        bk = T_fixes == T_u
                        print(f"  T={T_u:.2f}y  max_w={w[bk].max():.2f}  min_w={w[bk].min():.2f}")
                return w
    if verbose:
        print("No prior CSV found — using uniform strike weights")
    return None


# ── Loss factory ──────────────────────────────────────────────────────────────

def make_strike_weights(market_pvs_cached, calib_mask, T_fixes, device,
                        itm_weight=0.15, atm_otm_weight=1.0):
    """Build per-caplet PV-informed weights for the calibration loss.

    Deep-ITM caplets (excluded from calib_mask) are given ``itm_weight``
    so they don't dominate even when the mask is loosened.  ATM/OTM caplets
    get ``atm_otm_weight``.  Weights are normalised per maturity bucket so
    the bucket-RMSE scale is unchanged.

    Parameters
    ----------
    market_pvs_cached : Tensor  — (N,) market PVs
    calib_mask        : BoolTensor — True  = ATM/OTM (inside calibration)
    T_fixes           : Tensor  — maturity for each caplet
    device            : torch.device
    itm_weight        : float   — weight for deep-ITM caplets (default 0.15)
    atm_otm_weight    : float   — weight for ATM/OTM caplets (default 1.0)

    Returns
    -------
    strike_weights : Tensor (N,) — per-caplet weights, mean-normalised per bucket
    """
    import torch
    w = torch.where(calib_mask,
                    torch.tensor(atm_otm_weight, device=device),
                    torch.tensor(itm_weight,     device=device))
    mats_unique = torch.unique(T_fixes)
    for T_mat in mats_unique:
        bucket = (T_fixes == T_mat)
        w_sum = w[bucket].sum()
        if w_sum > 0:
            w[bucket] = w[bucket] / w_sum * bucket.sum().float()
    return w


def make_eval_loss(unpack, theta_nodes, rho_nodes_t, timeline,
                   f_key_vec, f_ois_vec, idx_mats,
                   strikes, market_pvs_cached, market_vegas_floored,
                   calib_mask, T_fixes, device, smooth_pen_weight=0.1,
                   mat_weights=None, strike_weights=None,
                   objective_mode='pv', market_vols=None,
                   w_pv=0.5, w_vol=0.5,
                   # Legacy aliases — ignored, kept for backward compat
                   idx_pays=None, tau=None, idx_fixes=None):
    """Factory: build ``eval_loss(p, n_paths, seed) → (loss, model_pvs)``.

    Captures all grid data in a closure so the notebook doesn't need
    ``compute_loss`` / ``eval_loss`` defined inline.

    Parameters
    ----------
    objective_mode : str
        'pv' (default) — vega-normalized PV loss
        'vol'         — direct implied-vol RMSE loss
        'hybrid'      — weighted blend of PV and vol losses
    market_vols : Tensor or None
        Market implied vols (required for 'vol' and 'hybrid' modes).
    mat_weights : Tensor or None
        Per-maturity weights (aligned with ``torch.unique(T_fixes)``).
        If None, all maturities weighted equally (backward-compatible).
    strike_weights : Tensor or None
        Per-caplet weights inside each bucket.  Build with
        ``make_strike_weights()``.  If None, uniform within bucket.
        Deep-ITM caplets should get a low weight (e.g. 0.15) so they
        don't pull calibration away from the hedging-relevant ATM region.
    """
    mats_unique = torch.unique(T_fixes)

    def eval_loss(p, n_paths, seed):
        result = unpack(p)
        if len(result) == 13:  # v18: two independent theta vectors, free v0
            theta1, theta2, kf, ks, wf, eps, lam, gam, xi, rho1, rho5, rho10, v0_free = result
            v0_pinned = v0_free if isinstance(v0_free, float) else v0_free.item()
            theta_vec  = theta_to_vec(theta1, theta_nodes, timeline)
            theta2_iter = theta_to_vec(theta2, theta_nodes, timeline)
            theta_for_penalty = torch.cat([theta1, theta2])
        elif len(result) == 12:  # v14+: free v0, shared theta
            theta1, kf, ks, wf, eps, lam, gam, xi, rho1, rho5, rho10, v0_free = result
            v0_pinned = v0_free if isinstance(v0_free, float) else v0_free.item()
            theta_vec  = theta_to_vec(theta1, theta_nodes, timeline)
            theta2_iter = theta_vec
            theta_for_penalty = theta1
        else:  # v10: v0 pinned to theta(3M)
            theta1, kf, ks, wf, eps, lam, gam, xi, rho1, rho5, rho10 = result
            v0_pinned = theta1[0].item()
            theta_vec  = theta_to_vec(theta1, theta_nodes, timeline)
            theta2_iter = theta_vec
            theta_for_penalty = theta1

        rho_nodes_v = torch.tensor([rho1, rho5, rho10],
                                   dtype=torch.float32, device=device)
        rho_vec = rho_to_vec(rho_nodes_v, rho_nodes_t, timeline)

        key_paths, ois_paths, _ = fast_simulate_2factor(
            n_paths, timeline, theta_vec, eps, v0_pinned,
            kf, ks, wf, lam, gam, xi,
            f_key_vec, f_ois_vec, device, seed=seed,
            rho_vx=rho_vec, antithetic=True,
            theta2_vec=theta2_iter,
        )
        model_pvs, F_model, P_model = batch_price_caplets(
            key_paths, ois_paths, timeline, idx_mats,
            T_fixes, strikes, device,
        )

        # Vol/hybrid mode: vectorized GPU bisection — ~50x faster than per-caplet Brentq
        model_vols_inv = None
        if objective_mode.lower() in ['vol', 'hybrid']:
            model_vols_inv = implied_vol_batch(
                F_model, strikes, T_fixes, model_pvs, P_model,
            )
            # Where bisection returns 0 (failed inversion: deep-ITM or sub-intrinsic MC
            # price due to discretisation noise), use a first-order vega-delta proxy:
            #   Δσ ≈ (PV_model − PV_market) / vega_market
            # This keeps gradient signal alive for those caplets instead of silently
            # zeroing their error (market_vol substitution would give zero loss → zero
            # gradient → optimizer never learns deep-ITM short-end caplets).
            if market_vols is not None:
                zero_mask = model_vols_inv == 0.0
                if zero_mask.any():
                    delta_pv   = model_pvs - market_pvs_cached
                    vol_proxy  = (market_vols +
                                  delta_pv / market_vegas_floored.clamp(min=1e-10)
                                  ).clamp(min=1e-4)
                    model_vols_inv = torch.where(zero_mask, vol_proxy, model_vols_inv)

        # Use unified loss function with objective_mode switch
        loss = loss_total(
            objective_mode=objective_mode,
            theta_values=theta_for_penalty,
            smooth_pen_weight=smooth_pen_weight,
            model_pvs=model_pvs,
            market_pvs=market_pvs_cached,
            market_vegas=market_vegas_floored,
            t_fixes=T_fixes,
            calib_mask=calib_mask,
            model_vols=model_vols_inv,
            market_vols=market_vols,
            w_pv=w_pv, w_vol=w_vol,
            mat_weights=mat_weights,
            strike_weights=strike_weights,
        )

        return loss, model_pvs

    return eval_loss


# ── Differentiable loss for AD-based optimisers (Adam) ────────────────────────

def make_ad_loss(theta_nodes, rho_nodes_t, timeline,
                 f_key_vec, f_ois_vec, idx_mats,
                 strikes, market_pvs_cached, market_vegas_floored,
                 calib_mask, T_fixes, device, smooth_pen_weight=0.1,
                 # Legacy aliases — ignored, kept for backward compat
                 idx_pays=None, tau=None, idx_fixes=None):
    """Build a differentiable loss closure for Adam / L-BFGS.

    Unlike ``make_eval_loss`` (SPSA), this version:
    - Takes raw tensor ``params`` (17-d, unit-cube) and ``lb``/``ub`` tensors
    - Keeps every operation on the graph (no ``.item()``, no ``np.*``)
    - Returns ``(loss, model_pvs.detach())``

    The simulation functions (fast_cir_paths, fast_ou_paths, fast_hw_paths,
    fast_simulate_2factor, batch_price_caplets) are already autograd-safe
    since v14 refactoring (list-based accumulation, tensor-safe ops).
    """
    mats_unique = torch.unique(T_fixes)

    def ad_loss(params, lb, ub, n_paths, seed):
        """Evaluate loss with full autograd graph.

        Parameters
        ----------
        params : Tensor [17], requires_grad=True, values in [0,1]
        lb, ub : Tensor [17]  (bounds, no grad needed)
        n_paths : int
        seed : int or None
        """
        raw = lb + params * (ub - lb)
        raw = torch.clamp(raw, lb, ub)

        theta   = raw[:6]
        kf      = raw[6]
        ks      = raw[7]
        wf      = raw[8]
        eps     = raw[9]
        lam_v   = raw[10]
        gam     = raw[11]
        xi      = raw[12]
        rho1    = raw[13]
        rho5    = raw[14]
        rho10   = raw[15]
        v0      = raw[16]

        theta_vec = theta_to_vec(theta, theta_nodes, timeline)
        rho_nodes_v = torch.stack([rho1, rho5, rho10])
        rho_vec = rho_to_vec(rho_nodes_v, rho_nodes_t, timeline)

        key_paths, ois_paths, _ = fast_simulate_2factor(
            n_paths, timeline, theta_vec, eps, v0,
            kf, ks, wf, lam_v, gam, xi,
            f_key_vec, f_ois_vec, device, seed=seed,
            rho_vx=rho_vec, antithetic=True,
        )
        model_pvs, _, _ = batch_price_caplets(
            key_paths, ois_paths, timeline, idx_mats,
            T_fixes, strikes, device,
        )

        # Per-bucket vega-weighted RMSE (same formula as SPSA loss)
        # eps inside sqrt: ∂√x/∂x = 1/(2√x) → ∞ as x→0, causing NaN gradients
        vol_err = (model_pvs - market_pvs_cached) / market_vegas_floored
        bucket_rmses = []
        for T_mat in mats_unique:
            mask = calib_mask & (T_fixes == T_mat)
            if mask.sum() > 0:
                bucket_rmses.append(torch.sqrt(torch.mean(vol_err[mask] ** 2) + 1e-12))
        loss = torch.stack(bucket_rmses).mean()

        smooth_pen = ((theta[1:] - theta[:-1]) ** 2).sum() * smooth_pen_weight
        return loss + smooth_pen, model_pvs.detach()

    return ad_loss


# ── Checkpoint I/O ────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path, device):
    """Load a calibration checkpoint and return (best_params, theta_nodes)."""
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    _th_main = ckpt.get('theta1_values', ckpt.get('theta_values')).to(device)
    best_params = {
        'theta':      _th_main,
        'kappa_fast': ckpt['kappa_fast'],
        'kappa_slow': ckpt['kappa_slow'],
        'w_fast':     ckpt['w_fast'],
        'epsilon':    ckpt['epsilon'],
        'lam':        ckpt['lam'],
        'gamma':      ckpt['gamma'],
        'xi':         ckpt['xi'],
        'rho_1y':     ckpt.get('rho_1y', 0.0),
        'rho_5y':     ckpt.get('rho_5y', 0.0),
        'rho_10y':    ckpt.get('rho_10y', 0.0),
    }
    if 'theta1_values' in ckpt:
        best_params['theta1'] = ckpt['theta1_values'].to(device)
        best_params['theta2'] = ckpt['theta2_values'].to(device)
    if 'rho_3m' in ckpt:
        best_params['rho_3m'] = ckpt['rho_3m']
    if 'v0' in ckpt:
        best_params['v0'] = ckpt['v0']
    theta_nodes = ckpt['theta_nodes'].to(device)
    print(f"Loaded {checkpoint_path}  (loss={ckpt.get('loss', '?'):.4e}, "
          f"iter={ckpt.get('iteration', '?')})")
    return best_params, theta_nodes, ckpt


def save_checkpoint(path, best_params, theta_nodes, best_loss, history,
                    delta_atm=None):
    """Persist calibration state to a pickle file."""
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    ckpt = {
        'theta_nodes':  theta_nodes.cpu(),
        'theta_values': best_params.get('theta1', best_params['theta']).cpu(),
        'kappa_fast':   _val(best_params['kappa_fast']),
        'kappa_slow':   _val(best_params['kappa_slow']),
        'w_fast':       _val(best_params['w_fast']),
        'epsilon':      _val(best_params['epsilon']),
        'lam':          _val(best_params['lam']),
        'gamma':        _val(best_params['gamma']),
        'xi':           _val(best_params['xi']),
        'rho_1y':       _val(best_params.get('rho_1y', 0.0)),
        'rho_5y':       _val(best_params.get('rho_5y', 0.0)),
        'rho_10y':      _val(best_params.get('rho_10y', 0.0)),
        'delta_atm':    delta_atm or {},
    }
    if 'theta1' in best_params:
        ckpt['theta1_values'] = best_params['theta1'].cpu()
        ckpt['theta2_values'] = best_params['theta2'].cpu()
    if 'rho_3m' in best_params and best_params['rho_3m'] is not None:
        ckpt['rho_3m'] = _val(best_params['rho_3m'])
    ckpt.update({
        'loss':         best_loss,
        'history':      history,
        'iteration':    len(history),
    })
    if 'v0' in best_params:
        ckpt['v0'] = _val(best_params['v0'])
    with open(path, 'wb') as f:
        pickle.dump(ckpt, f)
    print(f"Saved → {path}  (loss={best_loss:.4e}, {len(history)} iters)")


# ── Batched vol-surface evaluation ────────────────────────────────────────────

def evaluate_vol_surface(best_params, theta_nodes, rho_nodes_t,
                         timeline, f_key_vec, f_ois_vec,
                         strikes, idx_mats, T_fixes,
                         vol_key_rate, fwd_key_rate, fwd_ois,
                         device,
                         n_paths=200_000, n_batches=4,
                         version_name="Fast", seed_base=12345,
                         csv_path=None,
                         # Legacy aliases — ignored, kept for backward compat
                         idx_fixes=None, idx_pays=None, tau=None):
    """Run batched MC, build vol surface, plot, optionally export CSV.

    Returns (vol_results, vol_rmse, model_pvs_final).
    """
    bp = best_params
    batch_size = n_paths // n_batches
    theta1_vec = theta_to_vec(bp.get('theta1', bp['theta']), theta_nodes, timeline)
    theta2_vec = theta_to_vec(bp.get('theta2', bp.get('theta1', bp['theta'])), theta_nodes, timeline)
    if 'rho_3m' in bp:
        rho_nodes_v = torch.tensor(
            [_val(bp['rho_3m']), _val(bp['rho_1y']), _val(bp['rho_5y']), _val(bp['rho_10y'])],
            dtype=torch.float32, device=device)
    else:
        rho_nodes_v = torch.tensor(
            [_val(bp['rho_1y']), _val(bp['rho_5y']), _val(bp['rho_10y'])],
            dtype=torch.float32, device=device)
    rho_vec = rho_to_vec(rho_nodes_v, rho_nodes_t, timeline)
    v0 = _val(bp['v0']) if 'v0' in bp else bp['theta'][0].item()

    torch.cuda.empty_cache()
    n_caplets = len(strikes)
    pvs_acc = torch.zeros(n_caplets, device=device)
    FP_acc  = torch.zeros(n_caplets, device=device)   # accumulates F_b * P_b (numerator of weighted average)
    P_acc   = torch.zeros(n_caplets, device=device)

    for b in range(n_batches):
        print(f"  Batch {b+1}/{n_batches}...", end=" ", flush=True)
        kp, op, _ = fast_simulate_2factor(
            batch_size, timeline, theta1_vec,
            _val(bp['epsilon']), v0,
            _val(bp['kappa_fast']), _val(bp['kappa_slow']), _val(bp['w_fast']),
            _val(bp['lam']), _val(bp['gamma']), _val(bp['xi']),
            f_key_vec, f_ois_vec, device,
            seed=seed_base + b, rho_vx=rho_vec, antithetic=True,
            theta2_vec=theta2_vec,
        )
        pv_b, F_b, P_b = batch_price_caplets(
            kp, op, timeline, idx_mats, T_fixes, strikes, device)
        pvs_acc += pv_b; FP_acc += F_b * P_b; P_acc += P_b
        del kp, op, pv_b, F_b, P_b
        torch.cuda.empty_cache()
        print("done")

    model_pvs = pvs_acc / n_batches
    P_model   = P_acc   / n_batches
    # Correct T-forward measure mean: E^T[avg] = sum_b(F_b * P_b) / sum_b(P_b)
    F_model   = FP_acc / P_acc.clamp(min=1e-12)
    del pvs_acc, FP_acc, P_acc

    # Vol surface
    vkr = vol_key_rate.copy()
    vkr['pv_model_key'] = model_pvs.cpu().numpy()
    vol_results, vol_rmse = generate_caplet_vol_surface(
        vkr, fwd_key_rate, fwd_ois=fwd_ois, version_name=version_name,
        F_model=F_model.cpu().numpy(), P_model=P_model.cpu().numpy(),
    )
    plot_caplet_vol_surface(vol_results, version_name=version_name,
                            fwd_key_rate=fwd_key_rate)

    # Optional CSV export
    if csv_path is not None:
        csv_path = Path(csv_path)
        model_col = vol_results.columns[vol_results.columns.str.startswith('model_vol_')][0]
        error_col = vol_results.columns[vol_results.columns.str.startswith('vol_error_')][0]
        export_cols = [c for c in ['time_to_maturity', 'strike', 'implied_normal_vol',
                       model_col, error_col, 'pv_model_key'] if c in vol_results.columns]
        vol_results[export_cols].to_csv(csv_path, index=False, float_format='%.6f')
        print(f'Exported {len(vol_results)} rows → {csv_path}')

    return vol_results, vol_rmse, model_pvs


# ── Warm start ────────────────────────────────────────────────────────────────

def warm_start(checkpoint_paths, pack_fn, lb, ub,
               theta_nodes, market_vols, T_fixes, device):
    """Load checkpoint or fresh-start, return (p_current, best_params).

    Parameters
    ----------
    checkpoint_paths : list of (Path, str)
        Ordered list of (path, label) to try.
    pack_fn : callable
        v10: ``pack(theta, kf, ks, wf, eps, lam, gam, xi, rho1, rho5, rho10)``
        v14: ``pack(theta, kf, ks, wf, eps, lam, gam, xi, rho1, rho5, rho10, v0)``
    lb, ub : Tensor
        Lower/upper bounds vectors (length 16 or 17).
    """
    n_params = len(lb)
    free_v0  = n_params >= 17
    has_rho3m = n_params >= 20   # 8 theta + 4 rho + 7 scalars + v0 = 20
    sc = 8 if has_rho3m else 6   # scalar params offset (after theta nodes)
    # Bounds indices: theta at 0..sc-1, scalars at sc..sc+6
    theta_min, theta_max = lb[0].item(), ub[0].item()
    kf_min,  kf_max  = lb[sc].item(),   ub[sc].item()
    ks_min,  ks_max  = lb[sc+1].item(), ub[sc+1].item()
    wf_min,  wf_max  = lb[sc+2].item(), ub[sc+2].item()
    eps_min, eps_max = lb[sc+3].item(), ub[sc+3].item()
    lam_min, lam_max = lb[sc+4].item(), ub[sc+4].item()
    gam_min, gam_max = lb[sc+5].item(), ub[sc+5].item()
    xi_min,  xi_max  = lb[sc+6].item(), ub[sc+6].item()

    best_params = None
    for ckpt_path, label in checkpoint_paths:
        if Path(ckpt_path).exists():
            best_params, _, _ = load_checkpoint(ckpt_path, device)
            print(f"  → using {label} for warm start")
            break

    if best_params is not None:
        bv = _val
        theta_ws = best_params['theta'].to(device).clamp(theta_min, theta_max)
        eps_ws   = max(eps_min, min(eps_max, bv(best_params['epsilon'])))
        lam_ws   = max(lam_min, min(lam_max, bv(best_params['lam'])))
        gam_ws   = max(gam_min, min(gam_max, bv(best_params['gamma'])))
        xi_ws    = max(xi_min,  min(xi_max,  bv(best_params['xi'])))
        rho1_ws  = bv(best_params.get('rho_1y',  0.2))
        rho5_ws  = bv(best_params.get('rho_5y',  0.35))
        rho10_ws = bv(best_params.get('rho_10y', 0.55))

        if 'kappa_fast' in best_params:
            kf_ws = max(kf_min, min(kf_max, bv(best_params['kappa_fast'])))
            ks_ws = max(ks_min, min(ks_max, bv(best_params['kappa_slow'])))
            wf_ws = max(wf_min, min(wf_max, bv(best_params['w_fast'])))
            print(f"WARM START: \u03ba_fast={kf_ws:.2f}, \u03ba_slow={ks_ws:.3f}, "
                  f"w={wf_ws:.2f}, \u03b5={eps_ws:.3f}")
        else:
            kf_ws = max(kf_min, min(kf_max, 2.5))
            ks_ws = max(ks_min, min(ks_max, 0.30))
            wf_ws = max(wf_min, min(wf_max, 0.40))
            print(f"WARM START (2-factor defaults): \u03ba_fast={kf_ws:.2f}, "
                  f"\u03ba_slow={ks_ws:.3f}, w={wf_ws:.2f}, \u03b5={eps_ws:.3f}")

        p_current = pack_fn(theta_ws, kf_ws, ks_ws, wf_ws, eps_ws, lam_ws,
                            gam_ws, xi_ws, rho1_ws, rho5_ws, rho10_ws) \
                    if not free_v0 else (
                    pack_fn(theta_ws, kf_ws, ks_ws, wf_ws, eps_ws, lam_ws,
                            gam_ws, xi_ws,
                            _val(best_params.get('rho_3m', rho1_ws)),
                            rho1_ws, rho5_ws, rho10_ws,
                            _val(best_params.get('v0', theta_ws[0].item())))
                    if has_rho3m else
                    pack_fn(theta_ws, kf_ws, ks_ws, wf_ws, eps_ws, lam_ws,
                            gam_ws, xi_ws, rho1_ws, rho5_ws, rho10_ws,
                            _val(best_params.get('v0', theta_ws[0].item()))))
    else:
        # Fresh start: initialise theta from market ATM vols
        tn_np  = theta_nodes.cpu().numpy()
        mv_np  = market_vols.cpu().numpy()
        mT_np  = T_fixes.cpu().numpy()
        theta_init = []
        for tn in tn_np:
            near = np.abs(mT_np - tn) < max(0.15, tn * 0.3)
            avg  = np.median(mv_np[near]) if near.sum() > 0 else np.median(mv_np)
            theta_init.append(np.clip(avg ** 2, theta_min, theta_max))
        theta_t = torch.tensor(theta_init, dtype=torch.float32, device=device)
        v0_fresh = theta_t[0].item()
        if not free_v0:
            p_current = pack_fn(theta_t, 2.5, 0.3, 0.4, 0.4, 0.2, 0.1, 0.005,
                                0.2, 0.35, 0.55)
        elif has_rho3m:
            # rho_3m=-0.2: short-end reverse skew observed in RUB caplet market
            p_current = pack_fn(theta_t, 2.5, 0.3, 0.4, 0.4, 0.2, 0.1, 0.005,
                                -0.2, 0.2, 0.35, 0.55, v0_fresh)
        else:
            p_current = pack_fn(theta_t, 2.5, 0.3, 0.4, 0.4, 0.2, 0.1, 0.005,
                                0.2, 0.35, 0.55, v0_fresh)
        print(f"FRESH START from market ATM vols: \u03b8={theta_t.cpu().numpy()}")

    return p_current, best_params


# ── Experiment log ────────────────────────────────────────────────────────────

_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT, version TEXT, spsa_loss REAL, vol_rmse REAL,
    n_iter INTEGER, n_paths INTEGER, short_boost REAL, kappa REAL,
    kappa_fast REAL, kappa_slow REAL, w_fast REAL, epsilon REAL,
    v0 REAL, lam REAL, gamma REAL, xi REAL,
    rho_1y REAL, rho_5y REAL, rho_10y REAL, theta TEXT, notes TEXT
)"""

_MIGRATE_COLS = [('kappa_fast', 'REAL'), ('kappa_slow', 'REAL'), ('w_fast', 'REAL')]


def _open_runs_db(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_LOG_SCHEMA)
    existing = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
    for col, typ in _MIGRATE_COLS:
        if col not in existing:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")
    conn.commit()
    return conn


def log_run(db_path, best_params, best_loss, full_rmse, history,
            n_paths_grad, version='v10'):
    """Insert one calibration run into the SQLite leaderboard."""
    bp = best_params
    v = _val
    row = {
        'run_at':     datetime.datetime.now().isoformat(timespec='seconds'),
        'version':    version,
        'spsa_loss':  float(best_loss),
        'vol_rmse':   float(full_rmse),
        'n_iter':     int(len(history)),
        'n_paths':    int(n_paths_grad),
        'short_boost': 1.0,
        'kappa':      None,
        'kappa_fast': float(v(bp['kappa_fast'])),
        'kappa_slow': float(v(bp['kappa_slow'])),
        'w_fast':     float(v(bp['w_fast'])),
        'epsilon':    float(v(bp['epsilon'])),
        'v0':         float(_val(bp.get('v0', bp['theta'][0].item()))),
        'lam':        float(v(bp['lam'])),
        'gamma':      float(v(bp['gamma'])),
        'xi':         float(v(bp['xi'])),
        'rho_1y':     float(v(bp.get('rho_1y', 0.0))),
        'rho_5y':     float(v(bp.get('rho_5y', 0.0))),
        'rho_10y':    float(v(bp.get('rho_10y', 0.0))),
        'theta':      json.dumps([round(float(x), 6) for x in bp['theta'].cpu()]),
        'notes':      (f"2-factor Lifted CIR: kf={v(bp['kappa_fast']):.3f}, "
                       f"ks={v(bp['kappa_slow']):.3f}, w={v(bp['w_fast']):.3f}"),
    }
    conn = _open_runs_db(db_path)
    conn.execute(
        "INSERT INTO runs (run_at,version,spsa_loss,vol_rmse,n_iter,n_paths,"
        "short_boost,kappa,kappa_fast,kappa_slow,w_fast,epsilon,v0,lam,gamma,xi,"
        "rho_1y,rho_5y,rho_10y,theta,notes) VALUES "
        "(:run_at,:version,:spsa_loss,:vol_rmse,:n_iter,:n_paths,:short_boost,"
        ":kappa,:kappa_fast,:kappa_slow,:w_fast,:epsilon,:v0,:lam,:gamma,:xi,"
        ":rho_1y,:rho_5y,:rho_10y,:theta,:notes)", row)
    conn.commit()
    conn.close()
    print(f"Logged {version} run -> {db_path}")


def show_leaderboard(db_path):
    """Read all runs from the SQLite DB and return a formatted DataFrame."""
    conn = _open_runs_db(db_path)
    df = pd.read_sql(
        "SELECT id, run_at, version, spsa_loss, vol_rmse, n_iter, n_paths, "
        "kappa_fast, kappa_slow, w_fast, epsilon, v0, "
        "rho_1y, rho_5y, rho_10y FROM runs ORDER BY vol_rmse ASC", conn)
    conn.close()
    if df.empty:
        print("No runs logged yet.")
        return df
    df['vol_rmse_%'] = (df['vol_rmse'] * 100).round(3)
    df['spsa_loss'] = df['spsa_loss'].map('{:.4e}'.format)
    for c in ['kappa_fast', 'kappa_slow', 'w_fast', 'epsilon']:
        df[c] = df[c].round(3)
    df['v0'] = df['v0'].map(
        lambda x: f'{x:.5f}' if pd.notna(x) else 'pinned')
    return df.drop(columns=['vol_rmse'])
