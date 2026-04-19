"""Zeroth-order optimization algorithms (SPSA and Adam-SPSA).

Decoupled from pricing/loss logic — work with any eval_loss_fn signature.
Use these with make_eval_loss() or custom objective factories.
"""

import torch
import numpy as np
import pandas as pd
import time
from pathlib import Path
from IPython.display import display, clear_output


_val = lambda x: x.item() if isinstance(x, torch.Tensor) else float(x)


def run_spsa(eval_loss, unpack, p_current, *,
             max_iter, n_paths_grad, alpha, gamma_spsa, a, c, A, seed_base,
             param_names, param_bounds, param_range, n_params,
             theta_nodes, autosave_path,
             market_pvs_cached, market_vegas_floored,
             autosave_interval=200, version='v10',
             n_perturbations=1):
    """Run the full SPSA loop with live display, autosave, and MC variance check.

    Parameters
    ----------
    eval_loss : callable
        (p, n_paths, seed) → (loss, model_pvs)
    unpack : callable
        p → (theta, kf, ks, wf, eps, lam, gam, xi, rho1, rho5, rho10, [v0])
    p_current : Tensor [n_params] ∈ [0,1]
    max_iter, n_paths_grad : int
        Max iterations, paths per gradient eval
    alpha, gamma_spsa, a, c, A : float
        SPSA step-size schedule: a_k = a/(k+1+A)^α, c_k = c/(k+1)^γ
    n_perturbations : int
        Average this many independent SPSA gradients per step
        (costs 2× more evals per extra perturbation, gives cleaner gradient)

    Returns
    -------
    best_params, best_loss, history, model_pvs_k
    """
    autosave_path = Path(autosave_path)
    autosave_path.parent.mkdir(exist_ok=True)
    device = p_current.device

    best_loss   = float('inf')
    best_p      = p_current.clone()
    best_params = None
    best_theta  = None
    history     = []
    model_pvs_k = None
    start_time  = time.time()

    for k in range(max_iter):
        a_k = a / (k + 1 + A) ** alpha
        c_k = c / (k + 1) ** gamma_spsa

        # Average n_perturbations independent SPSA gradients
        g_acc = torch.zeros(n_params, device=device)
        loss_acc = 0.0
        for q in range(n_perturbations):
            delta = 2.0 * torch.bernoulli(0.5 * torch.ones(n_params, device=device)) - 1.0
            p_plus  = torch.clamp(p_current + c_k * delta, 0.0, 1.0)
            p_minus = torch.clamp(p_current - c_k * delta, 0.0, 1.0)
            seed_k = seed_base + k * n_perturbations + q
            loss_plus,  _           = eval_loss(p_plus,  n_paths_grad, seed_k)
            loss_minus, model_pvs_k = eval_loss(p_minus, n_paths_grad, seed_k)
            g_acc += (loss_plus.item() - loss_minus.item()) / (2.0 * c_k * delta)
            loss_acc += 0.5 * (loss_plus.item() + loss_minus.item())
        g_k = g_acc / n_perturbations
        loss_avg = loss_acc / n_perturbations
        p_current = torch.clamp(p_current - a_k * g_k, 0.0, 1.0)
        result_k = unpack(p_current)
        two_theta = len(result_k) == 13
        free_v0   = len(result_k) in (12, 13)
        if two_theta:
            theta1_k, theta2_k, kf_k, ks_k, wf_k, eps_k, lam_k, gam_k, xi_k, \
                rho1_k, rho5_k, rho10_k, v0_k = result_k
            theta_k = theta1_k
        elif free_v0:
            theta_k, kf_k, ks_k, wf_k, eps_k, lam_k, gam_k, xi_k, \
                rho1_k, rho5_k, rho10_k, v0_k = result_k
        else:
            theta_k, kf_k, ks_k, wf_k, eps_k, lam_k, gam_k, xi_k, \
                rho1_k, rho5_k, rho10_k = result_k
            v0_k = theta_k[0].item()

        if loss_avg < best_loss:
            best_loss  = loss_avg
            best_p     = p_current.clone()
            best_theta = theta_k.clone()
            best_params = {
                'theta': theta_k.clone(),
                'kappa_fast': kf_k, 'kappa_slow': ks_k, 'w_fast': wf_k,
                'epsilon': eps_k, 'lam': lam_k, 'gamma': gam_k, 'xi': xi_k,
                'rho_1y': rho1_k, 'rho_5y': rho5_k, 'rho_10y': rho10_k,
            }
            if two_theta:
                best_params['theta1'] = theta1_k.clone()
                best_params['theta2'] = theta2_k.clone()
            if free_v0:
                best_params['v0'] = v0_k

        g_orig = g_k / param_range
        hist_entry = {
            'iter': k, 'loss': loss_avg,
            'theta': theta_k.cpu().numpy().copy(),
            'kappa_fast': kf_k, 'kappa_slow': ks_k, 'w_fast': wf_k,
            'epsilon': eps_k, 'lam': lam_k, 'gamma': gam_k, 'xi': xi_k,
            'rho_1y': rho1_k, 'rho_5y': rho5_k, 'rho_10y': rho10_k,
            'grad_raw': g_orig.cpu().numpy().copy(),
        }
        if n_params >= 17:
            hist_entry.update({
                'grad_theta': g_orig[:6].cpu().numpy().copy(),
                'grad_kf': g_orig[6].item(), 'grad_ks': g_orig[7].item(),
                'grad_wf': g_orig[8].item(), 'grad_eps': g_orig[9].item(),
                'grad_lam': g_orig[10].item(), 'grad_gamma': g_orig[11].item(),
                'grad_xi': g_orig[12].item(), 'grad_rho1': g_orig[13].item(),
                'grad_rho5': g_orig[14].item(), 'grad_rho10': g_orig[15].item(),
            })
        if free_v0:
            hist_entry['v0'] = v0_k
        history.append(hist_entry)

        # Periodic auto-save
        if best_params is not None and (k + 1) % autosave_interval == 0:
            from affine_calibration.scripts.calib_objective import save_checkpoint
            save_checkpoint(autosave_path, best_params, theta_nodes,
                            best_loss, history)

        # Live display
        if k % 25 == 0:
            elapsed = time.time() - start_time
            lbs_d = [param_bounds[n][0] for n in param_names]
            ubs_d = [param_bounds[n][1] for n in param_names]
            lbs_t = torch.tensor(lbs_d, dtype=p_current.dtype, device=p_current.device)
            ubs_t = torch.tensor(ubs_d, dtype=p_current.dtype, device=p_current.device)
            values = (lbs_t + p_current * (ubs_t - lbs_t)).cpu().tolist()
            go = g_k.cpu().numpy()
            snapshot = pd.DataFrame({
                'Value':    [f'{v:.6f}' for v in values],
                'Gradient': [f'{g:.4f}'  for g in go],
                'LB':       [f'{b:.5f}'  for b in lbs_d],
                'UB':       [f'{b:.5f}'  for b in ubs_d],
            }, index=param_names)

            clear_output(wait=True)
            vol_err = (model_pvs_k - market_pvs_cached) / market_vegas_floored
            full_rmse = torch.sqrt(torch.mean(vol_err ** 2)).item()
            asi = ((k + 1) // autosave_interval) * autosave_interval
            if n_params >= 17:
                hl_f = np.log(2) / kf_k
                hl_s = np.log(2) / ks_k
                print(
                    f"SPSA {version} {k}/{max_iter} | Filt {loss_avg:.4e} | "
                    f"Full {full_rmse:.4e} | Best {best_loss:.4e} | "
                    f"κ_f={kf_k:.2f}(hl={hl_f:.2f}Y) "
                    f"κ_s={ks_k:.3f}(hl={hl_s:.1f}Y) "
                    f"w={wf_k:.2f} ε={eps_k:.4f} | "
                    f"ρ=[{rho1_k:+.2f},{rho5_k:+.2f},{rho10_k:+.2f}] | "
                    f"saved@{asi} | {elapsed:.0f}s")
            else:
                print(
                    f"SPSA {version} {k}/{max_iter} | Filt {loss_avg:.4e} | "
                    f"Full {full_rmse:.4e} | Best {best_loss:.4e} | "
                    f"saved@{asi} | {elapsed:.0f}s")
            display(snapshot)

    # Final save
    if best_params is not None:
        from affine_calibration.scripts.calib_objective import save_checkpoint
        save_checkpoint(autosave_path, best_params, theta_nodes,
                        best_loss, history)

    # MC variance check
    loss_samples = []
    for s in range(5):
        l, _ = eval_loss(best_p, 12000, seed=9999 + s * 7)
        loss_samples.append(l.item())
    loss_arr = np.array(loss_samples)
    print(f"Loss over 5 seeds (12k paths): mean={loss_arr.mean():.4e}, "
          f"std={loss_arr.std():.4e}, CV={loss_arr.std()/loss_arr.mean()*100:.1f}%")

    # Final results table
    elapsed_total = time.time() - start_time
    bp = best_params
    # v18 two-theta: concatenate theta1 + theta2 so len matches 23 param_names
    if two_theta and 'theta1' in bp and 'theta2' in bp:
        tc = np.concatenate([bp['theta1'].cpu().numpy(), bp['theta2'].cpu().numpy()])
    else:
        tc = best_theta.cpu().numpy()
    values = list(tc) + [_val(bp.get(k, 0.0)) for k in
                         ['kappa_fast', 'kappa_slow', 'w_fast',
                          'epsilon', 'lam', 'gamma', 'xi',
                          'rho_1y', 'rho_5y', 'rho_10y'] if k in bp]
    if free_v0 and 'v0' in bp:
        values.append(_val(bp['v0']))
    values = values[:len(param_names)]
    lbs_d = [param_bounds[n][0] for n in param_names]
    ubs_d = [param_bounds[n][1] for n in param_names]
    final_df = pd.DataFrame({'Value': values, 'LB': lbs_d, 'UB': ubs_d},
                            index=param_names)
    kf = bp.get('kappa_fast', 0.0); ks = bp.get('kappa_slow', 0.0)
    wf = bp.get('w_fast', 0.0)
    v0_disp = _val(bp['v0']) if 'v0' in bp else tc[0]
    print(f"\nSPSA {version} Done in {elapsed_total:.1f}s — "
          f"Best per-bucket RMSE loss: {best_loss:.4e}")
    print(f"  {max_iter} iter × {2*n_perturbations} evals × {n_paths_grad} paths = "
          f"{max_iter * 2 * n_perturbations * n_paths_grad:,} total MC paths")
    if kf > 0 and ks > 0:
        print(f"  κ_fast={kf:.3f} (hl={np.log(2)/kf:.2f}Y)  "
              f"κ_slow={ks:.3f} (hl={np.log(2)/ks:.2f}Y)  w_fast={wf:.3f}")
    print(f"  v0 = {v0_disp:.6f}" +
          (" (free)" if 'v0' in bp else " (pinned=θ(3M))"))
    if 'epsilon' in bp:
        print(f"  ε={bp['epsilon']:.4f}  "
              f"ρ=[{bp['rho_1y']:+.3f}, {bp['rho_5y']:+.3f}, {bp['rho_10y']:+.3f}]")
    display(final_df)

    return best_params, best_loss, history, model_pvs_k


def run_adam_spsa(eval_loss_fn, unpack, p_init, *,
                  max_iter, n_paths, lr=5e-3, c=0.02, seed_base=42,
                  n_perturbations=3,
                  param_names, param_bounds, param_range, n_params,
                  theta_nodes, autosave_path,
                  market_pvs_cached, market_vegas_floored,
                  autosave_interval=200,
                  patience=800, min_improvement=1e-6,
                  lr_schedule='cosine', lr_min_ratio=0.05,
                  c_schedule='cosine', c_min_ratio=0.2,
                  grad_clip=10.0,
                  n_restarts=1, reset_momentum_on_restart=True,
                  version='v15'):
    """Zeroth-order Adam: SPSA finite-difference gradients + Adam momentum.

    Why not AD-Adam?
    ~~~~~~~~~~~~~~~~
    Backpropagating through 3651 sequential Euler–Maruyama steps is unstable —
    gradient variance grows exponentially, causing NaN within tens of iterations.

    This keeps evaluation exactly like SPSA (no autograd, no backprop) but uses
    Adam's adaptive per-parameter learning rates and momentum instead of hand-tuned
    a_k / c_k schedules. Result:
        - Stable (no NaN ever — same forward-only MC as SPSA)
        - Faster convergence (momentum smooths MC noise)
        - Averaged perturbations reduce gradient variance

    Parameters
    ----------
    eval_loss_fn : callable
        (p, n_paths, seed) → (loss, model_pvs)
    n_perturbations : int
        Number of independent SPSA perturbations averaged per step (default 3)
    c : float
        Perturbation size in unit-cube (default 0.02 = 2% of range)
    lr_schedule : str
        'cosine' (default) — cosine annealing from lr to lr*lr_min_ratio over
        max_iter steps. Prevents oscillation at the noise floor near convergence.
        'none' — constant LR (original behaviour).
    lr_min_ratio : float
        Floor LR as fraction of peak: lr_min = lr * lr_min_ratio (default 0.05).
    c_schedule : str
        'cosine' (default) — cosine annealing of the SPSA perturbation size c
        from c to c*c_min_ratio. Smaller c near convergence reduces gradient bias,
        improving fine-tuning quality at the cost of slightly higher variance.
        'none' — constant c (original behaviour).
    c_min_ratio : float
        Floor c as fraction of peak: c_min = c * c_min_ratio (default 0.2).
    """
    import math
    autosave_path = Path(autosave_path)
    autosave_path.parent.mkdir(exist_ok=True)
    device = p_init.device

    p = p_init.clone().to(device)

    # Adam state (manual implementation — no autograd needed)
    m = torch.zeros_like(p)         # 1st moment
    v = torch.zeros_like(p)         # 2nd moment
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    best_loss   = float('inf')
    best_p      = p.clone()
    best_params = None
    best_theta  = None
    history     = []
    model_pvs_k = None
    no_improve_count = 0
    start_time  = time.time()

    for k in range(max_iter):
        seed_k = seed_base + k * n_perturbations * 2

        # ---- LR + c schedules (SGDR: n_restarts cosine cycles) ----
        period      = max(max_iter // max(n_restarts, 1), 1)
        k_in_period = k % period
        # Reset Adam momentum at each SGDR restart so optimizer re-explores
        if reset_momentum_on_restart and n_restarts > 1 and k > 0 and k_in_period == 0:
            m = torch.zeros_like(p)
            v = torch.zeros_like(p)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * k_in_period / period))
        lr_k = lr * (lr_min_ratio + (1.0 - lr_min_ratio) * cos_factor) if lr_schedule == 'cosine' else lr
        c_k  = c  * (c_min_ratio  + (1.0 - c_min_ratio)  * cos_factor) if c_schedule  == 'cosine' else c

        # ---- Averaged SPSA gradient ----
        g_acc = torch.zeros_like(p)
        loss_acc = 0.0
        for q in range(n_perturbations):
            delta = 2.0 * torch.bernoulli(0.5 * torch.ones(n_params, device=device)) - 1.0
            p_plus  = torch.clamp(p + c_k * delta, 0.0, 1.0)
            p_minus = torch.clamp(p - c_k * delta, 0.0, 1.0)
            s = seed_k + q * 2
            loss_p, _           = eval_loss_fn(p_plus,  n_paths, s)
            loss_m, model_pvs_q = eval_loss_fn(p_minus, n_paths, s)
            g_acc += (loss_p.item() - loss_m.item()) / (2.0 * c_k * delta)
            loss_acc += 0.5 * (loss_p.item() + loss_m.item())
        g_k = g_acc / n_perturbations
        loss_avg = loss_acc / n_perturbations
        model_pvs_k = model_pvs_q  # last perturbation's model PVs

        # ---- Gradient norm clipping (prevents MC noise spikes killing params) ----
        if grad_clip > 0.0:
            g_norm = torch.norm(g_k)
            if g_norm > grad_clip:
                g_k = g_k * (grad_clip / g_norm)

        # ---- Adam update (manual) ----
        m = beta1 * m + (1 - beta1) * g_k
        v = beta2 * v + (1 - beta2) * g_k ** 2
        m_hat = m / (1 - beta1 ** (k + 1))
        v_hat = v / (1 - beta2 ** (k + 1))
        p = p - lr_k * m_hat / (torch.sqrt(v_hat) + eps_adam)
        p = torch.clamp(p, 0.0, 1.0)

        # ---- Unpack for logging ----
        result_k = unpack(p)
        g_np = g_k.cpu().numpy()

        # Two-theta v18/v19: (theta1, theta2, kf, ks, wf, eps, lam, gam, xi, rho1, rho5, rho10, v0)
        two_theta_adam = (len(result_k) == 13 and
                          isinstance(result_k[0], torch.Tensor) and
                          isinstance(result_k[1], torch.Tensor) and
                          result_k[0].shape == result_k[1].shape)
        if two_theta_adam:
            theta1_k, theta2_k, kf_k, ks_k, wf_k, eps_k, lam_k, gam_k, xi_k, \
                rho1_k, rho5_k, rho10_k, v0_k = result_k
            theta_k  = theta1_k
            rho3m_k  = None
            n_theta_k = theta1_k.shape[0] * 2  # total theta params in p
            sc = n_theta_k
        else:
            n_theta_k = result_k[0].shape[0]
            has_rho3m = len(result_k) == 13
            if has_rho3m:
                theta_k, kf_k, ks_k, wf_k, eps_k, lam_k, gam_k, xi_k, \
                    rho3m_k, rho1_k, rho5_k, rho10_k, v0_k = result_k
            else:
                theta_k, kf_k, ks_k, wf_k, eps_k, lam_k, gam_k, xi_k, \
                    rho1_k, rho5_k, rho10_k, v0_k = result_k
                rho3m_k = None
            sc = n_theta_k

        if loss_avg < best_loss - min_improvement:
            best_loss  = loss_avg
            best_p     = p.clone()
            best_theta = theta_k.clone()
            best_params = {
                'theta': theta_k.clone(), 'kappa_fast': kf_k,
                'kappa_slow': ks_k, 'w_fast': wf_k,
                'epsilon': eps_k, 'lam': lam_k, 'gamma': gam_k, 'xi': xi_k,
                'rho_1y': rho1_k, 'rho_5y': rho5_k, 'rho_10y': rho10_k,
                'v0': v0_k,
            }
            if two_theta_adam:
                best_params['theta1'] = theta1_k.clone()
                best_params['theta2'] = theta2_k.clone()
            if rho3m_k is not None:
                best_params['rho_3m'] = rho3m_k
            no_improve_count = 0
        else:
            no_improve_count += 1

        # History
        hist_entry = {
            'iter': k, 'loss': loss_avg,
            'theta': theta_k.cpu().numpy().copy(),
            'kappa_fast': kf_k, 'kappa_slow': ks_k, 'w_fast': wf_k,
            'epsilon': eps_k, 'lam': lam_k, 'gamma': gam_k, 'xi': xi_k,
            'rho_1y': rho1_k, 'rho_5y': rho5_k, 'rho_10y': rho10_k,
            'v0': v0_k,
            'grad_theta': g_np[:sc],
            'grad_kf': g_np[sc],   'grad_ks': g_np[sc+1],
            'grad_wf': g_np[sc+2], 'grad_eps': g_np[sc+3],
            'grad_lam': g_np[sc+4], 'grad_gamma': g_np[sc+5],
            'grad_xi':  g_np[sc+6],
            'grad_rho1': g_np[sc+8 if (rho3m_k is not None and not two_theta_adam) else sc+7],
            'grad_rho5': g_np[sc+9 if (rho3m_k is not None and not two_theta_adam) else sc+8],
            'grad_rho10': g_np[sc+10 if (rho3m_k is not None and not two_theta_adam) else sc+9],
        }
        if two_theta_adam:
            hist_entry['theta2'] = theta2_k.cpu().numpy().copy()
        if rho3m_k is not None:
            hist_entry['rho_3m'] = rho3m_k
            hist_entry['grad_rho3m'] = g_np[sc+7]
        history.append(hist_entry)

        # Auto-save
        if best_params is not None and (k + 1) % autosave_interval == 0:
            from affine_calibration.scripts.calib_objective import save_checkpoint
            save_checkpoint(autosave_path, best_params, theta_nodes,
                            best_loss, history)

        # Live display
        if k % 25 == 0:
            elapsed = time.time() - start_time
            vol_err = (model_pvs_k - market_pvs_cached) / market_vegas_floored
            full_rmse = torch.sqrt(torch.mean(vol_err ** 2)).item()
            hl_f = np.log(2) / max(kf_k, 1e-6)
            hl_s = np.log(2) / max(ks_k, 1e-6)
            grad_norm = np.linalg.norm(g_np)
            clear_output(wait=True)
            rho_str = (f"[{rho3m_k:+.2f},{rho1_k:+.2f},{rho5_k:+.2f},{rho10_k:+.2f}]"
                       if rho3m_k is not None
                       else f"[{rho1_k:+.2f},{rho5_k:+.2f},{rho10_k:+.2f}]")
            print(
                f"ZO-Adam {version} {k}/{max_iter} | Filt {loss_avg:.4e} | "
                f"Full {full_rmse:.4e} | Best {best_loss:.4e} | "
                f"κ_f={kf_k:.2f}(hl={hl_f:.2f}Y) "
                f"κ_s={ks_k:.3f}(hl={hl_s:.1f}Y) "
                f"w={wf_k:.2f} ε={eps_k:.4f} v0={v0_k:.6f} | "
                f"ρ={rho_str} | "
                f"|g|={grad_norm:.3f} lr={lr_k:.2e} c={c_k:.3f} | "
                f"cyc={k // period + 1}/{n_restarts} pat={no_improve_count}/{patience} | "
                f"{elapsed:.0f}s")

        # Early stopping
        if no_improve_count >= patience:
            print(f"\nEarly stopping at iter {k}: no improvement for {patience} iters")
            break

    # Final save
    if best_params is not None:
        from affine_calibration.scripts.calib_objective import save_checkpoint
        save_checkpoint(autosave_path, best_params, theta_nodes,
                        best_loss, history)

    # MC variance check
    loss_samples = []
    for s in range(5):
        l, _ = eval_loss_fn(best_p, max(n_paths, 12000), seed=9999 + s * 7)
        loss_samples.append(l.item())
    loss_arr = np.array(loss_samples)
    print(f"\nLoss over 5 seeds ({max(n_paths,12000)} paths): "
          f"mean={loss_arr.mean():.4e}, std={loss_arr.std():.4e}, "
          f"CV={loss_arr.std()/loss_arr.mean()*100:.1f}%")

    # Final table
    elapsed_total = time.time() - start_time
    bp = best_params
    n_evals = (k + 1) * n_perturbations * 2
    # Two-theta (v18/v19): concatenate theta1 + theta2 so values aligns with param_names
    if 'theta1' in bp and 'theta2' in bp:
        tc = np.concatenate([bp['theta1'].cpu().numpy(), bp['theta2'].cpu().numpy()])
    else:
        tc = best_theta.cpu().numpy()
    if 'rho_3m' in bp:
        values = list(tc) + [bp['kappa_fast'], bp['kappa_slow'], bp['w_fast'],
                             bp['epsilon'], bp['lam'], bp['gamma'], bp['xi'],
                             bp['rho_3m'], bp['rho_1y'], bp['rho_5y'], bp['rho_10y'], bp['v0']]
    else:
        values = list(tc) + [bp['kappa_fast'], bp['kappa_slow'], bp['w_fast'],
                             bp['epsilon'], bp['lam'], bp['gamma'], bp['xi'],
                             bp['rho_1y'], bp['rho_5y'], bp['rho_10y'], bp['v0']]
    lbs_d = [param_bounds[n][0] for n in param_names]
    ubs_d = [param_bounds[n][1] for n in param_names]
    final_df = pd.DataFrame({'Value': values, 'LB': lbs_d, 'UB': ubs_d},
                            index=param_names)
    print(f"\nZO-Adam {version} Done in {elapsed_total:.1f}s — "
          f"Best per-bucket RMSE loss: {best_loss:.4e}")
    print(f"  {k+1} iter × {n_perturbations*2} evals × {n_paths} paths = "
          f"{n_evals * n_paths:,} total MC paths")
    print(f"  κ_fast={bp['kappa_fast']:.3f} (hl={np.log(2)/bp['kappa_fast']:.2f}Y)  "
          f"κ_slow={bp['kappa_slow']:.3f} (hl={np.log(2)/bp['kappa_slow']:.2f}Y)  "
          f"w_fast={bp['w_fast']:.3f}")
    print(f"  v0 = {bp['v0']:.6f} (free)")
    rho_str_f = (f"[{bp['rho_3m']:+.3f}, {bp['rho_1y']:+.3f}, {bp['rho_5y']:+.3f}, {bp['rho_10y']:+.3f}]"
                 if 'rho_3m' in bp
                 else f"[{bp['rho_1y']:+.3f}, {bp['rho_5y']:+.3f}, {bp['rho_10y']:+.3f}]")
    print(f"  ε={bp['epsilon']:.4f}  ρ={rho_str_f}")
    display(final_df)

    return best_params, best_loss, history, model_pvs_k
