"""MC simulation for the 2-factor Lifted CIR Hull-White-Heston model."""

import numpy as np
import torch
from pyquant.torch_spline import PchipSpline1D


# =============================================================================
# PARAMETER INTERPOLATION
# =============================================================================

def theta_to_vec(theta_vals, theta_nodes, timeline):
    """Interpolate theta(t) node values to the full simulation timeline via PCHIP.

    Parameters
    ----------
    theta_vals  : Tensor (n_nodes,) — theta values at node maturities
    theta_nodes : Tensor (n_nodes,) — node maturities (years)
    timeline    : Tensor (n_steps+1,) — simulation time grid

    Returns
    -------
    Tensor (n_steps+1,) on same device as timeline
    """
    spline = PchipSpline1D(theta_nodes.cpu(), theta_vals.cpu())
    return spline.evaluate(timeline.cpu()).to(timeline.device)


def rho_to_vec(rho_nodes_v, rho_nodes_t, timeline):
    """Interpolate rho(t) nodes to the full simulation timeline via PCHIP.

    Values are clipped to (-0.999, 0.999) to keep sqrt(1-rho^2) real.

    Parameters
    ----------
    rho_nodes_v : Tensor (n_nodes,) — rho values at node maturities
    rho_nodes_t : Tensor (n_nodes,) — node maturities (years)
    timeline    : Tensor (n_steps+1,)

    Returns
    -------
    Tensor (n_steps+1,) on same device as timeline
    """
    spline = PchipSpline1D(rho_nodes_t.cpu(), rho_nodes_v.cpu())
    rho_full = spline.evaluate(timeline.cpu()).to(timeline.device)
    return torch.clamp(rho_full, -0.999, 0.999)


# =============================================================================
# BUILDING-BLOCK PATH GENERATORS
# =============================================================================

def fast_cir_paths(n_paths, n_steps, dt, v0, kappa, theta_vec, epsilon, device, randn=None):
    """Euler-Maruyama CIR: dV = kappa*(theta(t)-V) dt + epsilon*sqrt(V) dW, reflected at 0."""
    v = torch.zeros((n_paths, n_steps + 1), dtype=torch.float32, device=device)
    v[:, 0] = float(v0)
    sqrt_dt = float(np.sqrt(dt))
    kappa_f = float(kappa)
    eps_f = float(epsilon)
    if randn is None:
        randn = torch.randn(n_paths, n_steps, device=device)
    for i in range(n_steps):
        v_curr = v[:, i]
        drift = kappa_f * (theta_vec[i] - v_curr) * dt
        diffusion = eps_f * torch.sqrt(torch.clamp(v_curr, min=1e-8)) * sqrt_dt * randn[:, i]
        v[:, i + 1] = torch.abs(v_curr + drift + diffusion)
    return v


def fast_ou_paths(n_paths, n_steps, dt, x0, lam, vol_paths, device, randn=None):
    """OU with stochastic vol: dx = -lambda*x dt + sqrt(V) dW_x."""
    x = torch.zeros((n_paths, n_steps + 1), dtype=torch.float32, device=device)
    x[:, 0] = float(x0)
    sqrt_dt = float(np.sqrt(dt))
    exp_lam_dt = float(np.exp(-float(lam) * dt))
    if randn is None:
        randn = torch.randn(n_paths, n_steps, device=device)
    for i in range(n_steps):
        x[:, i + 1] = (x[:, i] * exp_lam_dt
                       + torch.sqrt(torch.clamp(vol_paths[:, i], min=1e-9))
                       * sqrt_dt * randn[:, i])
    return x


def fast_hw_paths(n_paths, n_steps, dt, x0, gamma, xi, device, randn=None):
    """HW/OU spread with constant vol: dk = -gamma*k dt + xi dW_k (exact Gaussian transitions)."""
    k = torch.zeros((n_paths, n_steps + 1), dtype=torch.float32, device=device)
    k[:, 0] = float(x0)
    gamma_f = float(gamma)
    xi_f = float(xi)
    exp_gamma_dt = float(np.exp(-gamma_f * dt))
    var_factor = xi_f * xi_f * (1.0 - np.exp(-2.0 * gamma_f * dt)) / max(2.0 * gamma_f, 1e-8)
    std_factor = float(np.sqrt(max(var_factor, 1e-12)))
    if randn is None:
        randn = torch.randn(n_paths, n_steps, device=device)
    for i in range(n_steps):
        k[:, i + 1] = k[:, i] * exp_gamma_dt + std_factor * randn[:, i]
    return k


# =============================================================================
# 2-FACTOR HWH SIMULATOR
# =============================================================================

def _run_sim_2f(n_paths, n_steps, dt, theta_vec, epsilon, v0,
                kf, ks, wf, lam, gam, xi,
                f_key_vec, f_ois_vec, device,
                randn_v1, randn_v2, randn_x_perp, randn_k, rho_vec,
                theta2_vec=None):
    """Single-pass inner kernel (no antithetic)."""
    if theta2_vec is None:
        theta2_vec = theta_vec
    v1 = fast_cir_paths(n_paths, n_steps, dt, v0, kf, theta_vec,  epsilon, device, randn_v1)
    v2 = fast_cir_paths(n_paths, n_steps, dt, v0, ks, theta2_vec, epsilon, device, randn_v2)
    v_paths = float(wf) * v1 + (1.0 - float(wf)) * v2

    rho_t = rho_vec[:n_steps]
    sqrt_1_rho2 = torch.sqrt(torch.clamp(1.0 - rho_t ** 2, min=1e-6))
    randn_x = rho_t.unsqueeze(0) * randn_v1 + sqrt_1_rho2.unsqueeze(0) * randn_x_perp

    x_paths = fast_ou_paths(n_paths, n_steps, dt, 0.0, lam, v_paths, device, randn_x)
    ks_paths = fast_hw_paths(n_paths, n_steps, dt, 0.0, gam, xi, device, randn_k)

    key_paths = f_key_vec.unsqueeze(0) + x_paths + ks_paths
    ois_paths = f_ois_vec.unsqueeze(0) + x_paths
    return key_paths, ois_paths, v_paths


def fast_simulate_2factor(n_paths, timeline, theta_vec, epsilon, v0,
                          kf, ks, wf, lam, gam, xi,
                          f_key_vec, f_ois_vec, device,
                          seed=None, rho_vx=None, antithetic=False,
                          theta2_vec=None):
    """Simulate the 2-factor Lifted CIR Hull-White-Heston model.

    Model:
        V = wf*V1(kf) + (1-wf)*V2(ks)
        dV1 = kf*(theta(t)-V1) dt + eps*sqrt(V1) dW^v1
        dV2 = ks*(theta(t)-V2) dt + eps*sqrt(V2) dW^v2
        dx  = -lam*x dt + sqrt(V) dW^x,   dW^x = rho(t)*dW^v1 + sqrt(1-rho^2)*dW^x_perp
        dk  = -gam*k dt + xi dW^k
        r_key = f_key(t) + x + k
        r_ois = f_ois(t) + x

    Returns (key_paths, ois_paths, v_paths), each (n_eff, n_steps+1).
    n_eff = 2*n_paths when antithetic=True.
    """
    n_steps = len(timeline) - 1
    dt = (timeline[1] - timeline[0]).item()
    if seed is not None:
        torch.manual_seed(seed)

    _f = lambda x: float(x) if isinstance(x, torch.Tensor) else float(x)
    kf_f, ks_f, wf_f = _f(kf), _f(ks), _f(wf)
    lam_f, gam_f, xi_f = _f(lam), _f(gam), _f(xi)
    eps_f, v0_f = _f(epsilon), _f(v0)

    randn_v1     = torch.randn(n_paths, n_steps, device=device)
    randn_v2     = torch.randn(n_paths, n_steps, device=device)
    randn_x_perp = torch.randn(n_paths, n_steps, device=device)
    randn_k      = torch.randn(n_paths, n_steps, device=device)
    rho_vec = (rho_vx[:n_steps + 1].to(device)
               if rho_vx is not None
               else torch.zeros(n_steps + 1, device=device))

    kp, op, vp = _run_sim_2f(
        n_paths, n_steps, dt, theta_vec, eps_f, v0_f,
        kf_f, ks_f, wf_f, lam_f, gam_f, xi_f,
        f_key_vec, f_ois_vec, device,
        randn_v1, randn_v2, randn_x_perp, randn_k, rho_vec,
        theta2_vec=theta2_vec,
    )

    if antithetic:
        kp_a, op_a, vp_a = _run_sim_2f(
            n_paths, n_steps, dt, theta_vec, eps_f, v0_f,
            kf_f, ks_f, wf_f, lam_f, gam_f, xi_f,
            f_key_vec, f_ois_vec, device,
            -randn_v1, -randn_v2, -randn_x_perp, -randn_k, rho_vec,
            theta2_vec=theta2_vec,
        )
        kp = torch.cat([kp, kp_a], dim=0)
        op = torch.cat([op, op_a], dim=0)
        vp = torch.cat([vp, vp_a], dim=0)

    return kp, op, vp


# =============================================================================
# CAPLET PRICER
# =============================================================================

def batch_price_caplets(key_paths, ois_paths, timeline, idx_mats, T_fixes_vals,
                        strikes, device):
    """Price now-starting average-rate caplets by Monte Carlo.

    Each caplet averages the key rate over [0, T] and pays at T:

        PV_c = E[ T * max(mean(r_key[0:i_mat+1]) - K, 0) * exp(-sum(r_ois[0:i_mat+1])*dt) ]

    This matches compute_market_pvs, which uses:
        PV = T * disc * Bachelier(F, K, sigma * sqrt(T/3))
    where F = I(0,T)/T (average instantaneous forward from 0 to T).

    Parameters
    ----------
    key_paths    : Tensor (n_paths, n_steps) — simulated key rate paths
    ois_paths    : Tensor (n_paths, n_steps) — simulated OIS rate paths
    timeline     : Tensor (n_steps,)          — time grid
    idx_mats     : Tensor (n_caplets,)        — searchsorted(timeline, T_fixes)
    T_fixes_vals : Tensor (n_caplets,)        — caplet maturities T in years
    strikes      : Tensor (n_caplets,)        — caplet strikes K
    device       : torch.device

    Returns (model_pvs, F_model, P_model), each (n_caplets,).
    """
    n_caplets = len(strikes)
    dt = (timeline[1] - timeline[0]).item()

    model_pvs = torch.zeros(n_caplets, dtype=torch.float32, device=device)
    F_model   = torch.zeros(n_caplets, dtype=torch.float32, device=device)
    P_model   = torch.zeros(n_caplets, dtype=torch.float32, device=device)

    for c in range(n_caplets):
        i_mat = int(idx_mats[c].item() if isinstance(idx_mats[c], torch.Tensor) else idx_mats[c])
        T_c   = float(T_fixes_vals[c].item() if isinstance(T_fixes_vals[c], torch.Tensor) else T_fixes_vals[c])
        K     = float(strikes[c].item() if isinstance(strikes[c], torch.Tensor) else strikes[c])

        # Average key rate over [0, T] (now-starting)
        L_realized = key_paths[:, :i_mat + 1].mean(dim=1)
        # Discount factor to T (not T+tau)
        disc = torch.exp(-ois_paths[:, :i_mat + 1].sum(dim=1) * dt)

        # Payoff multiplier = T (averaging period length)
        payoff = torch.clamp(L_realized - K, min=0.0) * T_c * disc
        model_pvs[c] = payoff.mean()
        P_model[c]   = disc.mean()
        # T-forward measure expectation: E^T[avg] = E[disc*avg] / E[disc].
        # Using the physical-measure mean E[avg] overestimates F (because disc and
        # avg_rate are negatively correlated), making intrinsic > PV/(T*P) and
        # causing the time-value residual to clamp to zero for every deep-ITM
        # caplet → kills the gradient signal for the entire ITM region.
        F_model[c]   = (L_realized * disc).mean() / P_model[c].clamp(min=1e-12)

    return model_pvs, F_model, P_model
