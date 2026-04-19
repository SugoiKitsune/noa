"""Parameter space definitions for HWH caplet calibration.

Owns bounds, pack/unpack, and node tensors for each model version.
Import ``make_param_space(device, version='v17')`` in notebooks.

Supported versions
------------------
'v16' : 17 params, v0 free, vega-weighted PV loss
'v17' : 17 params, v0 free, direct vol RMSE (all strikes)
'v18' : 23 params, independent theta per CIR factor, theta_min raised to 1e-3
'v19' : same as v18 (23 params, two independent theta curves) — new checkpoint slot,
        optimized with Adam-SPSA (warm-starts from v18 or v17)
'v20' : same space as v18/v19, cosine LR+c annealing
'v21' : same space as v20, adds SGDR warm restarts + gradient clipping + smart_init
"""

import pickle
from pathlib import Path
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Shared node tensors
# ---------------------------------------------------------------------------

THETA_NODES = [0.25, 0.5, 1.0, 3.0, 5.0, 10.0]
RHO_NODES_T = [1.0, 5.0, 10.0]

PARAM_NAMES = [
    'θ(3M)', 'θ(6M)', 'θ(1Y)', 'θ(3Y)', 'θ(5Y)', 'θ(10Y)',
    'κ_fast', 'κ_slow', 'w_fast', 'ε', 'λ', 'γ', 'ξ',
    'ρ(1Y)', 'ρ(5Y)', 'ρ(10Y)', 'v0',
]

# ---------------------------------------------------------------------------
# Bounds (same for v16 and v17)
# ---------------------------------------------------------------------------

_BOUNDS = dict(
    theta = (1e-5, 0.04),
    kf    = (1.5,  8.0),
    ks    = (0.05, 0.8),
    wf    = (0.05, 0.95),
    eps   = (0.05, 1.5),
    lam   = (0.01, 1.0),
    gam   = (0.01, 3.0),
    xi    = (1e-6, 0.01),
    rho   = (-0.95, 0.95),
    v0    = (1e-6, 0.01),
)


def make_param_space(device, version='v17'):
    """Return all calibration-space objects as a dict.

    Parameters
    ----------
    device  : torch.device
    version : str — 'v17' (17 params, shared theta) or
                    'v18' (23 params, independent theta per factor)

    Keys returned
    -------------
    lb, ub, param_range, param_bounds, param_names, n_params,
    theta_nodes, rho_nodes_t, pack, unpack, warm_start

    ``warm_start(checkpoint_paths)`` loads the first existing checkpoint
    (list of (Path, label) pairs) and maps it into the current space,
    returning a normalized p vector ∈ [0,1]^n ready for SPSA.
    """
    B = _BOUNDS

    # ── v18+ : two independent theta curves, theta_min = 1e-3 ───────────────
    if version in ('v18', 'v19', 'v20', 'v21', 'v22'):
        THETA_MIN, THETA_MAX = 1e-3, 0.04
        _names = [
            'θ1(3M)', 'θ1(6M)', 'θ1(1Y)', 'θ1(3Y)', 'θ1(5Y)', 'θ1(10Y)',
            'θ2(3M)', 'θ2(6M)', 'θ2(1Y)', 'θ2(3Y)', 'θ2(5Y)', 'θ2(10Y)',
            'κ_fast', 'κ_slow', 'w_fast', 'ε', 'λ', 'γ', 'ξ',
            'ρ(1Y)', 'ρ(5Y)', 'ρ(10Y)', 'v0',
        ]
        lb = torch.tensor(
            [THETA_MIN] * 12 + [B['kf'][0], B['ks'][0], B['wf'][0], B['eps'][0],
                                 B['lam'][0], B['gam'][0], B['xi'][0],
                                 B['rho'][0], B['rho'][0], B['rho'][0], B['v0'][0]],
            dtype=torch.float32, device=device)
        ub = torch.tensor(
            [THETA_MAX] * 12 + [B['kf'][1], B['ks'][1], B['wf'][1], B['eps'][1],
                                 B['lam'][1], B['gam'][1], B['xi'][1],
                                 B['rho'][1], B['rho'][1], B['rho'][1], B['v0'][1]],
            dtype=torch.float32, device=device)
        param_range  = ub - lb
        theta_nodes  = torch.tensor(THETA_NODES, dtype=torch.float32, device=device)
        rho_nodes_t  = torch.tensor(RHO_NODES_T, dtype=torch.float32, device=device)
        param_bounds = {n: (lb[i].item(), ub[i].item()) for i, n in enumerate(_names)}

        def pack(theta1, theta2, kf, ks, wf, eps, lam, gam, xi, rho1, rho5, rho10, v0):
            raw = torch.cat([
                theta1.to(device), theta2.to(device),
                torch.tensor([kf, ks, wf, eps, lam, gam, xi, rho1, rho5, rho10, v0],
                             dtype=torch.float32, device=device),
            ])
            return torch.clamp((raw - lb) / param_range, 0.0, 1.0)

        def unpack(p):
            raw = lb + p * param_range
            raw = torch.clamp(raw, lb, ub)
            return (raw[:6],    # theta1
                    raw[6:12],  # theta2
                    raw[12].item(), raw[13].item(), raw[14].item(),  # kf, ks, wf
                    raw[15].item(), raw[16].item(), raw[17].item(), raw[18].item(),  # eps, lam, gam, xi
                    raw[19].item(), raw[20].item(), raw[21].item(),  # rho1, rho5, rho10
                    raw[22].item())  # v0

        def warm_start(checkpoint_paths):
            """Load first existing checkpoint → normalized p vector for v18.

            Works at the raw-dict level — no per-param clipping boilerplate.
            Old v17-style checkpoints are handled: theta is duplicated into theta1/theta2.
            """
            for path, label in checkpoint_paths:
                if Path(path).exists():
                    with open(path, 'rb') as f:
                        ckpt = pickle.load(f)
                    th1 = ckpt.get('theta1_values', ckpt.get('theta_values')).to(device)
                    th2 = ckpt.get('theta2_values', th1.clone()).to(device)
                    scalars = torch.tensor([
                        ckpt.get('kappa_fast', 3.0),
                        ckpt.get('kappa_slow', 0.15),
                        ckpt.get('w_fast', 0.4),
                        ckpt.get('epsilon', 0.4),
                        ckpt.get('lam', 0.2),
                        ckpt.get('gamma', 0.1),
                        ckpt.get('xi', 0.005),
                        ckpt.get('rho_1y', -0.2),
                        ckpt.get('rho_5y', -0.3),
                        ckpt.get('rho_10y', -0.5),
                        ckpt.get('v0', float(th1[0])),
                    ], dtype=torch.float32, device=device)
                    raw = torch.cat([th1, th2, scalars])
                    p = torch.clamp((raw - lb) / param_range, 0.0, 1.0)
                    print(f"Warm start from {label}: θ1={th1.cpu().numpy().round(5)}")
                    return p
            # Fresh start: midpoint of bounds (model can explore freely)
            print("Fresh start: midpoint initialization")
            return 0.5 * torch.ones(len(lb), device=device)

        def smart_init(market_vols_t, T_fixes_t):
            """Initialize p from market implied-vol term structure (no checkpoint needed).

            For each theta node maturity the median market implied vol at the
            closest available expiry is used as a direct theta estimate — this is
            a rough but directionally correct proxy that puts the optimizer in a
            much better basin than the blind mid-point.

            Physical priors are used for all dynamic parameters (κ, ρ, etc.).

            Parameters
            ----------
            market_vols_t : Tensor  — market implied normal vols (all caplets)
            T_fixes_t     : Tensor  — corresponding maturities

            Returns
            -------
            p : Tensor [23] in [0, 1]  — normalized unit-cube init vector
            """
            T_np = T_fixes_t.cpu().numpy()
            V_np = market_vols_t.cpu().numpy()
            node_mats = [0.25, 0.5, 1.0, 3.0, 5.0, 10.0]

            theta_vals = []
            for T_node in node_mats:
                diffs = np.abs(T_np - T_node)
                closest_mat = T_np[diffs.argmin()]
                mask = np.abs(T_np - closest_mat) < 0.01
                atm_vol = float(np.median(V_np[mask]))
                theta_guess = float(np.clip(atm_vol, THETA_MIN, THETA_MAX))
                theta_vals.append(theta_guess)

            raw = torch.zeros(len(lb), dtype=torch.float32, device=device)
            # theta1 and theta2: initialise to ATM vol at each node maturity
            raw[:6]  = torch.tensor(theta_vals, dtype=torch.float32, device=device)
            raw[6:12] = torch.tensor(theta_vals, dtype=torch.float32, device=device)
            # Dynamic params: physically motivated priors
            raw[12] = 2.5    # kappa_fast  (half-life ~3M)
            raw[13] = 0.25   # kappa_slow  (half-life ~2.8Y)
            raw[14] = 0.5    # w_fast
            raw[15] = 0.4    # epsilon
            raw[16] = 0.2    # lambda
            raw[17] = 0.5    # gamma
            raw[18] = 0.003  # xi
            raw[19] = -0.3   # rho_1y
            raw[20] = -0.5   # rho_5y
            raw[21] = -0.5   # rho_10y
            raw[22] = float(theta_vals[0])  # v0 ≈ theta(3M)
            p = torch.clamp((raw - lb) / param_range, 0.0, 1.0)
            print(f"Smart init from market: theta_vals={[round(v,5) for v in theta_vals]}")
            return p

        return dict(lb=lb, ub=ub, param_range=param_range, param_bounds=param_bounds,
                    param_names=_names, n_params=len(_names),
                    theta_nodes=theta_nodes, rho_nodes_t=rho_nodes_t,
                    pack=pack, unpack=unpack, warm_start=warm_start,
                    smart_init=smart_init)

    # ── v16 / v17 : 17 params, shared theta ──────────────────────────────────

    lb = torch.tensor(
        [B['theta'][0]] * 6 + [B['kf'][0], B['ks'][0], B['wf'][0],
                                B['eps'][0], B['lam'][0], B['gam'][0], B['xi'][0],
                                B['rho'][0], B['rho'][0], B['rho'][0], B['v0'][0]],
        dtype=torch.float32, device=device)

    ub = torch.tensor(
        [B['theta'][1]] * 6 + [B['kf'][1], B['ks'][1], B['wf'][1],
                                B['eps'][1], B['lam'][1], B['gam'][1], B['xi'][1],
                                B['rho'][1], B['rho'][1], B['rho'][1], B['v0'][1]],
        dtype=torch.float32, device=device)

    param_range  = ub - lb
    param_bounds = {n: (lb[i].item(), ub[i].item())
                    for i, n in enumerate(PARAM_NAMES)}

    theta_nodes = torch.tensor(THETA_NODES, dtype=torch.float32, device=device)
    rho_nodes_t = torch.tensor(RHO_NODES_T, dtype=torch.float32, device=device)

    def pack(theta, kf, ks, wf, eps, lam, gam, xi, rho1, rho5, rho10, v0):
        raw = torch.cat([
            theta.to(device),
            torch.tensor([kf, ks, wf, eps, lam, gam, xi, rho1, rho5, rho10, v0],
                         dtype=torch.float32, device=device),
        ])
        return (raw - lb) / param_range

    def unpack(p):
        raw = torch.clamp(lb + p * param_range, lb, ub)
        return (raw[:6],           # theta  (6 nodes)
                raw[6].item(),     # kappa_fast
                raw[7].item(),     # kappa_slow
                raw[8].item(),     # w_fast
                raw[9].item(),     # epsilon
                raw[10].item(),    # lam
                raw[11].item(),    # gamma
                raw[12].item(),    # xi
                raw[13].item(),    # rho_1y
                raw[14].item(),    # rho_5y
                raw[15].item(),    # rho_10y
                raw[16].item())    # v0

    return dict(
        lb=lb, ub=ub,
        param_range=param_range,
        param_bounds=param_bounds,
        param_names=PARAM_NAMES,
        n_params=len(PARAM_NAMES),
        theta_nodes=theta_nodes,
        rho_nodes_t=rho_nodes_t,
        pack=pack,
        unpack=unpack,
        warm_start=None,  # v17 callers use calib_objective.warm_start
    )
