"""Loss-function utilities for affine caplet calibration.

This module centralizes reusable loss definitions so notebooks and optimizers can
switch objective type without duplicating logic.

Supported objectives:
- PV/vega loss (stable calibration baseline)
- Vol-surface loss (fit implied vols directly)
- Hybrid loss (blend of PV and vol losses)
"""

from __future__ import annotations

import torch


EPS = 1e-12


def make_strike_weights(
    calib_mask: torch.Tensor,
    t_fixes: torch.Tensor,
    *,
    itm_weight: float = 0.15,
    atm_otm_weight: float = 1.0,
) -> torch.Tensor:
    """Build per-caplet strike weights, normalized within each maturity bucket.

    Args:
        calib_mask: Bool tensor where True marks ATM/OTM options used in core fit.
        t_fixes: Caplet maturities.
        itm_weight: Weight for deep-ITM points.
        atm_otm_weight: Weight for ATM/OTM points.
    """
    device = t_fixes.device
    weights = torch.where(
        calib_mask,
        torch.tensor(atm_otm_weight, device=device),
        torch.tensor(itm_weight, device=device),
    )

    for t_mat in torch.unique(t_fixes):
        bucket = t_fixes == t_mat
        w_sum = weights[bucket].sum()
        if w_sum > 0:
            # Keep average bucket scale at ~1.
            weights[bucket] = weights[bucket] / w_sum * bucket.sum().float()
    return weights


def _bucket_rmse(
    err: torch.Tensor,
    t_fixes: torch.Tensor,
    calib_mask: torch.Tensor,
    *,
    mat_weights: torch.Tensor | None = None,
    strike_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-maturity RMSE and aggregate to one scalar."""
    mats_unique = torch.unique(t_fixes)

    # If strike weights are provided, allow all points and let weights control impact.
    use_mask = calib_mask if strike_weights is None else torch.ones_like(calib_mask)

    bucket_rmses = []
    for t_mat in mats_unique:
        mask = use_mask & (t_fixes == t_mat)
        if mask.sum() == 0:
            continue

        err_b = err[mask]
        if strike_weights is None:
            rmse_b = torch.sqrt(torch.mean(err_b ** 2).clamp(min=EPS))
        else:
            sw = strike_weights[mask]
            sw = sw / (sw.sum() + EPS) * mask.sum().float()
            wmse = (sw * err_b ** 2).sum() / (sw.sum() + EPS)
            rmse_b = torch.sqrt(wmse.clamp(min=EPS))

        bucket_rmses.append(rmse_b)

    rmses = torch.stack(bucket_rmses)
    if mat_weights is None:
        return rmses.mean()
    return (rmses * mat_weights[: len(rmses)]).sum()


def smoothness_penalty(theta_nodes_values: torch.Tensor, weight: float = 0.1) -> torch.Tensor:
    """Quadratic roughness penalty for theta curve."""
    if theta_nodes_values.numel() < 2:
        return torch.zeros((), device=theta_nodes_values.device, dtype=theta_nodes_values.dtype)
    return weight * ((theta_nodes_values[1:] - theta_nodes_values[:-1]) ** 2).sum()


def loss_pv_vega(
    model_pvs: torch.Tensor,
    market_pvs: torch.Tensor,
    market_vegas: torch.Tensor,
    t_fixes: torch.Tensor,
    calib_mask: torch.Tensor,
    *,
    mat_weights: torch.Tensor | None = None,
    strike_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """PV objective on vega-normalized errors."""
    err = (model_pvs - market_pvs) / torch.clamp(market_vegas, min=EPS)
    return _bucket_rmse(
        err,
        t_fixes,
        calib_mask,
        mat_weights=mat_weights,
        strike_weights=strike_weights,
    )


def loss_vol_surface(
    model_vols: torch.Tensor,
    market_vols: torch.Tensor,
    t_fixes: torch.Tensor,
    calib_mask: torch.Tensor,
    *,
    mat_weights: torch.Tensor | None = None,
    strike_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Direct implied-vol surface loss."""
    err = model_vols - market_vols
    return _bucket_rmse(
        err,
        t_fixes,
        calib_mask,
        mat_weights=mat_weights,
        strike_weights=strike_weights,
    )


def loss_hybrid(
    model_pvs: torch.Tensor,
    market_pvs: torch.Tensor,
    market_vegas: torch.Tensor,
    model_vols: torch.Tensor,
    market_vols: torch.Tensor,
    t_fixes: torch.Tensor,
    calib_mask: torch.Tensor,
    *,
    w_pv: float = 0.5,
    w_vol: float = 0.5,
    mat_weights: torch.Tensor | None = None,
    strike_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Blend PV and vol losses.

    Notes:
        - Keep w_pv + w_vol ~= 1 for interpretability.
        - Start with w_pv > w_vol if direct-vol inversion is noisy.
    """
    pv_term = loss_pv_vega(
        model_pvs,
        market_pvs,
        market_vegas,
        t_fixes,
        calib_mask,
        mat_weights=mat_weights,
        strike_weights=strike_weights,
    )
    vol_term = loss_vol_surface(
        model_vols,
        market_vols,
        t_fixes,
        calib_mask,
        mat_weights=mat_weights,
        strike_weights=strike_weights,
    )
    return w_pv * pv_term + w_vol * vol_term


def total_loss(
    *,
    objective_mode: str,
    theta_values: torch.Tensor,
    smooth_pen_weight: float,
    model_pvs: torch.Tensor,
    market_pvs: torch.Tensor,
    market_vegas: torch.Tensor,
    t_fixes: torch.Tensor,
    calib_mask: torch.Tensor,
    model_vols: torch.Tensor | None = None,
    market_vols: torch.Tensor | None = None,
    w_pv: float = 0.5,
    w_vol: float = 0.5,
    mat_weights: torch.Tensor | None = None,
    strike_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unified entrypoint for objective selection.

    objective_mode:
        - "pv":     vega-normalized PV loss
        - "vol":    implied-vol RMSE loss
        - "hybrid": weighted blend of PV and vol losses
    """
    mode = objective_mode.lower().strip()

    if mode == "pv":
        core = loss_pv_vega(
            model_pvs,
            market_pvs,
            market_vegas,
            t_fixes,
            calib_mask,
            mat_weights=mat_weights,
            strike_weights=strike_weights,
        )
    elif mode == "vol":
        if model_vols is None or market_vols is None:
            raise ValueError("model_vols and market_vols are required for objective_mode='vol'.")
        core = loss_vol_surface(
            model_vols,
            market_vols,
            t_fixes,
            calib_mask,
            mat_weights=mat_weights,
            strike_weights=strike_weights,
        )
    elif mode == "hybrid":
        if model_vols is None or market_vols is None:
            raise ValueError("model_vols and market_vols are required for objective_mode='hybrid'.")
        core = loss_hybrid(
            model_pvs,
            market_pvs,
            market_vegas,
            model_vols,
            market_vols,
            t_fixes,
            calib_mask,
            w_pv=w_pv,
            w_vol=w_vol,
            mat_weights=mat_weights,
            strike_weights=strike_weights,
        )
    else:
        raise ValueError("objective_mode must be one of: 'pv', 'vol', 'hybrid'.")

    return core + smoothness_penalty(theta_values, smooth_pen_weight)
