"""Bachelier caplet pricing and vol inversion utilities.

Pure pricing math (no MC, no data dependencies). Use for:
- Computing caplet prices from implied vol
- Inverting prices back to vols
- Validating pricing formulas
"""

import math

import numpy as np
import torch
from scipy.stats import norm
from scipy.optimize import brentq


def _bachelier_tv_undiscounted(F, K, sigma, g_hat):
    """Numerically stable time-value of Bachelier formula (undiscounted per unit T).
    
    Uses TV = v·[φ(d) − d·Φ(−d)] for ITM (d>0), which avoids catastrophic
    cancellation that occurs when subtracting intrinsic from total price.
    scipy's norm.sf computes Φ(−d) accurately even for d > 30.
    
    Returns (intrinsic, time_value) so caller can use whichever form is needed.
    """
    v = sigma * g_hat
    intrinsic = max(F - K, 0.0)
    if v < 1e-300:
        return intrinsic, 0.0
    d = (F - K) / v
    if d > 0:
        tv = v * (norm.pdf(d) - d * norm.sf(d))
    else:
        # OTM/ATM: full price IS the time value (intrinsic = 0)
        tv = (F - K) * norm.cdf(d) + v * norm.pdf(d)
    return intrinsic, max(tv, 0.0)


def bachelier_caplet_price(F, K, T, sigma, disc=1.0):
    """
    Bachelier (normal) price for NOW-STARTING average rate caplet.
    
    PV = T · disc · [(F-K)Φ(d) + σ·ĝ·φ(d)]
    where ĝ = √(T/3), d = (F-K)/(σ·ĝ)
    
    This is the average rate variant (g_hat = sqrt(T/3)) for caplets whose
    payoff is max(∫₀ᵀ a_t dt - T·K, 0), consistent with the MC simulation.
    
    Args:
        F: Forward rate = I(0,T)/T, average instantaneous forward (decimal)
        K: Strike (decimal)
        T: Time to maturity (years). For now-starting, tenor = T.
        sigma: Normal vol (decimal, e.g., 0.04 = 4%)
        disc: Discount factor to payment date
    
    Returns:
        Price as fraction of notional (e.g., 0.01 = 1% = 100bp)
    """
    if T <= 0:
        return max(F - K, 0) * T * disc
    
    g_hat = np.sqrt(T / 3.0)
    if sigma * g_hat < 1e-10:
        return max(F - K, 0) * T * disc
    
    d = (F - K) / (sigma * g_hat)
    undiscounted = (F - K) * norm.cdf(d) + sigma * g_hat * norm.pdf(d)
    
    return T * disc * undiscounted


def bachelier_caplet_time_value(F, K, T, sigma, disc=1.0):
    """Numerically stable time-value of Bachelier avg rate caplet.
    
    For deep ITM (d > 8), the standard price (F-K)Φ(d) + σĝφ(d) stores
    intrinsic + TV in one float64, losing TV when TV < ε·intrinsic.
    This function computes TV directly via v·[φ(d) − d·Φ(−d)], which
    scipy evaluates accurately even for d > 30.
    
    Returns:
        Time value only (excluding intrinsic), as fraction of notional.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    g_hat = np.sqrt(T / 3.0)
    _, tv = _bachelier_tv_undiscounted(F, K, sigma, g_hat)
    return T * disc * tv


def implied_vol_avg_rate(F, K, T, pv, disc=1.0, tol=1e-9, max_iter=200):
    """
    Invert Bachelier formula for now-starting average rate caplet: PV → σ_n.

    Solves: undiscounted_unit = (F-K)Φ(d) + σ·ĝ·φ(d) for σ
    where ĝ = √(T/3), d = (F-K)/(σ·ĝ)

    For deep ITM (F ≫ K), falls back to time-value formulation using
    Φ(−d) via scipy's norm.sf, which is accurate to full precision.

    Args:
        F: Forward rate = I(0,T)/T (decimal)
        K: Strike (decimal)
        T: Time to maturity (years)
        pv: Caplet price (fraction of notional)
        disc: Discount factor to payment date
        tol: Root-finding tolerance
        max_iter: Maximum Brentq iterations

    Returns:
        Implied normal vol (decimal), or np.nan if inversion fails.
    """
    if T <= 0:
        return np.nan
    if pv <= 0:
        pv = 0.0  # treat as effectively zero — fall through to intrinsic / TV channel

    g_hat = np.sqrt(T / 3.0)
    und_unit = pv / (T * disc)
    intrinsic = max(F - K, 0.0)

    # Standard inversion for OTM/ATM or mild ITM
    if und_unit > intrinsic + 1e-15:
        def price_error(sigma):
            d = (F - K) / (sigma * g_hat + 1e-15)
            return (F - K) * norm.cdf(d) + sigma * g_hat * norm.pdf(d) - und_unit
        try:
            return brentq(price_error, 1e-10, 10.0, xtol=tol, maxiter=max_iter)
        except ValueError:
            pass  # fall through to TV solver

    # Deep ITM fallback: solve from time-value using stable Φ(−d) formulation
    if intrinsic > 0:
        time_value = max(und_unit - intrinsic, 0.0)
        # Sub-intrinsic: model PV < intrinsic (F_model overestimates forward).
        # time_value = 0 → norm.sf underflows to 0 at huge d, brentq sees f(lo)=0
        # and returns the lower bound (~1e-10) as a spurious root instead of NaN.
        # Return NaN so the caller can apply a meaningful fallback.
        if time_value < 1e-15:
            return np.nan
        def tv_error(sigma):
            v = sigma * g_hat
            d = (F - K) / (v + 1e-300)
            return v * (norm.pdf(d) - d * norm.sf(d)) - time_value
        try:
            return brentq(tv_error, 1e-10, 10.0, xtol=tol, maxiter=max_iter)
        except ValueError:
            return np.nan

    return np.nan


def implied_vol_from_tv(F, K, T, time_value, disc=1.0, tol=1e-9, max_iter=200):
    """Invert Bachelier vol from separately-computed time value.
    
    Use with bachelier_caplet_time_value() for lossless deep-ITM round-trips.
    The TV channel preserves full precision because it never adds TV to intrinsic.
    
    Args:
        F, K, T: Forward, strike, maturity
        time_value: Time value only (from bachelier_caplet_time_value)
        disc: Discount factor
    
    Returns:
        Implied normal vol (decimal), or np.nan if inversion fails.
    """
    if T <= 0 or time_value <= 0:
        return np.nan
    g_hat = np.sqrt(T / 3.0)
    tv_und = time_value / (T * disc)

    def tv_error(sigma):
        v = sigma * g_hat
        d = (F - K) / (v + 1e-300)
        if d > 0:
            return v * (norm.pdf(d) - d * norm.sf(d)) - tv_und
        else:
            return (F - K) * norm.cdf(d) + v * norm.pdf(d) - tv_und
    try:
        return brentq(tv_error, 1e-10, 10.0, xtol=tol, maxiter=max_iter)
    except ValueError:
        return np.nan


# ── Vectorized GPU bisection ───────────────────────────────────────────────────

def implied_vol_batch(F_vec, K_vec, T_vec, pv_vec, disc_vec, n_iter=60):
    """Vectorized Bachelier implied vol via GPU bisection (no per-caplet Python loop).

    Mirrors implied_vol_avg_rate() exactly:
    - OTM / mild ITM  → standard Bachelier bisection
    - Deep ITM        → time-value channel bisection (stable for large d)
    - Truly undefined → returns 0.0

    Uses float64 internally to avoid float32 cancellation eating tiny time values
    on deep ITM caplets. Output is cast back to the input dtype.

    Args:
        F_vec    : (N,) tensor — average forward rates
        K_vec    : (N,) tensor — strikes
        T_vec    : (N,) tensor — maturities (years)
        pv_vec   : (N,) tensor — caplet PVs (fraction of notional)
        disc_vec : (N,) tensor — discount factors
        n_iter   : int — bisection iterations per channel

    Returns:
        (N,) tensor of implied normal vols; 0.0 only where PV < intrinsic.
    """
    orig_dtype = F_vec.dtype

    # Upcast to float64 so deep-ITM time values aren't eaten by float32 rounding
    F    = F_vec.double()
    K    = K_vec.double()
    T    = T_vec.double()
    pv   = pv_vec.double()
    disc = disc_vec.double()

    _INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)
    _INV_SQRT2   = 1.0 / math.sqrt(2.0)

    g_hat      = torch.sqrt(T / 3.0)
    und_target = pv / (T * disc)
    intrinsic  = torch.clamp(F - K, min=0.0)

    # Masks ─────────────────────────────────────────────────────────────────
    # std_mask : OTM/ATM/mild-ITM → standard Bachelier channel
    # tv_mask  : deep ITM (und ≈ intrinsic) → time-value channel
    std_mask = und_target > intrinsic + 1e-12
    tv_mask  = (~std_mask) & (intrinsic > 0.0)
    tv_target = (und_target - intrinsic).clamp(min=0.0)   # time value

    def bach_und(sigma):
        """Undiscounted Bachelier price per unit T."""
        v     = (sigma * g_hat).clamp(min=1e-300)
        d     = (F - K) / v
        phi_d = 0.5 * (1.0 + torch.erf(d * _INV_SQRT2))
        npdf  = _INV_SQRT2PI * torch.exp(-0.5 * d.pow(2))
        return (F - K) * phi_d + v * npdf

    def tv_und(sigma):
        """Time-value via v·[φ(d) − d·Φ(−d)] — numerically stable for large d."""
        v       = (sigma * g_hat).clamp(min=1e-300)
        d       = (F - K) / v
        phi_neg = 0.5 * (1.0 - torch.erf(d * _INV_SQRT2))   # Φ(−d)
        npdf    = _INV_SQRT2PI * torch.exp(-0.5 * d.pow(2))
        return v * (npdf - d * phi_neg)

    # Standard channel bisection ─────────────────────────────────────────────
    lo1 = torch.full_like(F, 1e-8)
    hi1 = torch.full_like(F, 2.0)
    for _ in range(n_iter):
        mid   = 0.5 * (lo1 + hi1)
        f_mid = bach_und(mid) - und_target
        hi1   = torch.where(f_mid > 0, mid, hi1)
        lo1   = torch.where(f_mid <= 0, mid, lo1)
    std_result = 0.5 * (lo1 + hi1)

    # Time-value channel bisection (deep ITM) ────────────────────────────────
    lo2 = torch.full_like(F, 1e-8)
    hi2 = torch.full_like(F, 2.0)
    for _ in range(n_iter):
        mid   = 0.5 * (lo2 + hi2)
        f_mid = tv_und(mid) - tv_target
        hi2   = torch.where(f_mid > 0, mid, hi2)
        lo2   = torch.where(f_mid <= 0, mid, lo2)
    tv_result = 0.5 * (lo2 + hi2)
    # Sub-intrinsic: tv_target = 0 → norm.sf underflows at huge d, making f_mid ≥ 0
    # throughout, so bisection converges to lo = 1e-8 (spurious root).
    # Set to exactly 0.0 so zero_mask in calib_objective catches it and applies
    # the vega-delta proxy instead of silently reporting zero vol.
    tv_result = torch.where(tv_target < 1e-15, torch.zeros_like(tv_result), tv_result)

    # Combine ─────────────────────────────────────────────────────────────────
    result = torch.where(std_mask, std_result,
             torch.where(tv_mask,  tv_result,
             torch.zeros_like(F)))
    return result.to(orig_dtype)
