"""
Caplet Volatility Surface Generation and Plotting Utilities

Functions for:
- Computing implied normal volatility surfaces from model prices
- Comparing model vs market data for interest rate caplets
- Arbitrage checking using Dupire local volatility conditions

Arbitrage Conditions (Bachelier/Normal Model):
- Calendar Spread:  dw/dT >= 0 where w = sigma² * T (total variance)
- Butterfly Spread: d²C/dK² >= 0 (convexity in strike price)
- Dupire Local Vol: sigma_loc² = (dC/dT) / (0.5 * d²C/dK²) must be real & positive
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator, RectBivariateSpline
from pyquant.torch_spline import PchipSpline1D

from affine_calibration.scripts.pricing_models import (
    _bachelier_tv_undiscounted, bachelier_caplet_price,
    bachelier_caplet_time_value, implied_vol_avg_rate, implied_vol_from_tv,
)
from affine_calibration.scripts.plotting import (
    plot_arbitrage_heatmaps, plot_caplet_vol_surface,
    plot_caplet_price_heatmaps, plot_spsa_convergence,
)


# =============================================================================
# BACHELIER PRICING
# =============================================================================

# =============================================================================
# MARKET PVs & VEGAS (for calibration)
# =============================================================================

def compute_market_pvs(T_fixes, strikes, market_vols, market_fwds, f_ois_vec, timeline):
    """
    Bachelier market prices for now-starting average rate caplets.

    PV = T * disc * ((F-K)Φ(d) + σ·ĝ·φ(d)),  ĝ = √(T/3), F = I(0,T)/T.
    """
    device = T_fixes.device
    dt = (timeline[1] - timeline[0]).item()
    n = len(T_fixes)
    pv = torch.zeros(n, dtype=torch.float32, device=device)

    for c in range(n):
        T = T_fixes[c].item()
        idx_T = min(int(T / dt), len(f_ois_vec) - 1)
        disc = torch.exp(-f_ois_vec[:idx_T + 1].sum() * dt)
        g_hat = np.sqrt(max(T / 3.0, 1e-8))
        d = ((market_fwds[c] - strikes[c]) / (market_vols[c] * g_hat + 1e-10)).item()
        undiscounted = (market_fwds[c] - strikes[c]) * norm.cdf(d) + market_vols[c] * g_hat * norm.pdf(d)
        pv[c] = T * disc * undiscounted
    return pv


def compute_market_vegas(T_fixes, strikes, market_vols, market_fwds, f_ois_vec, timeline,
                         vega_floor_pct=0.05):
    """
    Bachelier vegas for vega-weighted calibration loss.

    vega_i = T_i · disc_i · ĝ_i · φ(d_i).  Floored at ``vega_floor_pct``
    × median to prevent far-OTM blow-up.

    Returns (vegas_floored, n_floored).
    """
    device = T_fixes.device
    dt = (timeline[1] - timeline[0]).item()
    n = len(T_fixes)

    disc_vec = torch.zeros(n, device=device)
    for c in range(n):
        idx_T = min(int(T_fixes[c].item() / dt), len(f_ois_vec) - 1)
        disc_vec[c] = torch.exp(-f_ois_vec[:idx_T + 1].sum() * dt)

    g_hat = torch.sqrt(T_fixes / 3.0)
    d = (market_fwds - strikes) / (market_vols * g_hat + 1e-10)
    phi_d = (1.0 / np.sqrt(2 * np.pi)) * torch.exp(-0.5 * d ** 2)
    vegas = T_fixes * disc_vec * g_hat * phi_d

    floor = vegas.median() * vega_floor_pct
    n_floored = int((vegas < floor).sum().item())
    return torch.clamp(vegas, min=floor.item()), n_floored


# =============================================================================
# ARBITRAGE CHECKING - DUPIRE LOCAL VOL
# =============================================================================

def compute_total_variance_conditions(T, K, vol_func, h_T=0.1, use_richardson=True):
    """
    Check calendar arbitrage using total variance w = sigma^2 * T.
    
    Calendar arbitrage condition: dw/dT >= 0
    
    This is the cleaner formulation (per Gatheral/SVI methodology):
    - Total variance must be non-decreasing in T at each strike
    
    Uses Richardson extrapolation for accuracy.
    
    Args:
        T: Maturity (years)
        K: Strike (decimal)
        vol_func: Function vol_func(T, K) -> implied normal vol
        h_T: Step size for time derivative
        use_richardson: Whether to use Richardson extrapolation
    
    Returns:
        dict with:
            - w: Total variance sigma^2 * T
            - dw_dT: Time derivative of total variance
            - calendar_arb: True if dw/dT < 0 (arbitrage)
    """
    def total_var(t):
        if t <= 0:
            return 0.0
        sigma = vol_func(t, K)
        return sigma ** 2 * t
    
    w = total_var(T)
    
    # dw/dT using central difference
    if T - h_T > 0:
        dw_dT_h = (total_var(T + h_T) - total_var(T - h_T)) / (2 * h_T)
    else:
        dw_dT_h = (total_var(T + h_T) - total_var(T)) / h_T
    
    if use_richardson:
        h_T2 = h_T / 2
        if T - h_T2 > 0:
            dw_dT_h2 = (total_var(T + h_T2) - total_var(T - h_T2)) / (2 * h_T2)
        else:
            dw_dT_h2 = (total_var(T + h_T2) - total_var(T)) / h_T2
        dw_dT = (4 * dw_dT_h2 - dw_dT_h) / 3
    else:
        dw_dT = dw_dT_h
    
    return {
        'T': T,
        'K': K,
        'w': w,
        'dw_dT': dw_dT,
        'calendar_arb': dw_dT < -1e-10
    }


def compute_dupire_conditions(T, K, price_func, vol_func=None, h_T=0.1, h_K=0.005, use_richardson=True):
    """
    Compute Dupire arbitrage conditions using numerical derivatives.
    
    Two complementary checks:
    1. **Calendar (total variance)**: dw/dT >= 0 where w = sigma^2 * T
    2. **Butterfly (price convexity)**: d²C/dK² >= 0
    
    Uses Richardson extrapolation for higher accuracy:
    1. Compute derivative with step h
    2. Compute again with step h/2
    3. Combine: (4*f(h/2) - f(h)) / 3
    
    Args:
        T: Maturity (years)
        K: Strike (decimal)
        price_func: Function price_func(T, K) -> caplet price
        vol_func: Optional function vol_func(T, K) -> implied vol (for total variance check)
        h_T: Step size for time derivative
        h_K: Step size for strike derivative
        use_richardson: Whether to use Richardson extrapolation
    
    Returns:
        dict with:
            - dC_dT: Price time derivative (for reference)
            - d2C_dK2: Strike convexity (butterfly condition)
            - dw_dT: Total variance time derivative (calendar condition)
            - local_var: Dupire local variance
            - local_vol: Dupire local vol (sqrt of variance)
            - is_valid: True if no arbitrage
            - calendar_arb: True if calendar arbitrage (dw/dT < 0)
            - butterfly_arb: True if butterfly arbitrage (d²C/dK² < 0)
    """
    # ========== dC/dT (central difference) ==========
    if T - h_T > 0:
        dC_dT_h = (price_func(T + h_T, K) - price_func(T - h_T, K)) / (2 * h_T)
    else:
        dC_dT_h = (price_func(T + h_T, K) - price_func(T, K)) / h_T
    
    if use_richardson:
        h_T2 = h_T / 2
        if T - h_T2 > 0:
            dC_dT_h2 = (price_func(T + h_T2, K) - price_func(T - h_T2, K)) / (2 * h_T2)
        else:
            dC_dT_h2 = (price_func(T + h_T2, K) - price_func(T, K)) / h_T2
        dC_dT = (4 * dC_dT_h2 - dC_dT_h) / 3
    else:
        dC_dT = dC_dT_h
    
    # ========== Total Variance Calendar Check: dw/dT where w = sigma^2 * T ==========
    dw_dT = np.nan
    if vol_func is not None:
        tv_result = compute_total_variance_conditions(T, K, vol_func, h_T, use_richardson)
        dw_dT = tv_result['dw_dT']
        calendar_arb = tv_result['calendar_arb']
    else:
        # Fallback to price-based check if no vol_func provided
        calendar_arb = dC_dT < -1e-10
    
    # ========== d²C/dK² (second derivative) - Butterfly ==========
    d2C_dK2_h = (price_func(T, K + h_K) - 2*price_func(T, K) + price_func(T, K - h_K)) / (h_K ** 2)
    
    if use_richardson:
        h_K2 = h_K / 2
        d2C_dK2_h2 = (price_func(T, K + h_K2) - 2*price_func(T, K) + price_func(T, K - h_K2)) / (h_K2 ** 2)
        d2C_dK2 = (4 * d2C_dK2_h2 - d2C_dK2_h) / 3
    else:
        d2C_dK2 = d2C_dK2_h
    
    # ========== Dupire local variance ==========
    # Butterfly check: d²C/dK² >= 0
    butterfly_arb = d2C_dK2 < -1e-10
    
    # Compute local vol wherever d²C/dK² > 0
    # Use smaller threshold to capture more points
    if d2C_dK2 > 1e-14:
        local_var = dC_dT / (0.5 * d2C_dK2)
    else:
        local_var = np.nan  # Can't compute when butterfly is flat or violated
    
    # Arbitrage-free = both conditions satisfied (don't require local_var > 0)
    # Note: local_var < 0 indicates dC/dT < 0, which can happen for deep ITM near expiry
    # but isn't strictly arbitrage in the calendar spread sense
    is_valid = (not calendar_arb) and (not butterfly_arb)
    
    # Track why local vol might not be computable
    local_vol_computable = (d2C_dK2 > 1e-14) and (not np.isnan(local_var)) and (local_var > 0)
    
    return {
        'T': T,
        'K': K,
        'dC_dT': dC_dT,
        'dw_dT': dw_dT,  # Total variance derivative (calendar condition)
        'd2C_dK2': d2C_dK2,
        'local_var': local_var,
        'local_vol': np.sqrt(local_var) if local_var > 0 else np.nan,
        'is_valid': is_valid,
        'calendar_arb': calendar_arb,  # Based on dw/dT
        'butterfly_arb': butterfly_arb,
        'dC_dT_positive': dC_dT > -1e-10,  # Track if price derivative is positive
        'local_vol_computable': local_vol_computable
    }


def check_surface_arbitrage(vol_surface_df, fwd_func, disc_func, tau=0.25,
                            T_range=None, K_range=None, h_T=0.1, h_K=0.005):
    """
    Check entire vol surface for arbitrage violations.
    
    Args:
        vol_surface_df: DataFrame with columns [time_to_maturity, strike, implied_normal_vol]
        fwd_func: Function fwd_func(T) -> forward rate at maturity T
        disc_func: Function disc_func(T) -> discount factor to payment date T+tau
        tau: Accrual period
        T_range: (T_min, T_max) to check, or None for all
        K_range: (K_min, K_max) to check, or None for all
        h_T, h_K: Step sizes for derivatives
    
    Returns:
        arb_df: DataFrame with arbitrage check results for each point
        summary: dict with summary statistics
    """
    maturities = sorted(vol_surface_df['time_to_maturity'].unique())
    strikes = sorted(vol_surface_df['strike'].unique())
    
    # Build vol grid and interpolator
    vol_grid = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            mask = (vol_surface_df['time_to_maturity'] == T) & (vol_surface_df['strike'] == K)
            if mask.sum() > 0:
                vol_grid[i, j] = vol_surface_df.loc[mask, 'implied_normal_vol'].values[0]
            else:
                vol_grid[i, j] = np.nan
    
    # Fill NaN with interpolation
    vol_grid_df = pd.DataFrame(vol_grid, index=maturities, columns=strikes)
    vol_grid_filled = vol_grid_df.interpolate(axis=0).interpolate(axis=1).values
    
    # Create bivariate spline
    vol_spline = RectBivariateSpline(maturities, strikes, vol_grid_filled)
    
    def get_vol(T, K):
        T = np.clip(T, maturities[0], maturities[-1])
        K = np.clip(K, strikes[0], strikes[-1])
        return float(vol_spline(T, K)[0, 0])
    
    def price_at(T, K):
        if T <= 0:
            return 0.0
        vol = get_vol(T, K)
        F = fwd_func(min(T, maturities[-1]))
        disc = disc_func(min(T, maturities[-1]))
        return bachelier_caplet_price(F, K, T, vol, disc)
    
    # Filter ranges - include ALL points (derivatives handle boundaries)
    if T_range:
        check_T = [t for t in maturities if T_range[0] <= t <= T_range[1]]
    else:
        check_T = maturities  # Include all maturities
    
    if K_range:
        check_K = [k for k in strikes if K_range[0] <= k <= K_range[1]]
    else:
        check_K = strikes  # Include all strikes
    
    # Check all points - using both price_at and get_vol for total variance
    results = []
    for T in check_T:
        for K in check_K:
            result = compute_dupire_conditions(T, K, price_at, vol_func=get_vol, h_T=h_T, h_K=h_K)
            result['market_vol'] = get_vol(T, K)
            results.append(result)
    
    arb_df = pd.DataFrame(results)
    
    # Summary
    n_total = len(arb_df)
    n_valid = arb_df['is_valid'].sum()
    n_calendar = arb_df['calendar_arb'].sum()
    n_butterfly = arb_df['butterfly_arb'].sum()
    
    summary = {
        'total_points': n_total,
        'valid_points': n_valid,
        'valid_pct': n_valid / n_total * 100 if n_total > 0 else 0,
        'calendar_violations': n_calendar,
        'butterfly_violations': n_butterfly,
        'is_arbitrage_free': n_calendar + n_butterfly == 0
    }
    
    return arb_df, summary


def print_arbitrage_summary(arb_df, summary, title=""):
    """
    Print summary of arbitrage check results.
    
    Args:
        arb_df: DataFrame from check_surface_arbitrage()
        summary: dict from check_surface_arbitrage()
        title: Title for the summary
    """
    print(f"\n{'='*70}")
    print(f"{title} ARBITRAGE ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nARBITRAGE CONDITIONS:")
    print(f"  Points checked:      {summary['total_points']}")
    print(f"  Arbitrage-free:      {summary['valid_points']} ({summary['valid_pct']:.1f}%)")
    print(f"  Calendar violations: {summary['calendar_violations']} (dw/dT < 0)")
    print(f"  Butterfly violations:{summary['butterfly_violations']} (d²C/dK² < 0)")
    
    # LOCAL VOL COMPUTABILITY BREAKDOWN
    n_total = len(arb_df)
    n_arb_free = arb_df['is_valid'].sum()
    
    # Check if local_vol_computable column exists (new version)
    if 'local_vol_computable' in arb_df.columns:
        n_local_vol_ok = arb_df['local_vol_computable'].sum()
    else:
        n_local_vol_ok = (~arb_df['local_vol'].isna()).sum()
    
    n_d2C_small = (arb_df['d2C_dK2'] <= 1e-14).sum()
    n_dC_dT_neg = (arb_df['dC_dT'] < -1e-10).sum()
    n_local_var_neg = ((~arb_df['local_var'].isna()) & (arb_df['local_var'] <= 0)).sum()
    
    print(f"\nLOCAL VOL COMPUTABILITY:")
    print(f"  Computable (σ_loc² > 0): {n_local_vol_ok}/{n_total} ({100*n_local_vol_ok/n_total:.1f}%)")
    print(f"  Failures breakdown:")
    print(f"    d²C/dK² ≈ 0:           {n_d2C_small} (gamma vanishes at deep ITM/OTM)")
    print(f"    dC/dT < 0:             {n_dC_dT_neg} (price derivative negative)")
    print(f"    local_var ≤ 0:         {n_local_var_neg} (dC/dT and d²C/dK² opposite signs)")
    
    # Explain the discrepancy
    if n_arb_free > n_local_vol_ok:
        print(f"\n  NOTE: {n_arb_free - n_local_vol_ok} points are ARBITRAGE-FREE but local vol undefined.")
        print(f"  This is theoretically expected (Gatheral-Jacquier 2014):")
        print(f"  - Calendar arb uses total variance: dw/dT ≥ 0 where w = σ²T")  
        print(f"  - Local vol uses price derivative: σ²_loc = (dC/dT) / (0.5 d²C/dK²)")
        print(f"  - dw/dT ≥ 0 does NOT imply dC/dT > 0 (nonlinear relationship)")
    
    if summary['is_arbitrage_free']:
        print(f"\n  ✓ Surface is ARBITRAGE-FREE")
    else:
        print(f"\n  ⚠ Surface has ARBITRAGE VIOLATIONS")
        
        if summary['calendar_violations'] > 0:
            print(f"\n  Calendar arbitrage locations (dw/dT < 0, w = σ²T):")
            cal_arb = arb_df[arb_df['calendar_arb']][['T', 'K', 'dw_dT', 'dC_dT']].copy()
            cal_arb['K_%'] = cal_arb['K'] * 100
            cal_arb['dw_dT_fmt'] = cal_arb['dw_dT'].apply(lambda x: f'{x:.2e}')
            print(cal_arb[['T', 'K_%', 'dw_dT_fmt']].head(10).to_string(index=False))
        
        if summary['butterfly_violations'] > 0:
            print(f"\n  Butterfly arbitrage locations (d²C/dK² < 0):")
            but_arb = arb_df[arb_df['butterfly_arb']][['T', 'K', 'd2C_dK2']].copy()
            but_arb['K_%'] = but_arb['K'] * 100
            print(but_arb[['T', 'K_%', 'd2C_dK2']].head(10).to_string(index=False))
    
    # Local vol statistics for computable points
    valid_local = arb_df[~arb_df['local_vol'].isna() & (arb_df['local_vol'] > 0)]
    if len(valid_local) > 0:
        print(f"\n  DUPIRE LOCAL VOL (computable points):")
        print(f"    Min:  {valid_local['local_vol'].min()*100:.2f}%")
        print(f"    Max:  {valid_local['local_vol'].max()*100:.2f}%")
        print(f"    Mean: {valid_local['local_vol'].mean()*100:.2f}%")
        
        print(f"\n  MARKET IMPLIED VOL (for comparison):")
        print(f"    Min:  {valid_local['market_vol'].min()*100:.2f}%")
        print(f"    Max:  {valid_local['market_vol'].max()*100:.2f}%")
        print(f"    Mean: {valid_local['market_vol'].mean()*100:.2f}%")


def check_market_arbitrage(vol_key_rate, fwd_key_rate, fwd_ois, tau=0.25,
                           h_T=0.1, h_K=0.005, plot=True, verbose=True,
                           surface_name="Market"):
    """
    Check vol surface for arbitrage using Dupire local vol.
    
    This is a convenience wrapper that builds forward/discount curves internally.
    
    Args:
        vol_key_rate: DataFrame with [time_to_maturity, strike, implied_normal_vol]
        fwd_key_rate: DataFrame with [time_to_maturity, forward_rate] for key rate
        fwd_ois: DataFrame with [time_to_maturity, forward_rate] for OIS (discounting)
        tau: Accrual period (default 0.25 for quarterly)
        h_T: Step size for time derivative (Richardson extrapolation)
        h_K: Step size for strike derivative (Richardson extrapolation)
        plot: Whether to show heatmaps
        verbose: Whether to print detailed summary
        surface_name: Name for titles ("Market" or "Model")
    
    Returns:
        arb_df: DataFrame with arbitrage check at each point
        summary: dict with summary statistics
    """
    # Build forward curve (key rate)
    fwd_sorted = fwd_key_rate.sort_values('time_to_maturity')
    fwd_interp = PchipInterpolator(
        fwd_sorted['time_to_maturity'].values,
        fwd_sorted['forward_rate'].values
    )
    
    # Build OIS curve for discounting
    ois_sorted = fwd_ois.sort_values('time_to_maturity')
    ois_interp = PchipInterpolator(
        ois_sorted['time_to_maturity'].values,
        ois_sorted['forward_rate'].values
    )
    
    T_max = fwd_sorted['time_to_maturity'].max()
    
    def get_forward(T):
        """Get forward rate at maturity T."""
        return float(fwd_interp(min(T, T_max)))
    
    def get_discount(T):
        """Get discount factor to payment date T + tau."""
        T_pay = min(T + tau, T_max)
        # Simpson integration for average rate
        n_pts = max(10, int(T_pay * 100))
        t_grid = np.linspace(0, T_pay, n_pts)
        r_avg = np.mean(ois_interp(t_grid))
        return np.exp(-r_avg * T_pay)
    
    # Run arbitrage check
    arb_df, summary = check_surface_arbitrage(
        vol_key_rate, get_forward, get_discount, tau, h_T=h_T, h_K=h_K
    )
    
    if verbose:
        print_arbitrage_summary(arb_df, summary, title=f"{surface_name.upper()} SURFACE")
    
    if plot:
        plot_arbitrage_heatmaps(arb_df, title_prefix=surface_name)
    
    return arb_df, summary


# =============================================================================
# VOL SURFACE GENERATION AND INVERSION
# =============================================================================


def generate_caplet_vol_surface(vol_key_rate, fwd_key_rate, fwd_ois=None, version_name="Model",
                                 F_model=None, P_model=None):
    """
    Generate volatility surface from model prices for NOW-STARTING average rate caplets.
    
    The MC model PV = E[max(∫₀ᵀ a_t dt - T·K, 0) · exp(-∫₀ᵀ r_t dt)]
    
    Under Bachelier for average rates:
        PV = T · P(0,T) · [(F-K)Φ(d) + σ_n·ĝ·φ(d)]
    where:
        F = I_KEY(0,T)/T  (average instantaneous forward, NOT period forward)
        ĝ = √(T/3)        (now-starting average rate adjustment)
        d = (F-K)/(σ_n·ĝ)
    
    To invert: first compute undiscounted_unit = PV / (T · P(0,T)),
    then solve: undiscounted_unit = (F-K)Φ(d) + σ_n·ĝ·φ(d) for σ_n.
    
    T-forward measure mode (F_model, P_model provided):
        E^T[payoff] = E[disc·payoff] / E[disc],  F_model = E^T[ā]
        By Jensen: E^T[max(ā−K,0)] ≥ max(F_model−K,0), so Bachelier
        inversion is always well-defined (undiscounted ≥ intrinsic).
    
    Parameters:
    -----------
    vol_key_rate : pd.DataFrame
        DataFrame with columns: time_to_maturity, strike, implied_normal_vol, pv_model_key
    fwd_key_rate : pd.DataFrame
        Forward rate curve (period rates for ZCB computation)
    fwd_ois : pd.DataFrame, optional
        OIS forward curve for discounting. If None, uses fwd_key_rate.
    version_name : str
        Version identifier for labeling (e.g., "v1", "v2")
    F_model : array-like, optional
        Model forward per caplet under T-forward measure: E^T[ā].
        When provided, used instead of market forward for Bachelier inversion.
    P_model : array-like, optional
        Model ZCB price per caplet: E[exp(-∫r ds)].
        When provided, used instead of deterministic OIS discount.
    
    Returns:
    --------
    vol_results : pd.DataFrame
        DataFrame with market and model vols, errors, and diagnostics
    vol_rmse : float
        Volatility RMSE across all valid caplets
    """
    # Build forward rate interpolator for PERIOD rates (for ZCB computation)
    fwd_sorted = fwd_key_rate.sort_values('time_to_maturity')
    fwd_interp_period = PchipInterpolator(
        fwd_sorted['time_to_maturity'].values,
        fwd_sorted['forward_rate'].values
    )
    
    # Build OIS forward interpolator for discounting
    if fwd_ois is not None:
        ois_sorted = fwd_ois.sort_values('time_to_maturity')
        ois_interp = PchipInterpolator(
            ois_sorted['time_to_maturity'].values,
            ois_sorted['forward_rate'].values
        )
    else:
        ois_interp = fwd_interp_period
    
    def avg_inst_forward(T):
        """Average instantaneous forward = I(0,T)/T = -ln(P(0,T))/T."""
        F_period = float(fwd_interp_period(T))
        zcb = 1.0 / (1.0 + T * F_period)
        return -np.log(max(zcb, 1e-15)) / max(T, 1e-10)
    
    def ois_discount(T):
        """OIS discount factor P(0,T) = 1/(1 + T*F_OIS(T))."""
        F_ois = float(ois_interp(T))
        return 1.0 / (1.0 + T * F_ois)
    
    vol_results = vol_key_rate.copy()
    model_vols = []
    vol_errors = []
    failed_inversions = 0
    arbitrage_violations = 0
    failure_reasons = {'bounds': 0, 'convergence': 0, 'other': 0}
    
    for i, (idx, row) in enumerate(vol_results.iterrows()):
        T = row['time_to_maturity']
        K = row['strike']
        model_pv = row['pv_model_key']
        market_vol = row['implied_normal_vol']
        
        if T <= 0:
            failure_reasons['bounds'] += 1
            failed_inversions += 1
            model_vols.append(np.nan)
            vol_errors.append(np.nan)
            continue
        
        # Forward and discount: use model estimates if provided (T-forward measure)
        F = float(F_model[i]) if F_model is not None else avg_inst_forward(T)
        disc = float(P_model[i]) if P_model is not None else ois_discount(T)
        
        intrinsic_und = max(F - K, 0)
        undiscounted_unit = model_pv / (T * disc + 1e-15)
        
        # Use module-level implied_vol_avg_rate (handles deep ITM via stable TV channel)
        model_vol = implied_vol_avg_rate(F, K, T, model_pv, disc)
        
        if undiscounted_unit < intrinsic_und:
            arbitrage_violations += 1  # convexity effect, not true arb
        
        # Sub-intrinsic retry: F_model may overestimate the forward (model miscalibrated),
        # causing intrinsic > und_unit and a NaN return. Re-invert using the market
        # forward so the displayed vol is the effective vol the market would quote for
        # the model's PV — meaningful for diagnosing calibration quality.
        if (np.isnan(model_vol) or model_vol < 1e-5) and model_pv > 0:
            F_mkt = avg_inst_forward(T)
            disc_mkt = ois_discount(T)
            model_vol_retry = implied_vol_avg_rate(F_mkt, K, T, model_pv, disc_mkt)
            if not np.isnan(model_vol_retry) and model_vol_retry >= 1e-5:
                model_vol = model_vol_retry
            else:
                model_vol = np.nan  # let vega-delta fallback below handle it
        
        # No evaluation fallback. For deep ITM caplets where the model prices at
        # intrinsic (time value ≈ 0), Bachelier vol inversion is ill-conditioned:
        # any PV difference divided by near-zero vega gives an astronomical correction.
        # Show NaN honestly — the training loss (calib_objective.py) has its own
        # vega-delta proxy for gradient signal; the display should not fake values.
        
        if np.isnan(model_vol):
            failed_inversions += 1
            failure_reasons['convergence'] += 1
            model_vols.append(np.nan)
            vol_errors.append(np.nan)
        else:
            model_vols.append(float(model_vol))
            vol_errors.append(float(model_vol) - market_vol)
    
    model_vols_arr = np.array(model_vols, dtype=float)

    # Per-maturity strike interpolation: fill NaN vols by linear interpolation
    # from valid inversion neighbors within the same maturity slice.
    # Needed for deep-ITM caplets where time-value ≈ 0 makes Bachelier inversion
    # ill-conditioned — the model IS correctly pricing them at intrinsic, but we
    # can't invert a number that's at floating-point noise. Linear interp from
    # the adjacent valid strikes (which ARE invertible) gives a smooth consistent
    # display without any fake proxy logic.
    mats_arr    = vol_results['time_to_maturity'].values
    strikes_all = vol_results['strike'].values
    model_vols_final = model_vols_arr.copy()
    n_interpolated = 0

    for T_u in np.unique(mats_arr):
        mask  = mats_arr == T_u
        K_sl  = strikes_all[mask]
        v_sl  = model_vols_arr[mask].copy()
        valid = ~np.isnan(v_sl)

        if valid.all() or not valid.any() or valid.sum() < 2:
            continue  # nothing to fill, or not enough points to interpolate

        # np.interp: linear within valid strike range, holds boundary value outside
        v_sl[~valid] = np.interp(K_sl[~valid], K_sl[valid], v_sl[valid]).clip(min=1e-5)
        model_vols_final[mask] = v_sl
        n_interpolated += (~valid).sum()

    if n_interpolated > 0:
        print(f"Interpolated {n_interpolated} NaN vol(s) from neighboring strikes within same maturity.")

    vol_results[f'model_vol_{version_name}'] = model_vols_final
    vol_errors = model_vols_final - vol_results['implied_normal_vol'].values
    vol_results[f'vol_error_{version_name}'] = vol_errors

    valid_mask = ~np.isnan(vol_errors)
    vol_rmse = np.sqrt(np.mean(vol_errors[valid_mask]**2)) if valid_mask.any() else np.nan
    success_rate = valid_mask.sum() / len(vol_errors) * 100

    n_failed = (~valid_mask).sum()
    
    print(f"\n{'='*70}")
    print(f"{version_name.upper()} VOLATILITY SURFACE DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"Total caplets:        {len(vol_errors)}")
    print(f"Valid vol inversions: {valid_mask.sum()} ({success_rate:.1f}%)")
    print(f"Failed inversions:    {n_failed}")
    print(f"Sub-intrinsic (conv): {arbitrage_violations} (MC price < det. intrinsic -> NaN)")
    print(f"Vol RMSE:             {vol_rmse*100:.3f}%")
    if valid_mask.any():
        valid_vols = model_vols_final[valid_mask]
        print(f"Model vol range:      {valid_vols.min()*100:.2f}% - {valid_vols.max()*100:.2f}%")
        print(f"Market vol range:     {vol_results['implied_normal_vol'].min()*100:.2f}% - {vol_results['implied_normal_vol'].max()*100:.2f}%")
    print(f"{'='*70}\n")
    
    return vol_results, vol_rmse


# =============================================================================
# INTERACTIVE VISUALISATION — Plotly 3D + Signum 2D per-maturity smiles
# =============================================================================

def plot_vol_surface_interactive(vol_results, version_name="v15",
                                  itm_threshold=0.03, fwd_key_rate=None,
                                  open_browser=False):
    """
    Primary interactive vol surface plot: market (Blues) + model (Oranges)
    overlaid on the same 3D axes, plus a 2D signed-error heatmap below.

    Parameters
    ----------
    vol_results : pd.DataFrame
        Output of evaluate_vol_surface / generate_caplet_vol_surface.
        Required columns: time_to_maturity, strike, implied_normal_vol,
        model_vol_{version_name}, vol_error_{version_name}.
    version_name : str
        Tag used to locate model_vol_{version_name} and vol_error_{version_name}.
    itm_threshold : float
        Kept for API compatibility — deep ITM masking is no longer applied;
        both surfaces cover the full strike-maturity grid.
    fwd_key_rate : pd.DataFrame, optional
        Unused (kept for API compatibility).
    open_browser : bool
        False (default) — render inline in the notebook.
        True  — write a standalone HTML file and open it in the system browser,
                which gives fully unobstructed WebGL 3D rotation.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed — run:  pip install plotly")
        return

    from scipy.ndimage import distance_transform_edt

    model_col = f'model_vol_{version_name}'
    error_col  = f'vol_error_{version_name}'

    maturities = sorted(vol_results['time_to_maturity'].unique())
    strikes_u  = sorted(vol_results['strike'].unique())
    n_K, n_T   = len(strikes_u), len(maturities)

    mkt_grid = np.full((n_K, n_T), np.nan)
    mdl_grid = np.full((n_K, n_T), np.nan)
    err_grid = np.full((n_K, n_T), np.nan)

    for ki, k in enumerate(strikes_u):
        for ti, t in enumerate(maturities):
            row = vol_results[
                (vol_results['time_to_maturity'] == t) &
                (vol_results['strike'] == k)
            ]
            if len(row) == 0:
                continue
            mkt_v = row['implied_normal_vol'].values[0]
            mdl_v = row[model_col].values[0]
            err_v = row[error_col].values[0]

            mkt_grid[ki, ti] = mkt_v * 100
            if not np.isnan(mdl_v):
                mdl_grid[ki, ti] = mdl_v * 100
            if not np.isnan(err_v):
                err_grid[ki, ti] = err_v * 100

    # nearest-neighbour fill for model NaN holes (visual only)
    nan_mask = np.isnan(mdl_grid)
    if nan_mask.any() and (~nan_mask).any():
        _, idx = distance_transform_edt(nan_mask, return_indices=True)
        mdl_display = mdl_grid.copy()
        mdl_display[nan_mask] = mdl_grid[tuple(idx[:, nan_mask])]
    else:
        mdl_display = mdl_grid.copy()

    K_pct = np.array(strikes_u) * 100

    z_lo = float(np.nanmin([mkt_grid, mdl_display])) * 0.95
    z_hi = float(np.nanmax([mkt_grid, mdl_display])) * 1.05

    err_vals = err_grid[~np.isnan(err_grid)]
    err_abs  = float(np.quantile(np.abs(err_vals), 0.95)) if len(err_vals) else 1.0
    err_abs  = max(err_abs, 0.01)

    # ---- Figure 1: overlaid 3D surfaces ----
    common = dict(x=maturities, y=K_pct,
                  showscale=True,
                  contours=dict(z=dict(show=False)),
                  lighting=dict(ambient=0.75, diffuse=0.6))

    fig = go.Figure()
    fig.add_trace(go.Surface(
        z=mkt_grid,
        colorscale='Blues',
        cmin=z_lo, cmax=z_hi,
        opacity=0.85,
        name='Market',
        hovertemplate='T=%{x:.2f}Y<br>K=%{y:.2f}%<br>Market σ=%{z:.3f}%<extra>Market</extra>',
        colorbar=dict(x=0.85, thickness=14, title='σ (%)'),
        **common,
    ))
    fig.add_trace(go.Surface(
        z=mdl_display,
        colorscale='Oranges',
        cmin=z_lo, cmax=z_hi,
        opacity=0.70,
        name=version_name,
        hovertemplate=f'T=%{{x:.2f}}Y<br>K=%{{y:.2f}}%<br>{version_name} σ=%{{z:.3f}}%'
                      f'<extra>{version_name}</extra>',
        colorbar=dict(x=1.0, thickness=14, title='σ (%)'),
        **common,
    ))
    fig.update_layout(
        title=dict(text=f'Market (blue) vs {version_name.upper()} Model (orange) — Caplet Vol Surface',
                   font=dict(size=15)),
        scene=dict(
            xaxis_title='Maturity (Y)',
            yaxis_title='Strike (%)',
            zaxis_title='σ (%)',
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
            aspectmode='auto',
            dragmode='orbit',
        ),
        height=750,
        margin=dict(l=0, r=0, t=50, b=0),
        template='plotly_dark',
        legend=dict(x=0.02, y=0.98),
    )
    # ---- Figure 2: 2D error heatmap ----
    err_bps = err_grid * 100          # vol-% → bps
    fig2 = go.Figure(go.Heatmap(
        z=err_bps,
        x=maturities,
        y=K_pct,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-err_abs * 100,
        zmax=err_abs * 100,
        hovertemplate='T=%{x:.2f}Y<br>K=%{y:.2f}%<br>Δσ=%{z:+.1f}bp<extra></extra>',
        colorbar=dict(title='Δσ (bps)'),
    ))
    fig2.update_layout(
        title=f'Vol Error — {version_name.upper()} minus Market (bps)',
        xaxis_title='Maturity (Y)',
        yaxis_title='Strike (%)',
        height=420,
        template='plotly_dark',
    )

    cfg = dict(scrollZoom=False, displayModeBar=True)
    if open_browser:
        # Standalone HTML — full unobstructed WebGL rotation (no notebook iframe)
        import tempfile, webbrowser, pathlib
        html = fig.to_html(full_html=False, include_plotlyjs='cdn', config=cfg)
        html += fig2.to_html(full_html=False, include_plotlyjs=False, config=cfg)
        out = pathlib.Path(tempfile.gettempdir()) / f'vol_surface_{version_name}.html'
        out.write_text(f'<html><head><meta charset="utf-8"></head><body>{html}</body></html>',
                       encoding='utf-8')
        webbrowser.open(out.as_uri())
        print(f'Opened in browser: {out}')
    else:
        fig.show(config=cfg)
        fig2.show(config=cfg)


def plot_smiles_signum(vol_results, version_name="v15",
                       plot_maturities=None, fwd_key_rate=None,
                       theme='dark', height=380):
    """
    Per-maturity smile charts using Signum (TradingView Lightweight Charts backbone).

    For each maturity in `plot_maturities`, renders one Chart with:
      - Market smile  (solid line, Viridis colour)
      - Model smile   (dashed line, same colour, thinner)
      - Vol error     (baseline chart, green above / red below zero)

    Parameters
    ----------
    vol_results : pd.DataFrame
    version_name : str
    plot_maturities : list[float] | None  — defaults to all unique maturities
    fwd_key_rate : pd.DataFrame | None    — unused (kept for API consistency)
    theme : str                           — 'dark', 'ft', 'midnight', etc.
    height : int                          — chart height in pixels
    """
    try:
        from signum import Chart, Dashboard
    except ImportError:
        print("signum not installed — pip install git+https://github.com/SugoiKitsune/signum")
        return

    model_col = f'model_vol_{version_name}'
    error_col  = f'vol_error_{version_name}'

    all_mats = sorted(vol_results['time_to_maturity'].unique())
    if plot_maturities is None:
        plot_maturities = all_mats
    mats_to_show = [m for m in plot_maturities if m in all_mats]

    # colour palette — one per maturity
    import matplotlib as mpl
    cmap = mpl.colormaps['tab10']
    palette = ['#' + ''.join(f'{int(c*255):02x}' for c in cmap(i)[:3])
               for i in np.linspace(0, 0.9, len(mats_to_show))]

    panes = []
    titles = []

    for ci, mat in enumerate(mats_to_show):
        subset = (vol_results[vol_results['time_to_maturity'] == mat]
                  .sort_values('strike').copy())

        lbl = f'{mat:.0f}Y' if mat >= 1 else f'{int(mat*12)}M'
        colour = palette[ci % len(palette)]

        # Signum .line() expects DataFrame with 'time' and 'value' columns.
        # We repurpose 'time' as strike (float, monotonically increasing) — fine for LWC.
        strikes_pct = subset['strike'].values * 100

        mkt_df = pd.DataFrame({
            'time':  strikes_pct,
            'value': subset['implied_normal_vol'].values * 100,
        })
        valid_mdl = ~subset[model_col].isna()
        mdl_df = pd.DataFrame({
            'time':  strikes_pct[valid_mdl.values],
            'value': subset.loc[valid_mdl, model_col].values * 100,
        })

        # vol error for baseline chart (green / red centred at 0)
        valid_err = ~subset[error_col].isna()
        err_df = pd.DataFrame({
            'time':  strikes_pct[valid_err.values],
            'value': subset.loc[valid_err, error_col].values * 100,
        })

        smile_chart = (
            Chart(theme=theme, height=height, watermark=lbl)
            .line(mkt_df,  name=f'Market {lbl}',       color=colour,  width=2)
            .line(mdl_df,  name=f'{version_name} {lbl}', color=colour, width=1)
        )
        err_chart = (
            Chart(theme=theme, height=max(height // 3, 120))
            .baseline(err_df, base_value=0, name=f'Δσ {lbl}')
        )

        panes.extend([smile_chart, err_chart])
        titles.extend([f'Vol Smile — {lbl}', f'Error (model−mkt)'])

    dash = Dashboard(panes=panes, titles=titles)
    return dash


def print_model_vs_market_table(vol_key_rate, fwd_key_rate, model_pvs, market_pvs, 
                                 vol_results=None, version_name="Model", top_n=15):
    """
    Print comprehensive model vs market comparison table.
    
    Parameters:
    -----------
    vol_key_rate : pd.DataFrame
        DataFrame with time_to_maturity, strike, implied_normal_vol columns
    fwd_key_rate : pd.DataFrame
        Forward rate curve
    model_pvs : array-like
        Model caplet prices (decimal)
    market_pvs : array-like
        Market caplet prices (decimal)  
    vol_results : pd.DataFrame, optional
        Output from generate_caplet_vol_surface with model vols
    version_name : str
        Version identifier
    top_n : int
        Number of worst/best fits to show
    """
    import numpy as np
    
    # Convert to numpy
    model_pvs = np.array(model_pvs.cpu()) if hasattr(model_pvs, 'cpu') else np.array(model_pvs)
    market_pvs = np.array(market_pvs.cpu()) if hasattr(market_pvs, 'cpu') else np.array(market_pvs)
    
    # Build forward interpolator
    fwd_sorted = fwd_key_rate.sort_values('time_to_maturity')
    fwd_interp = PchipInterpolator(
        fwd_sorted['time_to_maturity'].values,
        fwd_sorted['forward_rate'].values
    )
    
    # Build full comparison DataFrame
    df = pd.DataFrame({
        'Maturity': vol_key_rate['time_to_maturity'].values,
        'Strike_%': vol_key_rate['strike'].values * 100,
        'Forward_%': [fwd_interp(t) * 100 for t in vol_key_rate['time_to_maturity'].values],
        'Mkt_Vol_%': vol_key_rate['implied_normal_vol'].values * 100,
        'Model_PV_bp': model_pvs * 10000,
        'Market_PV_bp': market_pvs * 10000,
        'PV_Diff_bp': (model_pvs - market_pvs) * 10000,
        'PV_Diff_%': (model_pvs - market_pvs) / (market_pvs + 1e-10) * 100
    })
    
    # Add model vol if available
    if vol_results is not None:
        model_vol_col = f'model_vol_{version_name}'
        if model_vol_col in vol_results.columns:
            df['Model_Vol_%'] = vol_results[model_vol_col].values * 100
            df['Vol_Diff_%'] = (vol_results[model_vol_col].values - vol_key_rate['implied_normal_vol'].values) * 100
    
    # Moneyness
    df['Moneyness_%'] = df['Forward_%'] - df['Strike_%']
    df['Abs_PV_Diff'] = np.abs(df['PV_Diff_bp'])
    
    # Print header
    print(f"\n{'='*100}")
    print(f"{version_name.upper()} MODEL VS MARKET COMPARISON")
    print(f"{'='*100}")
    
    # Overall statistics
    pv_rmse = np.sqrt(np.mean(df['PV_Diff_bp']**2))
    pv_mae = np.mean(df['Abs_PV_Diff'])
    pv_max_err = df['Abs_PV_Diff'].max()
    
    print(f"\nOVERALL PRICE FIT STATISTICS:")
    print(f"  Price RMSE:     {pv_rmse:.2f} bp")
    print(f"  Price MAE:      {pv_mae:.2f} bp")
    print(f"  Max |error|:    {pv_max_err:.2f} bp")
    print(f"  Model PV range: {df['Model_PV_bp'].min():.2f} - {df['Model_PV_bp'].max():.2f} bp")
    print(f"  Market PV range:{df['Market_PV_bp'].min():.2f} - {df['Market_PV_bp'].max():.2f} bp")
    
    if 'Model_Vol_%' in df.columns:
        valid_vol = ~df['Model_Vol_%'].isna()
        if valid_vol.any():
            vol_rmse = np.sqrt(np.mean(df.loc[valid_vol, 'Vol_Diff_%']**2))
            print(f"\nOVERALL VOL FIT STATISTICS:")
            print(f"  Vol RMSE:       {vol_rmse:.2f}%")
            print(f"  Model vol range:{df.loc[valid_vol, 'Model_Vol_%'].min():.2f}% - {df.loc[valid_vol, 'Model_Vol_%'].max():.2f}%")
            print(f"  Market vol range:{df['Mkt_Vol_%'].min():.2f}% - {df['Mkt_Vol_%'].max():.2f}%")
    
    # Summary by maturity
    print(f"\n{'='*100}")
    print(f"SUMMARY BY MATURITY")
    print(f"{'='*100}")
    
    mat_summary = df.groupby('Maturity').agg({
        'Forward_%': 'mean',
        'Model_PV_bp': 'mean',
        'Market_PV_bp': 'mean',
        'PV_Diff_bp': 'mean',
        'Abs_PV_Diff': 'mean'
    }).round(2)
    mat_summary.columns = ['Fwd_%', 'Model_bp', 'Market_bp', 'Diff_bp', '|Diff|_bp']
    
    if 'Model_Vol_%' in df.columns:
        vol_summary = df.groupby('Maturity').agg({
            'Mkt_Vol_%': 'mean',
            'Model_Vol_%': 'mean'
        }).round(2)
        mat_summary['Mkt_Vol_%'] = vol_summary['Mkt_Vol_%']
        mat_summary['Model_Vol_%'] = vol_summary['Model_Vol_%']
    
    print(mat_summary.to_string())
    
    # Top worst fits
    df_sorted = df.sort_values('Abs_PV_Diff', ascending=False)
    
    print(f"\n{'='*100}")
    print(f"TOP {top_n} WORST FITS (by |PV difference|)")
    print(f"{'='*100}")
    
    cols_to_show = ['Maturity', 'Strike_%', 'Forward_%', 'Moneyness_%', 
                    'Model_PV_bp', 'Market_PV_bp', 'PV_Diff_bp', 'PV_Diff_%']
    if 'Model_Vol_%' in df.columns:
        cols_to_show.extend(['Model_Vol_%', 'Mkt_Vol_%'])
    
    print(df_sorted[cols_to_show].head(top_n).to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    
    # Top best fits
    print(f"\n{'='*100}")
    print(f"TOP {top_n} BEST FITS (by |PV difference|)")
    print(f"{'='*100}")
    print(df_sorted[cols_to_show].tail(top_n).to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    
    # Analysis by moneyness
    print(f"\n{'='*100}")
    print(f"ANALYSIS BY MONEYNESS")
    print(f"{'='*100}")
    
    df['Moneyness_Bucket'] = pd.cut(df['Moneyness_%'], 
                                     bins=[-np.inf, -2, -0.5, 0.5, 2, np.inf],
                                     labels=['Deep OTM (<-2%)', 'OTM (-2% to -0.5%)', 
                                            'ATM (-0.5% to 0.5%)', 'ITM (0.5% to 2%)', 
                                            'Deep ITM (>2%)'])
    
    moneyness_summary = df.groupby('Moneyness_Bucket', observed=True).agg({
        'Model_PV_bp': ['count', 'mean'],
        'Market_PV_bp': 'mean',
        'Abs_PV_Diff': 'mean'
    }).round(2)
    moneyness_summary.columns = ['Count', 'Model_bp', 'Market_bp', '|Diff|_bp']
    print(moneyness_summary.to_string())
    
    return df


# =============================================================================
# PARAMETER SUMMARY & CONVERGENCE DIAGNOSTICS
# =============================================================================

def params_to_dataframe(best_params, theta_nodes=None):
    """Build a single summary DataFrame from best_params dict.
    
    Returns DataFrame with columns: Parameter, Value, Extra.
    Extra contains half-life for kappa, sqrt(theta)% for theta nodes, etc.
    """
    _v = lambda x: x.item() if hasattr(x, 'item') else float(x)
    rows = []
    theta = best_params['theta']
    # Support 6-node (legacy) and 8-node (current) theta interpolation
    _default_labels_6 = ['3M', '6M', '1Y', '3Y', '5Y', '10Y']
    _default_labels_8 = ['1M', '2M', '3M', '6M', '1Y', '3Y', '5Y', '10Y']
    if theta_nodes is not None:
        n = len(theta_nodes)
    else:
        n = len(theta)
    if n == 8:
        labels = _default_labels_8
        nodes = theta_nodes if theta_nodes is not None else [0.0833, 0.1667, 0.25, 0.5, 1.0, 3.0, 5.0, 10.0]
    else:
        labels = _default_labels_6
        nodes = theta_nodes if theta_nodes is not None else [0.25, 0.5, 1.0, 3.0, 5.0, 10.0]
    if 'v0' in best_params:
        v0_val = _v(best_params['v0'])
        rows.append(('v₀', f'{v0_val:.6f}', f'√v₀={np.sqrt(v0_val)*100:.2f}%'))
    for i, lbl in enumerate(labels):
        tv = _v(theta[i])
        rows.append((f'θ({lbl})', f'{tv:.6f}', f'√θ={np.sqrt(tv)*100:.2f}%'))
    kf = _v(best_params['kappa_fast'])
    ks = _v(best_params['kappa_slow'])
    rows.append(('κ_fast', f'{kf:.4f}', f'hl={np.log(2)/kf:.2f}Y'))
    rows.append(('κ_slow', f'{ks:.4f}', f'hl={np.log(2)/ks:.2f}Y'))
    rows.append(('w_fast', f'{_v(best_params["w_fast"]):.4f}', ''))
    eps = _v(best_params['epsilon'])
    rows.append(('ε', f'{eps:.4f}', f'ε²={eps**2:.5f}'))
    rows.append(('λ', f'{_v(best_params["lam"]):.4f}', ''))
    rows.append(('γ', f'{_v(best_params["gamma"]):.4f}', ''))
    rows.append(('ξ', f'{_v(best_params["xi"]):.6f}', ''))
    if 'rho_3m' in best_params:
        rows.append(('ρ(3M)', f'{_v(best_params.get("rho_3m", 0)):+.4f}', ''))
    rows.append(('ρ(1Y)', f'{_v(best_params.get("rho_1y", 0)):+.4f}', ''))
    rows.append(('ρ(5Y)', f'{_v(best_params.get("rho_5y", 0)):+.4f}', ''))
    rows.append(('ρ(10Y)', f'{_v(best_params.get("rho_10y", 0)):+.4f}', ''))
    return pd.DataFrame(rows, columns=['Parameter', 'Value', 'Extra'])
