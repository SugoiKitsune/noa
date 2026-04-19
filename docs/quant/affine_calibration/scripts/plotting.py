"""Caplet vol surface visualization and diagnostics (Matplotlib + Plotly).

Decoupled rendering logic — independent of pricing/simulation.
Use these after you've computed model and market vols.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import PchipInterpolator


def plot_arbitrage_heatmaps(arb_df, title_prefix=""):
    """
    Plot heatmaps showing arbitrage conditions across the surface.
    
    Args:
        arb_df: DataFrame from check_surface_arbitrage()
        title_prefix: Prefix for plot titles (e.g., "Market" or "Model")
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    T_grid = sorted(arb_df['T'].unique())
    K_grid = sorted(arb_df['K'].unique())
    n_total_pts = len(T_grid) * len(K_grid)
    
    # 1. dw/dT heatmap (calendar condition on total variance w = σ²T)
    dw_dT_matrix = arb_df.pivot(index='K', columns='T', values='dw_dT').values
    im1 = axes[0, 0].imshow(dw_dT_matrix, aspect='auto', origin='lower',
                             extent=[T_grid[0], T_grid[-1], K_grid[0]*100, K_grid[-1]*100],
                             cmap='RdYlGn')
    axes[0, 0].set_xlabel('Maturity (Y)')
    axes[0, 0].set_ylabel('Strike (%)')
    axes[0, 0].set_title(f'{title_prefix} dw/dT (w=σ²T)\nCalendar Arb (should ≥ 0)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. d²C/dK² heatmap (butterfly condition)
    d2C_dK2_matrix = arb_df.pivot(index='K', columns='T', values='d2C_dK2').values
    im2 = axes[0, 1].imshow(d2C_dK2_matrix, aspect='auto', origin='lower',
                             extent=[T_grid[0], T_grid[-1], K_grid[0]*100, K_grid[-1]*100],
                             cmap='RdYlGn')
    axes[0, 1].set_xlabel('Maturity (Y)')
    axes[0, 1].set_ylabel('Strike (%)')
    axes[0, 1].set_title(f'{title_prefix} d²C/dK²\nButterfly Arb (should ≥ 0)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. dC/dT heatmap (price time derivative - needed for local vol)
    dC_dT_matrix = arb_df.pivot(index='K', columns='T', values='dC_dT').values
    n_dC_dT_pos = np.sum(dC_dT_matrix > -1e-10)
    im3 = axes[0, 2].imshow(dC_dT_matrix, aspect='auto', origin='lower',
                             extent=[T_grid[0], T_grid[-1], K_grid[0]*100, K_grid[-1]*100],
                             cmap='RdYlGn')
    axes[0, 2].set_xlabel('Maturity (Y)')
    axes[0, 2].set_ylabel('Strike (%)')
    axes[0, 2].set_title(f'{title_prefix} dC/dT (price derivative)\nNeeded for local vol ≥ 0: {n_dC_dT_pos}/{n_total_pts}')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 4. Local vol surface - show all computable values
    local_var_matrix = arb_df.pivot(index='K', columns='T', values='local_var').values
    # For display: show local_vol (sqrt) where positive, indicate negative variance
    local_vol_matrix = np.where(local_var_matrix > 0, np.sqrt(local_var_matrix) * 100, np.nan)
    # Use percentile for vmax, ignoring NaN
    valid_vals = local_vol_matrix[~np.isnan(local_vol_matrix)]
    vmax = np.percentile(valid_vals, 95) if len(valid_vals) > 0 else 1.0
    im4 = axes[1, 0].imshow(local_vol_matrix, aspect='auto', origin='lower',
                             extent=[T_grid[0], T_grid[-1], K_grid[0]*100, K_grid[-1]*100],
                             cmap='viridis', vmin=0, vmax=vmax)
    axes[1, 0].set_xlabel('Maturity (Y)')
    axes[1, 0].set_ylabel('Strike (%)')
    n_local_vol_valid = np.sum(~np.isnan(local_vol_matrix))
    axes[1, 0].set_title(f'{title_prefix} Dupire Local Vol (%)\nσ²_loc = dC/dT / (½d²C/dK²): {n_local_vol_valid}/{n_total_pts}')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # 5. Arbitrage violation map (based on dw/dT >= 0 AND d²C/dK² >= 0)
    valid_matrix = arb_df.pivot(index='K', columns='T', values='is_valid').values.astype(float)
    n_valid = int(np.sum(valid_matrix))
    im5 = axes[1, 1].imshow(valid_matrix, aspect='auto', origin='lower',
                             extent=[T_grid[0], T_grid[-1], K_grid[0]*100, K_grid[-1]*100],
                             cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 1].set_xlabel('Maturity (Y)')
    axes[1, 1].set_ylabel('Strike (%)')
    axes[1, 1].set_title(f'{title_prefix} Arbitrage-Free\n(dw/dT≥0 ∧ d²C/dK²≥0): {n_valid}/{n_total_pts} ({100*n_valid/n_total_pts:.1f}%)')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # 6. Local vol computable map (d²C/dK² > 0 AND dC/dT > 0)
    local_vol_ok = (~np.isnan(local_vol_matrix)).astype(float)
    n_ok = int(np.sum(local_vol_ok))
    im6 = axes[1, 2].imshow(local_vol_ok, aspect='auto', origin='lower',
                             extent=[T_grid[0], T_grid[-1], K_grid[0]*100, K_grid[-1]*100],
                             cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 2].set_xlabel('Maturity (Y)')
    axes[1, 2].set_ylabel('Strike (%)')
    axes[1, 2].set_title(f'{title_prefix} Local Vol Computable\n(d²C/dK²>0 ∧ dC/dT>0): {n_ok}/{n_total_pts} ({100*n_ok/n_total_pts:.1f}%)')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.show()


def plot_caplet_vol_surface(vol_results, version_name="Model", plot_maturities=[1.0, 3.0, 5.0, 7.0, 10.0],
                            fwd_key_rate=None):
    """
    Plot 3D volatility surface comparison: market vs model.
    
    Parameters:
    -----------
    vol_results : pd.DataFrame
        Output from generate_caplet_vol_surface() with market and model vols
    version_name : str
        Version identifier for plot titles
    plot_maturities : list
        Maturities to highlight in 2D slice plots
    fwd_key_rate : pd.DataFrame, optional
        Forward rate curve — if provided, forward line is drawn on smile plots
    """
    model_vol_col = f'model_vol_{version_name}'
    error_col = f'vol_error_{version_name}'
    
    maturities = sorted(vol_results['time_to_maturity'].unique())
    strikes = sorted(vol_results['strike'].unique())
    T_grid, K_grid = np.meshgrid(maturities, strikes)
    
    market_vol_grid = np.zeros_like(T_grid)
    model_vol_grid = np.full_like(T_grid, np.nan)
    error_grid = np.full_like(T_grid, np.nan)
    
    for i, k in enumerate(strikes):
        for j, t in enumerate(maturities):
            row = vol_results[(vol_results['time_to_maturity'] == t) & (vol_results['strike'] == k)]
            if len(row) > 0:
                market_vol_grid[i, j] = row['implied_normal_vol'].values[0] * 100
                model_val = row[model_vol_col].values[0]
                if not np.isnan(model_val):
                    model_vol_grid[i, j] = model_val * 100
                    error_grid[i, j] = (model_val - row['implied_normal_vol'].values[0]) * 100
    
    # Clamp model vols to a sane range around market to kill triangle artifacts
    mkt_lo = np.nanmin(market_vol_grid) * 0.2
    mkt_hi = np.nanmax(market_vol_grid) * 3.0
    # Fill NaN with nearest-neighbor interpolation from valid model vols
    nan_mask = np.isnan(model_vol_grid)
    if nan_mask.any() and (~nan_mask).any():
        _, nearest_idx = distance_transform_edt(nan_mask, return_distances=True, return_indices=True)
        model_vol_display = model_vol_grid.copy()
        model_vol_display[nan_mask] = model_vol_grid[tuple(nearest_idx[:, nan_mask])]
    else:
        model_vol_display = model_vol_grid.copy()
    model_vol_display = np.clip(model_vol_display, mkt_lo, mkt_hi)
    error_display = np.where(np.isnan(error_grid), 0, error_grid)
    err_lim = max(abs(np.nanpercentile(error_grid[~np.isnan(error_grid)], 5)),
                  abs(np.nanpercentile(error_grid[~np.isnan(error_grid)], 95))) if (~np.isnan(error_grid)).any() else 1.0
    error_display = np.clip(error_display, -err_lim * 1.5, err_lim * 1.5)
    
    # Build forward curve for reference
    fwd_interp = None
    if fwd_key_rate is not None:
        fs = fwd_key_rate.sort_values('time_to_maturity')
        fwd_interp = PchipInterpolator(fs['time_to_maturity'].values, fs['forward_rate'].values)
    
    total_points = T_grid.size
    valid_model_points = np.sum(~np.isnan(model_vol_grid))
    coverage_pct = valid_model_points / total_points * 100
    
    print(f"\nSurface Coverage: {valid_model_points}/{total_points} points ({coverage_pct:.1f}%)")
    if total_points > valid_model_points:
        print(f"Missing {total_points - valid_model_points} points due to inversion failures")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Shared z-limits for market & model so visual scale matches
    z_lo = min(np.nanmin(market_vol_grid), np.nanmin(model_vol_display)) * 0.9
    z_hi = max(np.nanmax(market_vol_grid), np.nanmax(model_vol_display)) * 1.1
    
    # Row 1: 3D Surfaces
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(T_grid, K_grid * 100, market_vol_grid,
                             cmap='viridis', alpha=0.9, edgecolor='none')
    ax1.set_xlabel('Maturity (y)', fontsize=10, labelpad=5)
    ax1.set_ylabel('Strike (%)', fontsize=10, labelpad=5)
    ax1.set_zlabel('Vol (%)', fontsize=10, labelpad=5)
    ax1.set_zlim(z_lo, z_hi)
    ax1.set_title('Market Volatility Surface', fontsize=12, fontweight='bold')
    ax1.view_init(elev=25, azim=135)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(T_grid, K_grid * 100, model_vol_display,
                             cmap='viridis', alpha=0.9, edgecolor='none')
    ax2.set_xlabel('Maturity (y)', fontsize=10, labelpad=5)
    ax2.set_ylabel('Strike (%)', fontsize=10, labelpad=5)
    ax2.set_zlabel('Vol (%)', fontsize=10, labelpad=5)
    ax2.set_zlim(z_lo, z_hi)
    ax2.set_title(f'{version_name.upper()} Model Volatility Surface', fontsize=12, fontweight='bold')
    ax2.view_init(elev=25, azim=135)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(T_grid, K_grid * 100, error_display,
                             cmap='RdBu_r', alpha=0.9, edgecolor='none')
    ax3.set_xlabel('Maturity (y)', fontsize=10, labelpad=5)
    ax3.set_ylabel('Strike (%)', fontsize=10, labelpad=5)
    ax3.set_zlabel('Error (%)', fontsize=10, labelpad=5)
    ax3.set_title(f'{version_name.upper()} - Market (Vol Error)', fontsize=12, fontweight='bold')
    ax3.view_init(elev=25, azim=135)
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
    
    # Row 2: 2D Maturity Slices
    ax4 = fig.add_subplot(2, 3, 4)
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_maturities)))
    for ci, mat in enumerate(plot_maturities):
        if mat in maturities:
            subset = vol_results[vol_results['time_to_maturity'] == mat].sort_values('strike')
            lbl = f'{mat:.0f}Y' if mat >= 1 else f'{mat*12:.0f}M'
            ax4.plot(subset['strike'] * 100, subset['implied_normal_vol'] * 100,
                    'o-', color=colors[ci], ms=3, lw=2, label=f'{lbl} Market')
    ax4.set_xlabel('Strike (%)', fontsize=11)
    ax4.set_ylabel('Implied Vol (%)', fontsize=11)
    ax4.set_title('Market Vol - Maturity Slices', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, ncol=2)
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(2, 3, 5)
    for ci, mat in enumerate(plot_maturities):
        if mat in maturities:
            subset = vol_results[vol_results['time_to_maturity'] == mat].sort_values('strike')
            lbl = f'{mat:.0f}Y' if mat >= 1 else f'{mat*12:.0f}M'
            # Market (faint)
            ax5.plot(subset['strike'] * 100, subset['implied_normal_vol'] * 100,
                    '-', color=colors[ci], alpha=0.3, lw=1)
            # Model
            valid = ~subset[model_vol_col].isna()
            ax5.plot(subset.loc[valid, 'strike'] * 100, subset.loc[valid, model_vol_col] * 100,
                    's-', color=colors[ci], ms=3, lw=2, label=f'{lbl}')
            # Forward line
            if fwd_interp is not None:
                fwd = float(fwd_interp(mat)) * 100
                ax5.axvline(fwd, color=colors[ci], ls=':', alpha=0.3, lw=0.8)
    ax5.set_xlabel('Strike (%)', fontsize=11)
    ax5.set_ylabel('Implied Vol (%)', fontsize=11)
    ax5.set_title(f'{version_name.upper()} Model vs Market (faint)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9, ncol=2)
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(2, 3, 6)
    for ci, mat in enumerate(plot_maturities):
        if mat in maturities:
            subset = vol_results[vol_results['time_to_maturity'] == mat].sort_values('strike')
            lbl = f'{mat:.0f}Y' if mat >= 1 else f'{mat*12:.0f}M'
            valid = ~subset[error_col].isna()
            ax6.plot(subset.loc[valid, 'strike'] * 100, subset.loc[valid, error_col] * 100,
                    '^-', color=colors[ci], ms=3, lw=1.5, label=f'{lbl} Error')
    ax6.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax6.set_xlabel('Strike (%)', fontsize=11)
    ax6.set_ylabel('Vol Error (%)', fontsize=11)
    ax6.set_title(f'{version_name.upper()} - Market Error', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9, ncol=2)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_caplet_price_heatmaps(vol_key_rate, model_pvs, market_pvs, version_name="Model"):
    """
    Plot heatmaps showing caplet price errors and model PVs.
    
    Parameters:
    -----------
    vol_key_rate : pd.DataFrame
        DataFrame with time_to_maturity and strike columns
    model_pvs : array-like
        Model caplet prices (in decimal, will be converted to bp)
    market_pvs : array-like
        Market caplet prices (in decimal, will be converted to bp)
    version_name : str
        Version identifier for plot titles
    """
    # Convert to numpy arrays
    model_pvs = np.array(model_pvs) if not hasattr(model_pvs, 'cpu') else model_pvs.cpu().numpy()
    market_pvs = np.array(market_pvs) if not hasattr(market_pvs, 'cpu') else market_pvs.cpu().numpy()
    
    # Build DataFrame
    caplet_grid = pd.DataFrame({
        'Maturity': vol_key_rate['time_to_maturity'].values,
        'Strike': vol_key_rate['strike'].values * 100,  # Convert to %
        'Model_PV': model_pvs * 10000,  # Convert to bp
        'Market_PV': market_pvs * 10000,
        'Diff_bp': (model_pvs - market_pvs) * 10000,
        'Diff_pct': (model_pvs - market_pvs) / (market_pvs + 1e-10) * 100
    })
    
    # Pivot for heatmaps
    pivot_diff_pct = caplet_grid.pivot_table(values='Diff_pct', index='Strike', columns='Maturity', aggfunc='mean')
    pivot_diff_bp = caplet_grid.pivot_table(values='Diff_bp', index='Strike', columns='Maturity', aggfunc='mean')
    pivot_model = caplet_grid.pivot_table(values='Model_PV', index='Strike', columns='Maturity', aggfunc='mean')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Percentage error
    im1 = axes[0].imshow(pivot_diff_pct.values, aspect='auto', cmap='RdBu_r', vmin=-100, vmax=100)
    axes[0].set_xticks(range(len(pivot_diff_pct.columns)))
    axes[0].set_xticklabels([f'{x:.1f}Y' for x in pivot_diff_pct.columns], rotation=45)
    axes[0].set_yticks(range(len(pivot_diff_pct.index)))
    axes[0].set_yticklabels([f'{x:.1f}%' for x in pivot_diff_pct.index])
    axes[0].set_xlabel('Maturity')
    axes[0].set_ylabel('Strike')
    axes[0].set_title(f'{version_name} Price Error (Model-Market)/Market %')
    plt.colorbar(im1, ax=axes[0], label='Error %')
    
    # Plot 2: Absolute error in bp
    bp_max = max(abs(np.nanmin(pivot_diff_bp.values)), abs(np.nanmax(pivot_diff_bp.values)))
    im2 = axes[1].imshow(pivot_diff_bp.values, aspect='auto', cmap='RdBu_r', vmin=-bp_max, vmax=bp_max)
    axes[1].set_xticks(range(len(pivot_diff_bp.columns)))
    axes[1].set_xticklabels([f'{x:.1f}Y' for x in pivot_diff_bp.columns], rotation=45)
    axes[1].set_yticks(range(len(pivot_diff_bp.index)))
    axes[1].set_yticklabels([f'{x:.1f}%' for x in pivot_diff_bp.index])
    axes[1].set_xlabel('Maturity')
    axes[1].set_ylabel('Strike')
    axes[1].set_title(f'{version_name} Price Error (Model-Market) bp')
    plt.colorbar(im2, ax=axes[1], label='Error (bp)')
    
    # Plot 3: Model PV
    im3 = axes[2].imshow(pivot_model.values, aspect='auto', cmap='viridis')
    axes[2].set_xticks(range(len(pivot_model.columns)))
    axes[2].set_xticklabels([f'{x:.1f}Y' for x in pivot_model.columns], rotation=45)
    axes[2].set_yticks(range(len(pivot_model.index)))
    axes[2].set_yticklabels([f'{x:.1f}%' for x in pivot_model.index])
    axes[2].set_xlabel('Maturity')
    axes[2].set_ylabel('Strike')
    axes[2].set_title(f'{version_name} Model PV (bp)')
    plt.colorbar(im3, ax=axes[2], label='PV (bp)')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    rmse_bp = np.sqrt(np.mean(caplet_grid['Diff_bp']**2))
    mae_bp = np.mean(np.abs(caplet_grid['Diff_bp']))
    print(f"\n{version_name} Price Fit Summary:")
    print(f"  RMSE: {rmse_bp:.2f} bp")
    print(f"  MAE:  {mae_bp:.2f} bp")
    print(f"  Model range: {caplet_grid['Model_PV'].min():.2f} - {caplet_grid['Model_PV'].max():.2f} bp")
    print(f"  Market range: {caplet_grid['Market_PV'].min():.2f} - {caplet_grid['Market_PV'].max():.2f} bp")


def plot_spsa_convergence(history, model_pvs=None, market_pvs=None):
    """Plot 2×4 SPSA convergence diagnostics from training history."""
    hist_df = pd.DataFrame(history)
    theta_names    = ['θ(3M)', 'θ(6M)', 'θ(1Y)', 'θ(3Y)', 'θ(5Y)', 'θ(10Y)']
    scalar_names   = ['kappa_fast', 'kappa_slow', 'w_fast', 'epsilon', 'lam', 'gamma', 'xi']
    scalar_display = ['κ_fast', 'κ_slow', 'w', 'ε', 'λ', 'γ', 'ξ']
    rho_names      = ['rho_1y', 'rho_5y', 'rho_10y']
    rho_display    = ['ρ(1Y)', 'ρ(5Y)', 'ρ(10Y)']
    grad_scalar    = ['grad_kf', 'grad_ks', 'grad_wf', 'grad_eps', 'grad_lam', 'grad_gamma', 'grad_xi']
    grad_rho       = ['grad_rho1', 'grad_rho5', 'grad_rho10']

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    # 1) Loss
    axes[0, 0].plot(hist_df['iter'], hist_df['loss'], 'b-', lw=2)
    axes[0, 0].set(xlabel='Iteration', ylabel='Loss', title='Loss Convergence')
    axes[0, 0].set_yscale('log'); axes[0, 0].grid(alpha=0.3)

    # 2) θ nodes as √θ %
    for i, name in enumerate(theta_names):
        vals = [np.sqrt(h['theta'][i]) * 100 for h in history]
        axes[0, 1].plot(hist_df['iter'], vals, '-', lw=2, label=name)
    if 'v0' in history[0]:
        v0_vals = [np.sqrt(h['v0']) * 100 for h in history]
        axes[0, 1].plot(hist_df['iter'], v0_vals, 'k--', lw=2, label='√v₀')
    axes[0, 1].set(xlabel='Iteration', ylabel='√θ (%)', title='θ(t) Nodes')
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3)

    # 3) 2-factor CIR: κ_fast, κ_slow, w_fast
    ax3 = axes[0, 2]; ax3r = ax3.twinx()
    ax3.plot(hist_df['iter'], hist_df['kappa_fast'], 'r-', lw=2, label='κ_fast')
    ax3.plot(hist_df['iter'], hist_df['kappa_slow'], 'b-', lw=2, label='κ_slow')
    ax3r.plot(hist_df['iter'], hist_df['w_fast'], 'g--', lw=2, label='w_fast')
    ax3.set(xlabel='Iteration', ylabel='κ', title='2-Factor CIR')
    ax3r.set_ylabel('w_fast', color='g'); ax3r.tick_params(axis='y', labelcolor='g')
    ax3.legend(loc='upper left', fontsize=8); ax3r.legend(loc='upper right', fontsize=8)
    ax3.grid(alpha=0.3)

    # 4) ρ(t) nodes
    for name, disp in zip(rho_names, rho_display):
        axes[0, 3].plot(hist_df['iter'], hist_df[name], '-', lw=2, label=disp)
    axes[0, 3].set(xlabel='Iteration', ylabel='ρ', title='ρ(t) Nodes')
    axes[0, 3].legend(fontsize=8); axes[0, 3].grid(alpha=0.3)
    axes[0, 3].axhline(0, color='grey', lw=0.5)

    # 5) θ gradient magnitudes
    for i, name in enumerate(theta_names):
        vals = [abs(h['grad_theta'][i]) for h in history]
        axes[1, 0].plot(hist_df['iter'], vals, '-', lw=2, label=name)
    axes[1, 0].set(xlabel='Iteration', ylabel='|grad|', title='θ Gradients')
    axes[1, 0].legend(fontsize=8); axes[1, 0].set_yscale('log'); axes[1, 0].grid(alpha=0.3)

    # 6) Scalar gradient magnitudes
    for gn, disp in zip(grad_scalar, scalar_display):
        axes[1, 1].plot(hist_df['iter'], hist_df[gn].abs(), '-', lw=2, label=disp)
    axes[1, 1].set(xlabel='Iteration', ylabel='|grad|', title='Scalar Gradients')
    axes[1, 1].legend(fontsize=8); axes[1, 1].set_yscale('log'); axes[1, 1].grid(alpha=0.3)

    # 7) ρ gradient magnitudes
    for gn, disp in zip(grad_rho, rho_display):
        axes[1, 2].plot(hist_df['iter'], hist_df[gn].abs(), '-', lw=2, label=disp)
    axes[1, 2].set(xlabel='Iteration', ylabel='|grad|', title='ρ Gradients')
    axes[1, 2].legend(fontsize=8); axes[1, 2].set_yscale('log'); axes[1, 2].grid(alpha=0.3)

    # 8) Price scatter (if PVs provided)
    if model_pvs is not None and market_pvs is not None:
        m_np = np.asarray(model_pvs) if not hasattr(model_pvs, 'cpu') else model_pvs.cpu().numpy()
        k_np = np.asarray(market_pvs) if not hasattr(market_pvs, 'cpu') else market_pvs.cpu().numpy()
        axes[1, 3].scatter(k_np, m_np, alpha=0.5, s=30)
        _mx = max(k_np.max(), m_np.max())
        axes[1, 3].plot([0, _mx], [0, _mx], 'r--', lw=2, label='Perfect')
        axes[1, 3].set(xlabel='Market PV', ylabel='Model PV', title='Price Fit')
        axes[1, 3].legend(); axes[1, 3].grid(alpha=0.3)
    else:
        axes[1, 3].text(0.5, 0.5, 'No PV data', ha='center', va='center',
                        transform=axes[1, 3].transAxes, fontsize=14, color='grey')
        axes[1, 3].set_title('Price Fit (skipped)')

    plt.tight_layout()
    plt.show()

    # Summary stats
    losses = hist_df['loss'].values
    best_idx = np.argmin(losses)
    print(f"SPSA: {len(history)} iters | best loss={losses[best_idx]:.4e} at iter {best_idx} | final={losses[-1]:.4e}")
