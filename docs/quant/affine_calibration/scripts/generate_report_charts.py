"""
Generate static PNG charts for the intermediate report from v22 calibration results.
Loads pre-computed v22 CSV and pkl to avoid re-running MC.
"""
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless rendering
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DATA_DIR      = Path(__file__).parent.parent / 'multi_theta' / 'calibration_weights'
IMAGES_DIR    = Path(__file__).parent.parent / 'multi_theta' / 'images'
IMAGES_DIR.mkdir(exist_ok=True)

VOL_CSV  = DATA_DIR / 'market_vs_model_v22.csv'
CKPT_PKL = DATA_DIR / 'fast_calibration_v22.pkl'

print(f"Loading {VOL_CSV}...")
df = pd.read_csv(VOL_CSV)
print("CSV columns:", df.columns.tolist())
print(df.head())

print(f"Loading {CKPT_PKL}...")
with open(CKPT_PKL, 'rb') as f:
    ckpt = pickle.load(f)
print("Checkpoint keys:", list(ckpt.keys())[:10])

# ── Detect column names ─────────────────────────────────────────────────────
model_col = next(c for c in df.columns if c.startswith('model_vol'))
error_col  = next(c for c in df.columns if c.startswith('vol_error'))
mkt_col    = 'implied_normal_vol'

version_name = model_col.replace('model_vol_', '')

maturities = sorted(df['time_to_maturity'].unique())
strikes    = sorted(df['strike'].unique())

T_grid_1d = np.array(maturities)
K_grid_1d = np.array(strikes)
T_grid, K_grid = np.meshgrid(T_grid_1d, K_grid_1d)

def build_grid(col):
    g = np.full_like(T_grid, np.nan)
    for i, k in enumerate(strikes):
        for j, t in enumerate(maturities):
            row = df[(df['time_to_maturity'] == t) & (df['strike'] == k)]
            if len(row) > 0:
                g[i, j] = row[col].values[0]
    return g

mkt_grid   = build_grid(mkt_col) * 100
model_grid = build_grid(model_col) * 100
error_grid = build_grid(error_col) * 100

# ── Fill NaN in model_grid for surface ──────────────────────────────────────
from scipy.ndimage import distance_transform_edt
nan_mask = np.isnan(model_grid)
if nan_mask.any() and (~nan_mask).any():
    _, idx = distance_transform_edt(nan_mask, return_distances=True, return_indices=True)
    model_disp = model_grid.copy()
    model_disp[nan_mask] = model_grid[tuple(idx[:, nan_mask])]
else:
    model_disp = model_grid.copy()
error_disp = np.where(np.isnan(error_grid), 0, error_grid)
err_lim = np.nanpercentile(np.abs(error_grid[~np.isnan(error_grid)]), 95)

z_lo = min(np.nanmin(mkt_grid), np.nanmin(model_disp)) * 0.88
z_hi = max(np.nanmax(mkt_grid), np.nanmax(model_disp)) * 1.08

# ── Figure 1: 3D surface comparison ─────────────────────────────────────────
print("Generating 3D surface chart...")
fig = plt.figure(figsize=(18, 6))
view_elev, view_azim = 28, 130

for pos, (grid, cmap, title) in enumerate([
    (mkt_grid,   'viridis',  'Market Vol Surface'),
    (model_disp, 'viridis',  f'Model Vol Surface ({version_name})'),
    (error_disp, 'RdBu_r',   'Error  (Model − Market)'),
], 1):
    ax = fig.add_subplot(1, 3, pos, projection='3d')
    if pos < 3:
        surf = ax.plot_surface(T_grid, K_grid * 100, grid,
                               cmap=cmap, alpha=0.92, edgecolor='none',
                               vmin=z_lo, vmax=z_hi)
        ax.set_zlim(z_lo, z_hi)
    else:
        surf = ax.plot_surface(T_grid, K_grid * 100, grid,
                               cmap=cmap, alpha=0.92, edgecolor='none',
                               vmin=-err_lim, vmax=err_lim)
    ax.set_xlabel('Maturity (y)', fontsize=9, labelpad=4)
    ax.set_ylabel('Strike (%)', fontsize=9, labelpad=4)
    ax.set_zlabel('Vol (%)', fontsize=9, labelpad=4)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.view_init(elev=view_elev, azim=view_azim)
    fig.colorbar(surf, ax=ax, shrink=0.45, aspect=10, pad=0.1)

fig.suptitle(f'Bachelier Implied Vol Surface — {version_name.upper()} Calibration',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
out = IMAGES_DIR / f'3d_vol_surface_{version_name}.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out}")

# ── Figure 2: 2D per-maturity smile slices ───────────────────────────────────
print("Generating 2D smile slices chart...")
plot_mats  = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
mat_labels = {0.25: '3M', 0.5: '6M', 1.0: '1Y', 2.0: '2Y', 5.0: '5Y', 10.0: '10Y'}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, T_target in zip(axes, plot_mats):
    # Nearest available maturity
    T_avail = maturities[np.argmin(np.abs(np.array(maturities) - T_target))]
    sub = df[df['time_to_maturity'] == T_avail].sort_values('strike')
    if sub.empty:
        ax.set_title(f'No data for T={T_target}')
        continue

    K_pct = sub['strike'].values * 100
    mkt_v  = sub[mkt_col].values * 100
    mod_v  = sub[model_col].values * 100

    ax.plot(K_pct, mkt_v, 'ko-', ms=4, lw=1.5, label='Market')
    ax.plot(K_pct, mod_v, 'r--', lw=2, label='Model')
    ax.fill_between(K_pct, mkt_v, mod_v, alpha=0.15, color='red')

    T_use = T_avail
    valid = ~np.isnan(sub[error_col].values)
    rmse = np.sqrt(np.mean((sub[error_col].values[valid] * 100) ** 2)) if valid.any() else np.nan
    ax.set_title(f'T = {mat_labels.get(T_target, f"{T_use:.2f}Y")}  (RMSE = {rmse:.2f}%)',
                 fontsize=11)
    ax.set_xlabel('Strike (%)', fontsize=9)
    ax.set_ylabel('Normal Implied Vol (%)', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle(f'Market vs Model Vol Smiles — {version_name.upper()}', fontsize=13, fontweight='bold')
plt.tight_layout()
out = IMAGES_DIR / f'2d_smile_slices_{version_name}.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out}")

# ── Figure 3: Convergence + parameter evolution ───────────────────────────────
print("Generating convergence chart...")
history = ckpt.get('history', [])
if history:
    hist_df = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Loss curve
    axes[0].plot(hist_df['iter'], hist_df['loss'], 'b-', lw=2)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Iteration', fontsize=10)
    axes[0].set_ylabel('Loss (log scale)', fontsize=10)
    axes[0].set_title('Loss Convergence', fontsize=11, fontweight='bold')
    axes[0].grid(alpha=0.3)
    best_iter = hist_df.loc[hist_df['loss'].idxmin(), 'iter']
    best_loss = hist_df['loss'].min()
    axes[0].axvline(best_iter, color='r', ls='--', lw=1.5,
                    label=f'Best @ iter {best_iter}\n({best_loss:.4f})')
    axes[0].legend(fontsize=9)

    # θ node convergence — use 'theta' key (factor 1) from history
    node_labels = ['3M', '6M', '1Y', '3Y', '5Y', '10Y']
    theta_key = 'theta'   # history stores factor-1 θ as 'theta'
    for i, lbl in enumerate(node_labels):
        axes[1].plot(hist_df['iter'],
                     [np.sqrt(max(h[theta_key][i], 1e-10)) * 100 for h in history],
                     '-', lw=2, label=f'θ₁({lbl})')
    axes[1].set_title('θ₁(t) Node Convergence', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Iteration', fontsize=10)
    axes[1].set_ylabel('√θ (%)', fontsize=10)
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(alpha=0.3)

    # κ_fast, κ_slow, w
    ax2 = axes[2]
    ax2r = ax2.twinx()
    ax2.plot(hist_df['iter'], hist_df['kappa_fast'], 'r-', lw=2, label='κ_fast')
    ax2.plot(hist_df['iter'], hist_df['kappa_slow'], 'b-', lw=2, label='κ_slow')
    ax2r.plot(hist_df['iter'], hist_df['w_fast'], 'g--', lw=2, label='w')
    ax2.set_xlabel('Iteration', fontsize=10)
    ax2.set_ylabel('Mean-reversion speed κ', fontsize=10)
    ax2r.set_ylabel('Mixing weight w', color='g', fontsize=10)
    ax2r.tick_params(axis='y', labelcolor='g')
    ax2.set_title('2-Factor CIR Parameters', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2r.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle(f'Adam-SPSA Calibration Diagnostics — {version_name.upper()}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = IMAGES_DIR / f'convergence_{version_name}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")
else:
    print("  No history found in checkpoint — skipping convergence plot")

print("\nAll charts generated.")
