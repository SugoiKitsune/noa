"""
Generate the combined 2x3 vol surface chart using the exact same
plot_caplet_vol_surface() function that the notebook calls inside evaluate_vol_surface().
Matches the notebook output exactly: maturities 1Y, 3Y, 5Y, 7Y, 10Y + forward curve.
"""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent.parent.parent  # noa/
DATA_DIR   = Path(__file__).parent.parent / 'multi_theta' / 'calibration_weights'
IMAGES_DIR = Path(__file__).parent.parent / 'multi_theta' / 'images'
DATA_CSV_DIR = ROOT / 'data'

sys.path.insert(0, str(ROOT))

from affine_calibration.scripts.plotting import plot_caplet_vol_surface

df          = pd.read_csv(DATA_DIR / 'market_vs_model_v22.csv')
fwd_key_rate = pd.read_csv(DATA_CSV_DIR / 'forward_key_rate.csv')

# Monkey-patch plt.show to capture the figure before it's displayed
import matplotlib.pyplot as plt as _plt_real  # noqa — handled below

_captured_fig = []
_orig_show = plt.show
def _capture_show(*a, **kw):
    _captured_fig.append(plt.gcf())
plt.show = _capture_show

plot_caplet_vol_surface(
    df,
    version_name='v22',
    plot_maturities=[1.0, 3.0, 5.0, 7.0, 10.0],
    fwd_key_rate=fwd_key_rate,
)

plt.show = _orig_show
fig = _captured_fig[0]

out = IMAGES_DIR / 'vol_surface_combined_v22.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}")


df = pd.read_csv(DATA_DIR / 'market_vs_model_v22.csv')

model_col  = 'model_vol_v22'
error_col  = 'vol_error_v22'
mkt_col    = 'implied_normal_vol'
version    = 'v22'

maturities = sorted(df['time_to_maturity'].unique())
strikes    = sorted(df['strike'].unique())
T_grid, K_grid = np.meshgrid(np.array(maturities), np.array(strikes))

def build_grid(col):
    g = np.full_like(T_grid, np.nan)
    for i, k in enumerate(strikes):
        for j, t in enumerate(maturities):
            row = df[(df['time_to_maturity'] == t) & (df['strike'] == k)]
            if len(row) and not pd.isna(row[col].values[0]):
                g[i, j] = row[col].values[0]
    return g

mkt_grid   = build_grid(mkt_col) * 100
model_grid = build_grid(model_col) * 100
error_grid = build_grid(error_col) * 100

# Fill NaN in model for 3D display
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

# ── Plot maturities for 2D row ────────────────────────────────────────────────
plot_mats = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]  # 6M, 1Y, 2Y, 3Y, 5Y, 10Y
colors = plt.cm.tab10(np.linspace(0, 1, len(plot_mats)))
mat_labels = {0.5: '6M', 1.0: '1Y', 2.0: '2Y', 3.0: '3Y', 5.0: '5Y', 10.0: '10Y'}

fig = plt.figure(figsize=(20, 12))
view_elev, view_azim = 28, 130

# ── Row 1: 3D surfaces ───────────────────────────────────────────────────────
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
surf1 = ax1.plot_surface(T_grid, K_grid * 100, mkt_grid,
                         cmap='viridis', alpha=0.92, edgecolor='none',
                         vmin=z_lo, vmax=z_hi)
ax1.set_xlabel('Maturity (y)', fontsize=10, labelpad=5)
ax1.set_ylabel('Strike (%)',   fontsize=10, labelpad=5)
ax1.set_zlabel('Vol (%)',      fontsize=10, labelpad=5)
ax1.set_zlim(z_lo, z_hi)
ax1.set_title('Market Volatility Surface', fontsize=12, fontweight='bold')
ax1.view_init(elev=view_elev, azim=view_azim)
fig.colorbar(surf1, ax=ax1, shrink=0.45, aspect=10, pad=0.1)

ax2 = fig.add_subplot(2, 3, 2, projection='3d')
surf2 = ax2.plot_surface(T_grid, K_grid * 100, model_disp,
                         cmap='viridis', alpha=0.92, edgecolor='none',
                         vmin=z_lo, vmax=z_hi)
ax2.set_xlabel('Maturity (y)', fontsize=10, labelpad=5)
ax2.set_ylabel('Strike (%)',   fontsize=10, labelpad=5)
ax2.set_zlabel('Vol (%)',      fontsize=10, labelpad=5)
ax2.set_zlim(z_lo, z_hi)
ax2.set_title(f'{version.upper()} Model Volatility Surface', fontsize=12, fontweight='bold')
ax2.view_init(elev=view_elev, azim=view_azim)
fig.colorbar(surf2, ax=ax2, shrink=0.45, aspect=10, pad=0.1)

ax3 = fig.add_subplot(2, 3, 3, projection='3d')
surf3 = ax3.plot_surface(T_grid, K_grid * 100, error_disp,
                         cmap='RdBu_r', alpha=0.92, edgecolor='none',
                         vmin=-err_lim, vmax=err_lim)
ax3.set_xlabel('Maturity (y)', fontsize=10, labelpad=5)
ax3.set_ylabel('Strike (%)',   fontsize=10, labelpad=5)
ax3.set_zlabel('Error (%)',    fontsize=10, labelpad=5)
ax3.set_title(f'{version.upper()} − Market (Error Surface)', fontsize=12, fontweight='bold')
ax3.view_init(elev=view_elev, azim=view_azim)
fig.colorbar(surf3, ax=ax3, shrink=0.45, aspect=10, pad=0.1)

# ── Row 2: 2D slice panels ───────────────────────────────────────────────────
ax4 = fig.add_subplot(2, 3, 4)
for ci, mat in enumerate(plot_mats):
    t_avail = min(maturities, key=lambda x: abs(x - mat))
    sub = df[df['time_to_maturity'] == t_avail].sort_values('strike')
    lbl = mat_labels.get(mat, f'{mat}Y')
    ax4.plot(sub['strike'] * 100, sub[mkt_col] * 100,
             'o-', color=colors[ci], ms=3, lw=2, label=f'{lbl} Market')
ax4.set_xlabel('Strike (%)', fontsize=11)
ax4.set_ylabel('Implied Vol (%)', fontsize=11)
ax4.set_title('Market Vol — Maturity Slices', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9, ncol=2); ax4.grid(alpha=0.3)

ax5 = fig.add_subplot(2, 3, 5)
for ci, mat in enumerate(plot_mats):
    t_avail = min(maturities, key=lambda x: abs(x - mat))
    sub = df[df['time_to_maturity'] == t_avail].sort_values('strike')
    lbl = mat_labels.get(mat, f'{mat}Y')
    ax5.plot(sub['strike'] * 100, sub[mkt_col] * 100,
             '-', color=colors[ci], alpha=0.3, lw=1.5)
    valid = ~sub[model_col].isna()
    ax5.plot(sub.loc[valid, 'strike'] * 100, sub.loc[valid, model_col] * 100,
             's-', color=colors[ci], ms=3, lw=2, label=lbl)
ax5.set_xlabel('Strike (%)', fontsize=11)
ax5.set_ylabel('Implied Vol (%)', fontsize=11)
ax5.set_title(f'{version.upper()} Model vs Market (faint)', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9, ncol=2); ax5.grid(alpha=0.3)

ax6 = fig.add_subplot(2, 3, 6)
for ci, mat in enumerate(plot_mats):
    t_avail = min(maturities, key=lambda x: abs(x - mat))
    sub = df[df['time_to_maturity'] == t_avail].sort_values('strike')
    lbl = mat_labels.get(mat, f'{mat}Y')
    valid = ~sub[error_col].isna()
    ax6.plot(sub.loc[valid, 'strike'] * 100, sub.loc[valid, error_col] * 100,
             '^-', color=colors[ci], ms=3, lw=1.5, label=f'{lbl}')
ax6.axhline(0, color='k', ls='--', lw=1, alpha=0.5)
ax6.set_xlabel('Strike (%)', fontsize=11)
ax6.set_ylabel('Vol Error (%)', fontsize=11)
ax6.set_title(f'{version.upper()} − Market Error by Maturity', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9, ncol=2); ax6.grid(alpha=0.3)

fig.suptitle(f'Bachelier Implied Vol Surface — {version.upper()} Calibration (200k paths, daily timeline)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
out = IMAGES_DIR / f'vol_surface_combined_{version}.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}")
