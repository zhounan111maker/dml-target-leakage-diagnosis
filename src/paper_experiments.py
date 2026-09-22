"""
Reproduces every number and figure in paper.md.

    python src/paper_experiments.py

Outputs
-------
  results/main_results.csv    30-seed three-arm comparison
  results/sweep_results.csv   leakage-intensity sweep
  results/paper_tables.json   every reported value, machine-readable
  figures_paper/fig{1..4}_*.pdf / .png

Completed model fits are cached, so figures can be regenerated without
re-running the estimators. Delete the CSVs in results/ to force a recompute.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy import stats
from sklearn.model_selection import train_test_split

from src.dgp import (load_data, simulate_treatment, simulate_outcome,
                     true_ate_full, ground_truth_table, TREAT_MULT, OUTCOME_BASE)
from src.features import build_features, FEATURES
from src.dml import dml_estimate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAIN_SEEDS = 30
SWEEP_SEEDS = 15
ALPHAS = np.round(np.arange(0.0, 1.01, 0.1), 2)

FIG_DIR = 'figures_paper'
RES_DIR = 'results'
MAIN_CSV = os.path.join(RES_DIR, 'main_results.csv')
SWEEP_CSV = os.path.join(RES_DIR, 'sweep_results.csv')

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

C_ORIG, C_LEAK, C_CLEAN, C_ACC = '#4C72B0', '#C44E52', '#55A868', '#8172B2'

plt.rcParams.update({
    'figure.dpi': 140, 'savefig.dpi': 300, 'font.size': 10,
    'font.family': 'sans-serif', 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.autolayout': True,
})


def save_fig(fig, name):
    """Save in three formats: PDF (paper), PNG (slides), SVG (GitHub-renderable text)."""
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(f'{FIG_DIR}/{name}.{ext}', bbox_inches='tight')


def rule(title):
    print('\n' + '=' * 74 + f'\n{title}\n' + '=' * 74)


# ===========================================================================
rule('Loading data')
df_encoded = load_data('data/BankChurners.csv')
X_all = df_encoded.drop(columns=[OUTCOME_BASE])
print(f'  n = {len(df_encoded)}, raw covariates = {X_all.shape[1]}')


# ===========================================================================
rule('Table 1 - Ground-truth definitions')
_rng = np.random.RandomState(42)
_D0 = simulate_treatment(df_encoded, _rng)
_y0 = simulate_outcome(df_encoded, _D0)
table1 = ground_truth_table(df_encoded, _D0, _y0)
for k, v in table1.items():
    print(f'  {k:28s} {v:>12}')
print(f'\n  The treatment arm baseline is {table1["arm_imbalance_pct"]}% higher '
      f'than control.')
print('  The two group means are therefore NOT interchangeable.')


# ===========================================================================
rule(f'Main comparison - 3 arms x {MAIN_SEEDS} independent seeds')

if os.path.exists(MAIN_CSV):
    res = pd.read_csv(MAIN_CSV)
    print(f'  [cache] {MAIN_CSV}')
else:
    rows = []
    for seed in range(MAIN_SEEDS):
        rng = np.random.RandomState(seed)
        D = simulate_treatment(df_encoded, rng)
        y = simulate_outcome(df_encoded, D)
        tau = true_ate_full(df_encoded[OUTCOME_BASE])

        Xtr, Xte, Dtr, Dte, ytr, yte = train_test_split(
            X_all, D, y, test_size=0.2, random_state=seed)

        # Y(0) on the training rows only -- consumed solely by the leak channel
        y0_tr = df_encoded.loc[Xtr.index, OUTCOME_BASE]

        # Derived features are APPENDED, never substituted: this isolates the
        # contribution of the derived features from that of feature count.
        X_leaky = pd.concat([Xtr, build_features(Xtr, y0_tr, 1.0)], axis=1)
        X_clean = pd.concat([Xtr, build_features(Xtr, y0_tr, 0.0)], axis=1)

        ate_o, se_o = dml_estimate(Xtr, Dtr, ytr)
        ate_l, se_l = dml_estimate(X_leaky, Dtr, ytr)
        ate_c, se_c = dml_estimate(X_clean, Dtr, ytr)

        rows.append(dict(
            seed=seed,
            ate_orig=ate_o, ate_leaky=ate_l, ate_clean=ate_c,
            se_orig=se_o, se_leaky=se_l, se_clean=se_c,
            err_orig=abs(ate_o - tau), err_leaky=abs(ate_l - tau),
            err_clean=abs(ate_c - tau)))
        if (seed + 1) % 10 == 0:
            print(f'    seed {seed + 1}/{MAIN_SEEDS}')

    res = pd.DataFrame(rows)
    res.to_csv(MAIN_CSV, index=False)
    print(f'  [saved] {MAIN_CSV}')

summary = pd.DataFrame([
    dict(config='A - Original',
         mean_bias=res.err_orig.mean(), sd_bias=res.err_orig.std(),
         mean_ate=res.ate_orig.mean(), sd_ate=res.ate_orig.std(),
         mean_se=res.se_orig.mean()),
    dict(config='B - Derived (leaky)',
         mean_bias=res.err_leaky.mean(), sd_bias=res.err_leaky.std(),
         mean_ate=res.ate_leaky.mean(), sd_ate=res.ate_leaky.std(),
         mean_se=res.se_leaky.mean()),
    dict(config='C - Derived (clean)',
         mean_bias=res.err_clean.mean(), sd_bias=res.err_clean.std(),
         mean_ate=res.ate_clean.mean(), sd_ate=res.ate_clean.std(),
         mean_se=res.se_clean.mean()),
])
print('\n' + summary.round(3).to_string(index=False))

base = res.err_orig.mean()
print('\n  Change in mean |bias| vs Arm A:')
for name, col in [('B (leaky)', 'err_leaky'), ('C (clean)', 'err_clean')]:
    chg = (res[col].mean() - base) / base * 100
    wins = int((res[col] < res.err_orig).sum())
    print(f'    {name:12s} {chg:+7.2f}%   wins {wins}/{MAIN_SEEDS}')

t_lc, p_lc = stats.ttest_rel(res.err_leaky, res.err_clean)
t_co, p_co = stats.ttest_rel(res.err_clean, res.err_orig)
print(f'\n  Paired t  B vs C : t = {t_lc:.4f}   p = {p_lc:.3e}')
print(f'  Paired t  C vs A : t = {t_co:.4f}   p = {p_co:.4f}   <-- the key result')
print(f'\n  Variance collapse  SD(theta) : {res.ate_orig.std():.3f} -> '
      f'{res.ate_leaky.std():.3f}  '
      f'({res.ate_orig.std() / res.ate_leaky.std():.2f}x)')
print(f'  Reported SE collapse         : {res.se_orig.mean():.3f} -> '
      f'{res.se_leaky.mean():.3f}  '
      f'({res.se_orig.mean() / res.se_leaky.mean():.2f}x)')


# ===========================================================================
rule('Figure 1 - ATE bias boxplot (variance-collapse fingerprint)')
fig, ax = plt.subplots(figsize=(5.6, 3.9))
data = [res.err_orig, res.err_leaky, res.err_clean]
labels = ['Original\nfeatures', 'Derived\n(leaky)', 'Derived\n(clean)']
colors = [C_ORIG, C_LEAK, C_CLEAN]

bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55,
                medianprops=dict(color='black', linewidth=1.4),
                flierprops=dict(marker='o', markersize=3.5, alpha=0.5))
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c); patch.set_alpha(0.65)
    patch.set_edgecolor('black'); patch.set_linewidth(0.8)

_rj = np.random.RandomState(0)
for i, (d, c) in enumerate(zip(data, colors), start=1):
    ax.scatter(_rj.normal(i, 0.055, size=len(d)), d, s=11, color=c,
               edgecolor='white', linewidth=0.4, alpha=0.8, zorder=3)

ax.set_ylabel('Absolute ATE bias')
ax.set_title(f'ATE bias over {MAIN_SEEDS} independent seeds')
ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)
ax.set_axisbelow(True)
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
save_fig(fig, 'fig1_bias_boxplot')
plt.close(fig)


# ===========================================================================
rule('Figure 2 - Per-seed win/loss scatter')
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
for ax, (b_col, title) in zip(axes, [
        ('err_leaky', 'Original vs Derived (leaky)'),
        ('err_clean', 'Original vs Derived (clean)')]):
    a, b = res.err_orig.values, res[b_col].values
    lim = [0, max(a.max(), b.max()) * 1.08]
    win = int((b < a).sum())
    ax.plot(lim, lim, ls='--', lw=1.0, color='gray', zorder=1)
    ax.scatter(a[b < a], b[b < a], s=34, facecolor=C_CLEAN, edgecolor='black',
               lw=0.5, zorder=3, label=f'derived wins ({win})')
    ax.scatter(a[b >= a], b[b >= a], s=34, facecolor=C_ORIG, edgecolor='black',
               lw=0.5, zorder=3, label=f'original wins ({len(a) - win})')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Original features: |bias|')
    ax.set_ylabel('Derived features: |bias|')
    ax.set_title(title)
    ax.legend(loc='upper left', frameon=False)
    ax.grid(ls=':', lw=0.6, alpha=0.6); ax.set_axisbelow(True)
axes[0].text(0.97, 0.06, f'p = {p_lc:.1e} (paired)', transform=axes[0].transAxes,
             ha='right', fontsize=8.5, color=C_LEAK)
axes[1].text(0.97, 0.06, f'p = {p_co:.3f} (not significant)',
             transform=axes[1].transAxes, ha='right', fontsize=8.5, color='gray')
save_fig(fig, 'fig2_winrate_scatter')
plt.close(fig)


# ===========================================================================
rule(f'Leakage-intensity sweep - {len(ALPHAS)} alphas x {SWEEP_SEEDS} seeds')

if os.path.exists(SWEEP_CSV):
    sw = pd.read_csv(SWEEP_CSV)
    print(f'  [cache] {SWEEP_CSV}')
else:
    sweep = []
    for alpha in ALPHAS:
        for seed in range(SWEEP_SEEDS):
            rng = np.random.RandomState(1000 + seed)
            D = simulate_treatment(df_encoded, rng)
            y = simulate_outcome(df_encoded, D)
            tau = true_ate_full(df_encoded[OUTCOME_BASE])

            Xtr, _, Dtr, _, ytr, _ = train_test_split(
                X_all, D, y, test_size=0.2, random_state=seed)
            y0_tr = df_encoded.loc[Xtr.index, OUTCOME_BASE]

            Xa = pd.concat([Xtr, build_features(Xtr, y0_tr, alpha)], axis=1)
            ate, se = dml_estimate(Xa, Dtr, ytr)
            sweep.append(dict(alpha=float(alpha), seed=seed, ate=ate,
                              bias=ate - tau, abs_bias=abs(ate - tau), se=se))
        print(f'    alpha = {alpha:.1f} done')
    sw = pd.DataFrame(sweep)
    sw.to_csv(SWEEP_CSV, index=False)
    print(f'  [saved] {SWEEP_CSV}')

agg = sw.groupby('alpha').agg(
    mean_abs_bias=('abs_bias', 'mean'), sd_ate=('ate', 'std'),
    mean_ate=('ate', 'mean'), mean_se=('se', 'mean')).reset_index()
print('\n' + agg.round(3).to_string(index=False))
print(f'\n  SD(theta) collapse across alpha: {agg.sd_ate.iloc[0]:.2f} -> '
      f'{agg.sd_ate.iloc[-1]:.2f}  ({agg.sd_ate.iloc[0]/agg.sd_ate.iloc[-1]:.2f}x)')
print(f'  Reported SE collapse           : {agg.mean_se.iloc[0]:.2f} -> '
      f'{agg.mean_se.iloc[-1]:.2f}  ({agg.mean_se.iloc[0]/agg.mean_se.iloc[-1]:.2f}x)')


# ===========================================================================
rule('Figure 3 - Bias response to leakage intensity')
fig, ax = plt.subplots(figsize=(5.6, 3.9))
ax.plot(agg.alpha, agg.mean_abs_bias, marker='o', ms=5.5, lw=1.8,
        color=C_LEAK, label='Derived features (varying $\\alpha$)')
ax.axhline(base, ls='--', lw=1.2, color=C_ORIG,
           label=f'Original features ({base:.1f})')
ax.fill_between(agg.alpha, 0, agg.mean_abs_bias, color=C_LEAK, alpha=0.10)
ax.set_xlabel('Leakage intensity  ' + r'$\alpha$')
ax.set_ylabel('Mean |ATE bias|')
ax.set_title('Bias declines as leakage increases')
ax.legend(frameon=False, loc='upper right')
ax.grid(ls=':', lw=0.6, alpha=0.6); ax.set_axisbelow(True)
save_fig(fig, 'fig3_sweep_bias')
plt.close(fig)


# ===========================================================================
rule('Figure 4 - Dispersion collapse')
fig, ax = plt.subplots(figsize=(5.6, 3.9))
ax.plot(agg.alpha, agg.sd_ate, marker='s', ms=5.5, lw=1.8, color=C_ACC,
        label='SD of ATE estimates (across seeds)')
ax.plot(agg.alpha, agg.mean_se, marker='^', ms=5.5, lw=1.8, color=C_LEAK,
        label='Mean reported SE (from OLS)')
ax.set_xlabel('Leakage intensity  ' + r'$\alpha$')
ax.set_ylabel('Dispersion')
ax.set_title('Variance collapse: the fingerprint of leakage')
ax.legend(frameon=False, loc='upper right')
ax.grid(ls=':', lw=0.6, alpha=0.6); ax.set_axisbelow(True)

ratio = agg.sd_ate.iloc[0] / agg.sd_ate.iloc[-1]
ax.annotate(f'{ratio:.1f}x collapse',
            xy=(agg.alpha.iloc[-1], agg.sd_ate.iloc[-1]),
            xytext=(-80, 30), textcoords='offset points', fontsize=9,
            color=C_LEAK,
            arrowprops=dict(arrowstyle='->', color=C_LEAK, lw=1.0))
save_fig(fig, 'fig4_sweep_variance')
plt.close(fig)


# ===========================================================================
rule('Exporting tables')
tables = {
    'table1_ground_truth': table1,
    'table2_main_summary': summary.round(4).to_dict(orient='records'),
    'table2_stats': {
        'n_seeds': MAIN_SEEDS,
        'bias_change_leaky_pct': round(float((res.err_leaky.mean() - base) / base * 100), 2),
        'bias_change_clean_pct': round(float((res.err_clean.mean() - base) / base * 100), 2),
        'wins_leaky': int((res.err_leaky < res.err_orig).sum()),
        'wins_clean': int((res.err_clean < res.err_orig).sum()),
        'paired_t_leaky_vs_clean': round(float(t_lc), 4),
        'paired_p_leaky_vs_clean': float(f'{p_lc:.3e}'),
        'paired_t_clean_vs_orig': round(float(t_co), 4),
        'paired_p_clean_vs_orig': float(f'{p_co:.6f}'),
        'sd_ate_orig': round(float(res.ate_orig.std()), 3),
        'sd_ate_leaky': round(float(res.ate_leaky.std()), 3),
        'sd_ate_clean': round(float(res.ate_clean.std()), 3),
        'variance_collapse_ratio': round(float(res.ate_orig.std() / res.ate_leaky.std()), 2),
        'se_collapse_ratio': round(float(res.se_orig.mean() / res.se_leaky.mean()), 2),
    },
    'table3_sweep': agg.round(4).to_dict(orient='records'),
}
with open(os.path.join(RES_DIR, 'paper_tables.json'), 'w', encoding='utf-8') as f:
    json.dump(tables, f, ensure_ascii=False, indent=2)

print(f'  -> {RES_DIR}/paper_tables.json')
print(f'  -> {FIG_DIR}/ (4 figures, PDF + 300dpi PNG)')
print('\n  DONE')
