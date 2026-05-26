import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def _find_latest_eval(scenario: str) -> str:
    """Return the evaluations.npz path for the latest training run of *scenario*."""
    matches = sorted(glob.glob(f'logs/protagonist_{scenario}_*/evaluations.npz'))
    if not matches:
        raise FileNotFoundError(
            f"No evaluations.npz found for scenario '{scenario}'. "
            "Run training first (scripts/run_full_retrain.ps1) or update the path manually."
        )
    return matches[-1]


scenarios = {
    'Crossing':   _find_latest_eval('crossing'),
    'Overtaking': _find_latest_eval('overtaking'),
    'Head-on':    _find_latest_eval('head_on'),
    'Merging':    _find_latest_eval('merging'),
}

colors  = ['#1f77b4', '#2ca02c', '#d62728', '#8c6d31']
styles  = ['-',       '--',      '-.',      ':']
markers = ['o',       's',       '^',       'D']

fig, ax = plt.subplots(figsize=(9, 5.5))

for (name, path), color, style, marker in zip(scenarios.items(), colors, styles, markers):
    d    = np.load(path)
    ts   = d['timesteps'] / 1000.0
    rs   = d['results']
    mean = rs.mean(axis=1)
    std  = rs.std(axis=1)

    alpha_ema = 0.4
    ema = np.zeros_like(mean)
    ema[0] = mean[0]
    for i in range(1, len(mean)):
        ema[i] = alpha_ema * mean[i] + (1 - alpha_ema) * ema[i - 1]

    ax.fill_between(ts, mean - std, mean + std, alpha=0.12, color=color)
    ax.plot(ts, mean, color=color, alpha=0.35, linewidth=0.8, linestyle=style)
    ax.plot(ts, ema,  color=color, linewidth=2.0, linestyle=style,
            marker=marker, markevery=4, markersize=7, label=name)

ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xlabel('Training steps (\u00d710\u00b3)', fontsize=12)
ax.set_ylabel('Mean episode reward', fontsize=12)
ax.set_xlim(0, 100)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()

out = 'figures/fig_training_curves.png'
os.makedirs('figures', exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches='tight')
print('Saved:', out)
