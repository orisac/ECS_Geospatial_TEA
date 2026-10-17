"""
run_TEA: cost breakdown, uncertainty and sensitivity analysis for electrochemical
nitrogen recovery into four fertilizer products.

Figures
  1  Stacked cost breakdown per kg product, with Monte Carlo 5th-95th percentile error bars
  2  Same breakdown per kg N
  3  Ammonium nitrate one-at-a-time sensitivity (tornado), 10% parameter nudge
  4  Same, 20% nudge
  5  Same, 30% nudge

Inputs are read from TEA_inputs.xlsx; the model lives in adapted_TEA_model.py.
"""

import copy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from scipy.special import expit

from adapted_TEA_model import EconomicCase, load_product_cases

# ------------------------------------------------------------------ #
# Plot style
# ------------------------------------------------------------------ #
WIDTH = 3.5
HEIGHT = 3.5
BLUE = np.array([33.05, 86.66, 211.75]) / 255
RED = np.array([206.04, 59.16, 59.16]) / 255
FONT_SIZE = 10

# One color per entry of EconomicCase.BREAKDOWN_LABELS
BREAKDOWN_COLORS = [
    '#CE3B3B',  # Electricity
    '#007F00',  # Chemicals
    '#D29B19',  # Overhead
    '#750075',  # Balance of Plant
    '#12A8AB',  # Reactor Housing
    '#B97341',  # Catalyst
    '#4472C4',  # Membrane
]
BD_LABELS = EconomicCase.BREAKDOWN_LABELS

# ------------------------------------------------------------------ #
# Build one TEA case per fertilizer product
# ------------------------------------------------------------------ #
INPUTS = Path(__file__).resolve().parent / 'TEA_inputs.xlsx'
products, cases = load_product_cases(INPUTS)
names = products['name'].tolist()
ammonium_sulfate, ammonium_nitrate = cases[0], cases[1]

# Nitrogen mass fraction of each product, for the per-kg-N view
N_MW = 14.007
n_frac = (products['n_atoms_N'] * N_MW / products['prod_mw']).values

# ------------------------------------------------------------------ #
# Uncertain parameters (shared by the Monte Carlo and tornado analyses)
# ------------------------------------------------------------------ #
# normal_params    (attr, label, frac): Gaussian, std = frac x |baseline|;
#                  the tornado nudges the parameter by +/- frac.
# lognormal_params (attr, label, sigma_log_mc, oat_decades): strictly positive,
#                  sampled as baseline x exp(N(0, sigma_log_mc)). sigma_ln = 1.0 spans
#                  the order-of-magnitude range of real waste-stream concentrations
#                  (5th-95th = 0.19x-5.2x). The tornado instead shifts log10(C) by a
#                  fixed +/- oat_decades (0.1 decades = 0.794x-1.259x), because a
#                  percentage of log10(C) is unit-dependent.
# sigmoid_params   (attr, label, frac): bounded on (0, 1), sampled in logit space
#                  with std_logit = frac / (1 - baseline).
normal_params = [
    ('electricity_price',    'Electricity cost (\\$/kWh)',  0.1),
    ('catalyst_price',       'Catalyst cost (\\$/kg)',      0.1),
    ('base_bop_price',       'BOP cost (\\$)',              0.1),
    ('current_density',      'Current density (A/m²)',      0.1),
    ('cell_gap',             'Cell gap (cm)',               0.1),
    ('conductivity',         'Conductivity (S/cm)',         0.1),
    ('solvent_price',        'Chemical cost (\\$/kg)',      0.1),
    ('catalyst_loading',     'Catalyst loading (mg/cm²)',   0.1),
    ('reference_area_price', 'Areal capital cost (\\$/m²)', 0.1),
    ('lifetime',             'Lifetime (yrs)',              0.1),
    ('production_rate',      'Production rate (mol/s)',     0.1),
    ('additional_factor',    'Overhead (%)',                0.1),
]

lognormal_params = [
    ('molarity', 'Magnitude of wastewater\nnitrogen concentration (-)', 1.0, 0.10),
]

sigmoid_params = [
    ('conversion', 'Single pass removal (%)', 0.1),
    ('product_fe', 'Current efficiency (%)',  0.1),
]

all_params = normal_params + lognormal_params + sigmoid_params


def _apply_param(case, attr, val):
    """Set one parameter, keeping conductivity tied to molarity as in EconomicCase.vary."""
    setattr(case, attr, val)
    if attr == 'molarity':
        case.conductivity = val / case.k


def _baseline_cost(case):
    return float(np.atleast_1d(case.cost).flat[0])


# ------------------------------------------------------------------ #
# Monte Carlo error bars (500 samples per product)
# ------------------------------------------------------------------ #
def _run_mc(case, N=500, seed=0):
    """Return the (5th, 95th) percentile cost from a Monte Carlo over all_params."""
    rng_mc = np.random.default_rng(seed)
    costs = np.full(N, np.nan)
    for k in range(N):
        try:
            c = copy.deepcopy(case)
            # Conductivity is drawn as a relative error and reapplied after molarity,
            # so the two draws compose instead of overwriting each other.
            cond_factor = 1.0
            for attr, _, std_frac in normal_params:
                bval = getattr(case, attr)
                val = float(rng_mc.normal(bval, std_frac * abs(bval)))
                if attr == 'conductivity':
                    cond_factor = val / bval
                    continue
                setattr(c, attr, val)
            for attr, _, sigma_log, _ in lognormal_params:
                bval = getattr(case, attr)
                _apply_param(c, attr, float(bval * np.exp(rng_mc.normal(0, sigma_log))))
            for attr, _, std_frac in sigmoid_params:
                bval = getattr(case, attr)
                logit_b = np.log(bval / (1.0 - bval))
                val = float(expit(rng_mc.normal(logit_b, std_frac / (1.0 - bval))))
                if attr == 'product_fe':
                    val = min(val, 1.0 - case.her_fe - 1e-6)
                setattr(c, attr, val)
            c.conductivity = c.molarity / c.k * cond_factor
            c.run_model()
            v = _baseline_cost(c)
            if np.isfinite(v) and v > 0:
                costs[k] = v
        except Exception:
            pass
    valid = np.isfinite(costs)
    if valid.sum() < 10:
        return np.nan, np.nan
    return tuple(np.percentile(costs[valid], [5, 95]))


mc_ranges = [_run_mc(c, N=500, seed=i) for i, c in enumerate(cases)]

print("\nElectrochemical stripping prices (5th–95th pct error bars):")
for name, case, (p5, p95) in zip(names, cases, mc_ranges):
    base_cost = _baseline_cost(case)
    if np.isfinite(p5) and np.isfinite(p95):
        print(f"  {name}: ${base_cost:.2f}/kg  (+${p95 - base_cost:.2f} / -${base_cost - p5:.2f})")
    else:
        print(f"  {name}: ${base_cost:.2f}/kg  (error bars unavailable)")


# ------------------------------------------------------------------ #
# Figures 1-2: stacked cost breakdown with Monte Carlo error bars
# ------------------------------------------------------------------ #
def _plot_cost_breakdown(ylabel, scale, tick_step=0.5):
    """Stacked-bar TEA breakdown for every product.

    `scale` is a per-product divisor applied to every cost, so the same code draws
    the per-kg-product (scale = 1) and per-kg-N (scale = N mass fraction) views.
    """
    fig, ax = plt.subplots(figsize=(2.5 * WIDTH, HEIGHT))
    x_positions = np.arange(len(cases), dtype=float)

    for i, (case, x) in enumerate(zip(cases, x_positions)):
        bd = np.asarray(case.breakdown) / scale[i]
        bottom = 0.0
        for j, label in enumerate(BD_LABELS):
            ax.bar(x, bd[j], bottom=bottom, width=0.5, color=BREAKDOWN_COLORS[j],
                   label=label if i == 0 else '_nolegend_')
            bottom += bd[j]

        base_cost = _baseline_cost(case) / scale[i]
        p5, p95 = mc_ranges[i][0] / scale[i], mc_ranges[i][1] / scale[i]
        if np.isfinite(p5) and np.isfinite(p95):
            ax.errorbar(x, base_cost, yerr=[[base_cost - p5], [p95 - base_cost]],
                        fmt='none', color='black', capsize=4, linewidth=1.5, zorder=10)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(names, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_xlim(x_positions[0] - 0.6, x_positions[-1] + 0.6)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.tick_params(axis='y', labelsize=FONT_SIZE - 1)

    handles = [Patch(facecolor=BREAKDOWN_COLORS[j], label=BD_LABELS[j])
               for j in range(len(BD_LABELS))][::-1]
    ax.legend(handles=handles, loc='upper left', fontsize=FONT_SIZE - 1, framealpha=0.9,
              title='TEA Breakdown', title_fontsize=FONT_SIZE - 1)

    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    return fig, ax


fig, ax = _plot_cost_breakdown('Cost [$/kg Product]', np.ones(len(cases)))
fig_n, ax_n = _plot_cost_breakdown('Cost [$/kg N]', n_frac, tick_step=2)

# ------------------------------------------------------------------ #
# Full Monte Carlo on ammonium sulfate (10,000 samples, summary only)
# ------------------------------------------------------------------ #
N_SAMPLES = 10000
rng = np.random.default_rng(36)
baseline = ammonium_sulfate

param_samples = {}
for attr, _, std_frac in normal_params:
    bval = getattr(baseline, attr)
    param_samples[attr] = rng.normal(bval, std_frac * abs(bval), N_SAMPLES)

for attr, _, sigma_log, _ in lognormal_params:
    bval = getattr(baseline, attr)
    param_samples[attr] = bval * np.exp(rng.normal(0, sigma_log, N_SAMPLES))

for attr, _, std_frac in sigmoid_params:
    bval = getattr(baseline, attr)
    logit_b = np.log(bval / (1.0 - bval))
    param_samples[attr] = expit(rng.normal(logit_b, std_frac / (1.0 - bval), N_SAMPLES))

max_pfe = 1.0 - baseline.her_fe - 1e-6
param_samples['product_fe'] = np.clip(param_samples['product_fe'], 1e-6, max_pfe)

# Conductivity draw applied as a relative error on top of the molarity-coupled value
cond_factor_samples = param_samples['conductivity'] / baseline.conductivity

costs_mc = np.full(N_SAMPLES, np.nan)
for i in range(N_SAMPLES):
    try:
        case = copy.deepcopy(baseline)
        for attr, *_ in all_params:
            if attr == 'conductivity':
                continue
            _apply_param(case, attr, float(param_samples[attr][i]))
        case.conductivity = case.molarity / case.k * float(cond_factor_samples[i])
        case.run_model()
        c = _baseline_cost(case)
        if np.isfinite(c) and c > 0:
            costs_mc[i] = c
    except Exception:
        pass

valid = np.isfinite(costs_mc)
print(f"MC: {valid.sum()}/{N_SAMPLES} valid samples  "
      f"| median ${np.median(costs_mc[valid]):.2f}/kg  "
      f"| 5–95 pct: ${np.percentile(costs_mc[valid], 5):.2f}–${np.percentile(costs_mc[valid], 95):.2f}/kg")


# ------------------------------------------------------------------ #
# Figures 3-5: one-at-a-time sensitivity (tornado) charts
# ------------------------------------------------------------------ #
def _oat_bounds(case, scale=1.0):
    """Cost with each parameter nudged down and up, all others at baseline.

    Returns rows of (label, lo_cost, hi_cost). `scale` multiplies every nudge
    (2.0 turns 10% steps into 20%, 0.1 into 0.2 decades, and doubles the logit step).
    """
    max_pfe = 1.0 - case.her_fe - 1e-6

    def cost_at(attr, val):
        c = copy.deepcopy(case)
        _apply_param(c, attr, val)
        c.run_model()
        return _baseline_cost(c)

    rows = []
    for attr, label, frac in normal_params:
        bval = getattr(case, attr)
        lo_val, hi_val = bval * (1 - frac * scale), bval * (1 + frac * scale)
        rows.append((label, cost_at(attr, lo_val), cost_at(attr, hi_val)))

    for attr, label, _, oat_decades in lognormal_params:
        bval = getattr(case, attr)
        lo_val, hi_val = bval * 10 ** -(oat_decades * scale), bval * 10 ** (oat_decades * scale)
        rows.append((label, cost_at(attr, lo_val), cost_at(attr, hi_val)))

    for attr, label, frac in sigmoid_params:
        bval = getattr(case, attr)
        logit_b = np.log(bval / (1.0 - bval))
        std_logit = frac * scale / (1.0 - bval)
        lo_val = float(expit(logit_b - std_logit))
        hi_val = float(expit(logit_b + std_logit))
        if attr == 'product_fe':
            lo_val, hi_val = min(lo_val, max_pfe), min(hi_val, max_pfe)
        rows.append((label, cost_at(attr, lo_val), cost_at(attr, hi_val)))

    return rows


def _plot_tornado_pct(case, title, scale=1.0):
    """Tornado chart of % change in total cost when each parameter is nudged up
    (red) or down (blue), sorted by swing with the largest on top."""
    baseline_cost = _baseline_cost(case)
    nudge = 10 * scale

    packaged = []
    for label, lo_cost, hi_cost in _oat_bounds(case, scale=scale):
        up_pct = (hi_cost - baseline_cost) / baseline_cost * 100
        down_pct = (lo_cost - baseline_cost) / baseline_cost * 100
        packaged.append(dict(label=label, up_pct=up_pct, down_pct=down_pct,
                             swing=abs(up_pct - down_pct)))

    packaged.sort(key=lambda d: d['swing'], reverse=True)
    n = len(packaged)
    y_positions = np.arange(n)[::-1]

    fig, ax = plt.subplots(figsize=(2 * WIDTH, 0.4 * n + 1.0))
    yticklabels = [None] * n
    for yi, d in zip(y_positions, packaged):
        ax.barh(yi, d['up_pct'], left=0, height=0.6, color=RED, edgecolor='black',
                linewidth=0.5, alpha=0.6, zorder=3)
        ax.barh(yi, d['down_pct'], left=0, height=0.6, color=BLUE, edgecolor='black',
                linewidth=0.5, alpha=0.6, zorder=3)
        yticklabels[yi] = d['label']

    ax.axvline(0, color='k', linewidth=1.0, zorder=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels(yticklabels, fontsize=FONT_SIZE - 1)
    ax.set_ylim(-0.7, n - 0.1)
    ax.set_xlabel('Change in Total Cost [%]', fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE - 1)

    handles = [Patch(facecolor=RED, alpha=0.6, label=f'Parameter +{nudge:.0f}%'),
               Patch(facecolor=BLUE, alpha=0.6, label=f'Parameter -{nudge:.0f}%')]
    ax.legend(handles=handles, loc='lower right', fontsize=FONT_SIZE - 2, framealpha=0.9)

    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

    print(f"\n{title} — % change in total cost (baseline = ${baseline_cost:.2f}/kg):")
    print(f"  {'Parameter':<22}{'Parameter ↑':>16}{'Parameter ↓':>16}")
    for d in packaged:
        label = d['label'].replace('\n', ' ').replace('\\$', '$')
        print(f"  {label:<22}{d['up_pct']:>+15.1f}%{d['down_pct']:>+15.1f}%")

    return fig, ax


fig_t, ax_t = _plot_tornado_pct(ammonium_nitrate, 'Ammonium Nitrate — cost sensitivity')
fig_t20, ax_t20 = _plot_tornado_pct(ammonium_nitrate, 'Ammonium Nitrate — cost sensitivity (20% nudge)', scale=2.0)
fig_t30, ax_t30 = _plot_tornado_pct(ammonium_nitrate, 'Ammonium Nitrate — cost sensitivity (30% nudge)', scale=3.0)

# Symmetric x-axes about 0%: fixed for the 10% chart, sized to the widest bar otherwise
ax_t.set_xlim(-12.5, 12.5)
ax_t.set_xticks(np.arange(-10, 12.5, 2.5))
for _ax in (ax_t20, ax_t30):
    lim = max(abs(b.get_width()) for b in _ax.patches) * 1.1
    _ax.set_xlim(-lim, lim)

# ------------------------------------------------------------------ #
# Specific energy of each product
# ------------------------------------------------------------------ #
print("\nSpecific energy (kJ/kg product):")
for name, case in zip(names, cases):
    print(f"  {name}: {float(np.atleast_1d(case.specific_energy).flat[0]):.3f}")

plt.show()
