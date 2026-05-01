"""
lumin/network_plots.py

Visualisations for network-level spike-train analysis.

All functions return a matplotlib Figure and save a PDF to
<output_dir> when one is supplied.

Public functions
----------------
raster_plot          – spike raster per cell
population_activity  – fraction of active cells per frame
synchrony_matrix     – pairwise Pearson correlation heatmap
isi_distribution     – ISI histogram (pooled or per condition)
network_event_overlay– population trace + detected event markers
metric_barplots      – per-condition bar + swarm for each metric
summary_dashboard    – 2×3 grid combining key plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.signal import find_peaks


# ── colour helpers ────────────────────────────────────────────────────────────

def _get_palette(cell_df: pd.DataFrame) -> dict:
    """Build a condition → hex-colour palette from the cell metrics table."""
    if 'stimulation' not in cell_df.columns:
        return {}
    cats = cell_df['stimulation'].astype('category').cat.categories
    colors = sns.color_palette('tab10', n_colors=len(cats))
    return {c: colors[i] for i, c in enumerate(cats)}


def _save(fig: plt.Figure, output_dir: str | None, filename: str) -> plt.Figure:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    return fig


# ── 1. Raster plot ────────────────────────────────────────────────────────────

def raster_plot(
    S_bin: np.ndarray,
    fs: float,
    cell_df: pd.DataFrame | None = None,
    max_cells: int = 200,
    output_dir: str | None = None,
    method: str = '',
) -> plt.Figure:
    """
    Spike raster: one row per cell, x-axis in seconds.

    Cells are sorted by first spike time.  If > max_cells neurons,
    a random subsample is drawn for readability.
    """
    n_neurons, n_frames = S_bin.shape
    t = np.arange(n_frames) / fs

    # subsample if needed
    if n_neurons > max_cells:
        idx = np.sort(np.random.choice(n_neurons, max_cells, replace=False))
        S_plot = S_bin[idx]
        subtitle = f'(random subsample of {max_cells}/{n_neurons} cells)'
    else:
        idx    = np.arange(n_neurons)
        S_plot = S_bin
        subtitle = f'({n_neurons} cells)'

    # sort by first spike
    first_spike = np.argmax(S_plot > 0, axis=1)
    order = np.argsort(first_spike)
    S_plot = S_plot[order]

    fig, ax = plt.subplots(figsize=(12, max(4, len(order) * 0.06)))
    for row_i, vec in enumerate(S_plot):
        spike_t = t[vec > 0]
        ax.scatter(spike_t, np.full_like(spike_t, row_i),
                   s=1.2, c='black', linewidths=0)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Cell', fontsize=11)
    ax.set_title(f'Spike raster  {subtitle}\n{method}', fontsize=12)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-1, len(order))
    fig.tight_layout()
    return _save(fig, output_dir, f'raster_{method}.pdf')


# ── 2. Population activity trace ──────────────────────────────────────────────

def population_activity_plot(
    S_bin: np.ndarray,
    fs: float,
    participation_thr: float = 0.10,
    min_peak_distance_s: float = 0.5,
    output_dir: str | None = None,
    method: str = '',
) -> plt.Figure:
    """
    Fraction of co-active neurons per frame, with detected network
    events marked as vertical dashed lines.
    """
    t   = np.arange(S_bin.shape[1]) / fs
    pop = S_bin.mean(axis=0)

    min_dist = max(1, int(min_peak_distance_s * fs))
    peaks, _ = find_peaks(pop, height=participation_thr, distance=min_dist)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, pop * 100, lw=0.8, color='steelblue', label='Population activity')
    ax.axhline(participation_thr * 100, color='tomato', lw=0.8,
               ls='--', label=f'Threshold ({participation_thr*100:.0f}%)')
    for p in peaks:
        ax.axvline(t[p], color='tomato', lw=0.6, alpha=0.6)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Active cells (%)', fontsize=11)
    ax.set_title(f'Population activity  |  {len(peaks)} network events  |  {method}', fontsize=12)
    ax.legend(fontsize=9, frameon=False)
    ax.set_xlim(0, t[-1])
    fig.tight_layout()
    return _save(fig, output_dir, f'population_activity_{method}.pdf')


# ── 3. Synchrony matrix ───────────────────────────────────────────────────────

def synchrony_matrix_plot(
    S_bin: np.ndarray,
    cell_df: pd.DataFrame | None = None,
    max_cells: int = 150,
    output_dir: str | None = None,
    method: str = '',
) -> plt.Figure:
    """
    Pairwise Pearson correlation heatmap between active cells.
    Cells are clustered by hierarchical linkage.
    """
    active_idx = np.where(S_bin.sum(axis=1) > 0)[0]
    if len(active_idx) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Fewer than 2 active cells', ha='center', va='center')
        return _save(fig, output_dir, f'synchrony_matrix_{method}.pdf')

    # subsample
    if len(active_idx) > max_cells:
        active_idx = np.sort(np.random.choice(active_idx, max_cells, replace=False))

    corr = np.corrcoef(S_bin[active_idx])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
    ax.set_title(f'Pairwise synchrony  ({len(active_idx)} active cells)  |  {method}', fontsize=12)
    ax.set_xlabel('Cell index', fontsize=11)
    ax.set_ylabel('Cell index', fontsize=11)
    fig.tight_layout()
    return _save(fig, output_dir, f'synchrony_matrix_{method}.pdf')


# ── 4. ISI distribution ───────────────────────────────────────────────────────

def isi_distribution_plot(
    S_bin: np.ndarray,
    fs: float,
    cell_df: pd.DataFrame | None = None,
    n_bins: int = 50,
    output_dir: str | None = None,
    method: str = '',
) -> plt.Figure:
    """
    Pooled inter-spike interval histogram in seconds.
    If stimulation column present, overlaid per condition.
    """
    palette = _get_palette(cell_df) if cell_df is not None else {}

    fig, ax = plt.subplots(figsize=(7, 4))

    def _isis_for_rows(rows):
        isis = []
        for vec in rows:
            spk, _ = find_peaks(vec, height=0.5)
            if len(spk) > 1:
                isis.extend((np.diff(spk) / fs).tolist())
        return np.array(isis)

    if cell_df is not None and 'stimulation' in cell_df.columns and palette:
        for cond, grp in cell_df.groupby('stimulation', observed=True):
            idx  = grp['cell_index'].values
            isis = _isis_for_rows(S_bin[idx])
            if len(isis):
                ax.hist(isis, bins=n_bins, alpha=0.55, label=str(cond),
                        color=palette.get(cond), density=True)
    else:
        isis = _isis_for_rows(S_bin)
        if len(isis):
            ax.hist(isis, bins=n_bins, color='steelblue', density=True, alpha=0.8)

    ax.set_xlabel('Inter-spike interval (s)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'ISI distribution  |  {method}', fontsize=12)
    if palette:
        ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    return _save(fig, output_dir, f'isi_distribution_{method}.pdf')


# ── 5. Network event overlay ──────────────────────────────────────────────────

def network_event_overlay(
    S_bin: np.ndarray,
    fs: float,
    participation_thr: float = 0.10,
    min_peak_distance_s: float = 0.5,
    output_dir: str | None = None,
    method: str = '',
) -> plt.Figure:
    """
    Top panel: raster (up to 100 cells).
    Bottom panel: population activity + network event markers.
    """
    n_neurons, n_frames = S_bin.shape
    t   = np.arange(n_frames) / fs
    pop = S_bin.mean(axis=0)

    min_dist = max(1, int(min_peak_distance_s * fs))
    peaks, _ = find_peaks(pop, height=participation_thr, distance=min_dist)

    # subsample raster
    n_show = min(n_neurons, 100)
    idx    = np.sort(np.random.choice(n_neurons, n_show, replace=False)) \
             if n_neurons > n_show else np.arange(n_neurons)

    fig = plt.figure(figsize=(13, 6))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)

    # — raster —
    ax0 = fig.add_subplot(gs[0])
    for row_i, cell_i in enumerate(idx):
        spike_t = t[S_bin[cell_i] > 0]
        ax0.scatter(spike_t, np.full_like(spike_t, row_i),
                    s=1, c='black', linewidths=0)
    for p in peaks:
        ax0.axvline(t[p], color='tomato', lw=0.7, alpha=0.5)
    ax0.set_ylabel('Cell', fontsize=10)
    ax0.set_xlim(0, t[-1])
    ax0.tick_params(labelbottom=False)
    ax0.set_title(f'Network event overlay  |  {method}', fontsize=12)

    # — population activity —
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.fill_between(t, pop * 100, alpha=0.4, color='steelblue')
    ax1.plot(t, pop * 100, lw=0.7, color='steelblue')
    ax1.axhline(participation_thr * 100, color='tomato', lw=0.8, ls='--')
    for p in peaks:
        ax1.axvline(t[p], color='tomato', lw=0.7, alpha=0.5)
    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.set_ylabel('Active (%)', fontsize=10)
    ax1.set_xlim(0, t[-1])

    return _save(fig, output_dir, f'network_event_overlay_{method}.pdf')


# ── 6. Per-condition metric bar plots ─────────────────────────────────────────

def metric_barplots(
    recording_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    output_dir: str | None = None,
    method: str = '',
) -> plt.Figure:
    """
    One panel per metric, bar = mean across recordings,
    individual dots = recording-level values.
    """
    metrics = {
        'pct_active_neurons':    '% active neurons',
        'mean_frequency_hz':     'Mean frequency (Hz)',
        'mean_mean_amplitude':   'Mean spike amplitude',
        'mean_mean_isi_s':       'Mean ISI (s)',
        'synchrony_index':       'Synchrony index',
        'network_event_rate_hz': 'Network event rate (Hz)',
    }
    metrics = {k: v for k, v in metrics.items() if k in recording_df.columns}
    if not metrics:
        return plt.figure()

    condition_col = next(
        (c for c in ('stimulation', 'condition') if c in recording_df.columns), None
    )
    if condition_col is None:
        return plt.figure()

    conditions = recording_df[condition_col].unique()
    palette    = dict(zip(conditions, sns.color_palette('tab10', len(conditions))))

    n = len(metrics)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, (col, label) in zip(axes, metrics.items()):
        means = recording_df.groupby(condition_col, observed=True)[col].mean()
        sems  = recording_df.groupby(condition_col, observed=True)[col].sem()

        x     = np.arange(len(conditions))
        colors = [palette[c] for c in conditions]

        ax.bar(x, [means.get(c, np.nan) for c in conditions],
               yerr=[sems.get(c, 0) for c in conditions],
               color=colors, alpha=0.7, capsize=4, error_kw={'lw': 1.2})

        # individual recording dots
        for xi, cond in enumerate(conditions):
            vals = recording_df.loc[recording_df[condition_col] == cond, col].dropna()
            ax.scatter(np.full(len(vals), xi) + np.random.uniform(-0.12, 0.12, len(vals)),
                       vals, color=palette[cond], edgecolors='k',
                       linewidths=0.5, s=30, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11)
        sns.despine(ax=ax)

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(f'Network metrics per condition  |  {method}', fontsize=13, y=1.01)
    fig.tight_layout()
    return _save(fig, output_dir, f'metric_barplots_{method}.pdf')


# ── 7. Summary dashboard ──────────────────────────────────────────────────────

def summary_dashboard(
    S_bin: np.ndarray,
    fs: float,
    cell_df: pd.DataFrame,
    recording_df: pd.DataFrame,
    participation_thr: float = 0.10,
    min_peak_distance_s: float = 0.5,
    output_dir: str | None = None,
    method: str = '',
) -> plt.Figure:
    """
    2 × 3 grid combining:
      [0,0] Population activity
      [0,1] Synchrony matrix
      [0,2] ISI distribution
      [1,0] % active neurons per condition
      [1,1] Network event rate per condition
      [1,2] Synchrony index per condition
    """
    t   = np.arange(S_bin.shape[1]) / fs
    pop = S_bin.mean(axis=0)
    min_dist = max(1, int(min_peak_distance_s * fs))
    peaks, _ = find_peaks(pop, height=participation_thr, distance=min_dist)

    palette      = _get_palette(cell_df)
    condition_col = next(
        (c for c in ('stimulation', 'condition') if c in recording_df.columns), None
    )

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # — [0,0] population activity —
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.plot(t, pop * 100, lw=0.7, color='steelblue')
    ax00.axhline(participation_thr * 100, color='tomato', lw=0.8, ls='--')
    for p in peaks:
        ax00.axvline(t[p], color='tomato', lw=0.5, alpha=0.4)
    ax00.set_xlabel('Time (s)', fontsize=9)
    ax00.set_ylabel('Active cells (%)', fontsize=9)
    ax00.set_title('Population activity', fontsize=10)

    # — [0,1] synchrony matrix —
    ax01 = fig.add_subplot(gs[0, 1])
    active_idx = np.where(S_bin.sum(axis=1) > 0)[0]
    if len(active_idx) >= 2:
        show = active_idx[:100]
        corr = np.corrcoef(S_bin[show])
        im   = ax01.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
        plt.colorbar(im, ax=ax01, shrink=0.85, label='r')
    ax01.set_title('Synchrony matrix', fontsize=10)

    # — [0,2] ISI distribution —
    ax02 = fig.add_subplot(gs[0, 2])
    all_isis = []
    for vec in S_bin:
        spk, _ = find_peaks(vec, height=0.5)
        if len(spk) > 1:
            all_isis.extend((np.diff(spk) / fs).tolist())
    if all_isis:
        ax02.hist(all_isis, bins=40, color='steelblue', density=True, alpha=0.8)
    ax02.set_xlabel('ISI (s)', fontsize=9)
    ax02.set_ylabel('Density', fontsize=9)
    ax02.set_title('ISI distribution', fontsize=10)

    # — bottom row: bar plots per condition —
    bottom_metrics = [
        ('pct_active_neurons',    '% active neurons'),
        ('network_event_rate_hz', 'Network event rate (Hz)'),
        ('synchrony_index',       'Synchrony index'),
    ]

    for col_i, (metric, label) in enumerate(bottom_metrics):
        ax = fig.add_subplot(gs[1, col_i])
        if condition_col and metric in recording_df.columns:
            conditions = recording_df[condition_col].unique()
            x = np.arange(len(conditions))
            for xi, cond in enumerate(conditions):
                vals   = recording_df.loc[recording_df[condition_col] == cond, metric].dropna()
                color  = palette.get(cond, 'steelblue')
                ax.bar(xi, vals.mean(), color=color, alpha=0.7,
                       yerr=vals.sem() if len(vals) > 1 else 0, capsize=3)
                ax.scatter(
                    np.full(len(vals), xi) + np.random.uniform(-0.1, 0.1, len(vals)),
                    vals, color=color, edgecolors='k', linewidths=0.5, s=20, zorder=3
                )
            ax.set_xticks(x)
            ax.set_xticklabels(conditions, rotation=25, ha='right', fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10)
        sns.despine(ax=ax)

    fig.suptitle(f'Network activity summary  |  {method}', fontsize=13)
    return _save(fig, output_dir, f'summary_dashboard_{method}.pdf')


# ── main entry point ──────────────────────────────────────────────────────────

def generate_all_plots(
    S_bin: np.ndarray,
    fs: float,
    cell_df: pd.DataFrame,
    recording_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    project_dir: str,
    method: str,
    participation_thr: float = 0.10,
    min_peak_distance_s: float = 0.5,
) -> None:
    """Generate and save all network visualisations."""
    plot_dir = os.path.join(project_dir, 'Network_activity', 'Plots')
    os.makedirs(plot_dir, exist_ok=True)

    figs = [
        raster_plot(S_bin, fs, cell_df, output_dir=plot_dir, method=method),
        population_activity_plot(S_bin, fs, participation_thr, min_peak_distance_s,
                                 output_dir=plot_dir, method=method),
        synchrony_matrix_plot(S_bin, cell_df, output_dir=plot_dir, method=method),
        isi_distribution_plot(S_bin, fs, cell_df, output_dir=plot_dir, method=method),
        network_event_overlay(S_bin, fs, participation_thr, min_peak_distance_s,
                              output_dir=plot_dir, method=method),
        metric_barplots(recording_df, condition_df, output_dir=plot_dir, method=method),
        summary_dashboard(S_bin, fs, cell_df, recording_df,
                          participation_thr, min_peak_distance_s,
                          output_dir=plot_dir, method=method),
    ]

    for fig in figs:
        plt.close(fig)

    print(f'Network plots saved → {plot_dir}')