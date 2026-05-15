"""
Visualisations for network-level spike-train analysis.

Per-recording plots  (one file per image_id):
  raster_plot              – Rastermap-sorted (or first-spike fallback) spike raster
  population_activity_plot – fraction of active cells per frame + event markers
  network_event_overlay    – raster + population activity combined

Pooled / condition-level plots (one file across all recordings):
  synchrony_matrix_plot    – pairwise Pearson correlation heatmap
  isi_distribution_plot    – ISI histogram per condition
  metric_barplots          – per-condition bar + swarm for each metric
  summary_dashboard        – 2×3 overview grid

Entry point:
  generate_all_plots       – calls everything, saves to
                             <project_dir>/Network_activity/Plots/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.signal import find_peaks


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_palette(df: pd.DataFrame) -> dict:
    if 'stimulation' not in df.columns:
        return {}
    cats   = df['stimulation'].astype('category').cat.categories
    colors = sns.color_palette('tab10', n_colors=len(cats))
    return {c: colors[i] for i, c in enumerate(cats)}


def _save(fig: plt.Figure, output_dir: str | None, filename: str) -> plt.Figure:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        fig.savefig(path, bbox_inches='tight')
    return fig


def _recording_label(cell_df: pd.DataFrame) -> str:
    for col in ('filename', 'image_id'):
        if col in cell_df.columns:
            return str(cell_df[col].iloc[0])
    return 'recording'


def _safe_filename(label: str) -> str:
    return "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in label)


def _rastermap_sort(S_bin: np.ndarray) -> np.ndarray:
    n_neurons = S_bin.shape[0]
    if n_neurons < 10:
        raise ValueError('too few neurons for Rastermap')
    from rastermap import Rastermap
    from scipy.stats import zscore
    S_z = zscore(S_bin.astype(np.float32), axis=1)
    S_z = np.nan_to_num(S_z, nan=0.0)
    model = Rastermap(
        n_clusters      = min(100, n_neurons // 2),
        n_PCs           = min(200, n_neurons - 1),
        locality        = 0.75,
        time_lag_window = 5,
        verbose         = False,
    ).fit(S_z)
    return model.isort


def _sort_neurons(S_bin: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        isort = _rastermap_sort(S_bin)
        return isort, 'Rastermap order'
    except Exception as e:
        print(f'Rastermap unavailable ({e}), using first-spike order.')
        first_spike = np.argmax(S_bin > 0, axis=1)
        return np.argsort(first_spike), 'first-spike order'


# ── per-recording plots ───────────────────────────────────────────────────────

def raster_plot(
    S_bin: np.ndarray,
    fs: float,
    recording_label: str = '',
    max_cells: int = 200,
    output_dir: str | None = None,
    method: str = '',
) -> plt.Figure:
    n_neurons, n_frames = S_bin.shape
    t = np.arange(n_frames) / fs
    isort, sort_label = _sort_neurons(S_bin)
    S_sorted = S_bin[isort]
    if n_neurons > max_cells:
        step   = max(1, n_neurons // max_cells)
        S_plot = S_sorted[::step]
        subtitle = f'1 in {step} cells shown  |  {sort_label}'
    else:
        S_plot   = S_sorted
        subtitle = f'{n_neurons} cells  |  {sort_label}'
    fig, ax = plt.subplots(figsize=(12, max(4, S_plot.shape[0] * 0.06)))
    for row_i, vec in enumerate(S_plot):
        spike_t = t[vec > 0]
        ax.scatter(spike_t, np.full_like(spike_t, row_i), s=1.2, c='black', linewidths=0)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Cell (sorted)', fontsize=11)
    ax.set_title(f'Spike raster  |  {recording_label}\n{subtitle}  |  {method}', fontsize=11)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-1, S_plot.shape[0])
    fig.tight_layout()
    return _save(fig, output_dir, f'raster_{_safe_filename(recording_label)}_{method}.pdf')


def population_activity_plot(
    S_bin: np.ndarray,
    fs: float,
    participation_thr: float = 0.10,
    min_peak_distance_s: float = 0.5,
    recording_label: str = '',
    output_dir: str | None = None,
    method: str = '',
) -> plt.Figure:
    t   = np.arange(S_bin.shape[1]) / fs
    pop = S_bin.mean(axis=0)
    min_dist = max(1, int(min_peak_distance_s * fs))
    peaks, _ = find_peaks(pop, height=participation_thr, distance=min_dist)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, pop * 100, lw=0.8, color='steelblue', label='Population activity')
    ax.axhline(participation_thr * 100, color='tomato', lw=0.8, ls='--',
               label=f'Threshold ({participation_thr*100:.0f} %)')
    for p in peaks:
        ax.axvline(t[p], color='tomato', lw=0.6, alpha=0.6)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Active cells (%)', fontsize=11)
    ax.set_title(
        f'Population activity  |  {recording_label}  |  {len(peaks)} events  |  {method}',
        fontsize=11,
    )
    ax.legend(fontsize=9, frameon=False)
    ax.set_xlim(0, t[-1])
    fig.tight_layout()
    return _save(fig, output_dir, f'population_activity_{_safe_filename(recording_label)}_{method}.pdf')


def network_event_overlay(
    S_bin: np.ndarray,
    fs: float,
    participation_thr: float = 0.10,
    min_peak_distance_s: float = 0.5,
    recording_label: str = '',
    output_dir: str | None = None,
    method: str = '',
) -> plt.Figure:
    n_neurons, n_frames = S_bin.shape
    t   = np.arange(n_frames) / fs
    pop = S_bin.mean(axis=0)
    min_dist = max(1, int(min_peak_distance_s * fs))
    peaks, _ = find_peaks(pop, height=participation_thr, distance=min_dist)
    isort, sort_label = _sort_neurons(S_bin)
    n_show = min(n_neurons, 100)
    step   = max(1, n_neurons // n_show)
    show   = isort[::step]
    fig = plt.figure(figsize=(13, 6))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)
    ax0 = fig.add_subplot(gs[0])
    for row_i, cell_i in enumerate(show):
        spike_t = t[S_bin[cell_i] > 0]
        ax0.scatter(spike_t, np.full_like(spike_t, row_i), s=1, c='black', linewidths=0)
    for p in peaks:
        ax0.axvline(t[p], color='tomato', lw=0.7, alpha=0.5)
    ax0.set_ylabel('Cell (sorted)', fontsize=10)
    ax0.set_xlim(0, t[-1])
    ax0.tick_params(labelbottom=False)
    ax0.set_title(
        f'Network event overlay  |  {recording_label}  |  {sort_label}  |  {method}',
        fontsize=11,
    )
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.fill_between(t, pop * 100, alpha=0.4, color='steelblue')
    ax1.plot(t, pop * 100, lw=0.7, color='steelblue')
    ax1.axhline(participation_thr * 100, color='tomato', lw=0.8, ls='--')
    for p in peaks:
        ax1.axvline(t[p], color='tomato', lw=0.7, alpha=0.5)
    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.set_ylabel('Active (%)', fontsize=10)
    ax1.set_xlim(0, t[-1])
    return _save(fig, output_dir, f'network_event_overlay_{_safe_filename(recording_label)}_{method}.pdf')


# ── pooled / condition-level plots ────────────────────────────────────────────

def synchrony_matrix_plot(
    S_bin: np.ndarray,
    cell_df: pd.DataFrame | None = None,
    max_cells: int = 150,
    output_dir: str | None = None,
    method: str = '',
    recording_label: str = '',
) -> plt.Figure:
    active_idx = np.where(S_bin.sum(axis=1) > 0)[0]
    fname_suffix = f'_{_safe_filename(recording_label)}' if recording_label else ''
    if len(active_idx) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Fewer than 2 active cells', ha='center', va='center')
        return _save(fig, output_dir, f'synchrony_matrix{fname_suffix}_{method}.pdf')
    if len(active_idx) > max_cells:
        active_idx = np.sort(np.random.choice(active_idx, max_cells, replace=False))
    corr = np.corrcoef(S_bin[active_idx])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
    n_cells_str = f'{len(active_idx)} active cells'
    title = f'Pairwise synchrony  ({n_cells_str})  |  {method}'
    if recording_label:
        title += f'\n{recording_label}'
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Cell index', fontsize=11)
    ax.set_ylabel('Cell index', fontsize=11)
    fig.tight_layout()
    return _save(fig, output_dir, f'synchrony_matrix{fname_suffix}_{method}.pdf')


def isi_distribution_plot(
    S_bin: np.ndarray,
    fs: float,
    cell_df: pd.DataFrame | None = None,
    n_bins: int = 50,
    output_dir: str | None = None,
    method: str = '',
) -> plt.Figure:
    palette = _get_palette(cell_df) if cell_df is not None else {}
    fig, ax = plt.subplots(figsize=(7, 4))
    def _isis(rows):
        out = []
        for vec in rows:
            spk, _ = find_peaks(vec, height=0.5)
            if len(spk) > 1:
                out.extend((np.diff(spk) / fs).tolist())
        return np.array(out)
    if cell_df is not None and 'stimulation' in cell_df.columns and palette:
        for cond, grp in cell_df.groupby('stimulation', observed=True):
            isis = _isis(S_bin[grp['cell_index'].values])
            if len(isis):
                ax.hist(isis, bins=n_bins, alpha=0.55, label=str(cond),
                        color=palette.get(cond), density=True)
        ax.legend(fontsize=9, frameon=False)
    else:
        isis = _isis(S_bin)
        if len(isis):
            ax.hist(isis, bins=n_bins, color='steelblue', density=True, alpha=0.8)
    ax.set_xlabel('Inter-spike interval (s)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'ISI distribution  |  {method}', fontsize=12)
    fig.tight_layout()
    return _save(fig, output_dir, f'isi_distribution_{method}.pdf')


def _publication_bar_ax(
    ax,
    recording_df: pd.DataFrame,
    condition_col: str,
    conditions: list,
    palette: dict,
    col: str,
    ylabel: str,
) -> None:
    """Draw a single publication-style bar: filled bar + SEM whisker + scatter dots."""
    bar_width = 0.55

    for xi, cond in enumerate(conditions):
        vals  = recording_df.loc[recording_df[condition_col] == cond, col].dropna().values
        if len(vals) == 0:
            continue
        mean  = vals.mean()
        sem   = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        color = palette[cond]

        # filled bar up to mean
        ax.bar(xi, mean, width=bar_width, color=color, alpha=0.85,
               linewidth=0, zorder=2)

        # SEM error bar (no cap line on bar itself — drawn separately as whisker)
        ax.errorbar(xi, mean, yerr=sem, fmt='none', color='black',
                    capsize=4, capthick=1.2, elinewidth=1.2, zorder=4)

        # individual data points — jittered, filled circles with dark edge
        jitter = np.random.uniform(-0.12, 0.12, len(vals))
        ax.scatter(
            np.full(len(vals), xi) + jitter, vals,
            s=22, color='white', edgecolors=color,
            linewidths=0.8, zorder=5, alpha=0.9,
        )

    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlim(-0.6, len(conditions) - 0.4)

    # y-axis starts at 0, add small top padding
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.15)

    sns.despine(ax=ax)
    ax.tick_params(axis='both', labelsize=9)


def metric_barplots(
    recording_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    output_dir: str | None = None,
    method: str = '',
) -> plt.Figure:
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

    conditions = list(recording_df[condition_col].unique())
    palette    = dict(zip(conditions, sns.color_palette('tab10', len(conditions))))

    ncols = 3
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(3.5 * ncols, 4 * nrows),
                              facecolor='white')
    axes = np.array(axes).flatten()

    for ax, (col, label) in zip(axes, metrics.items()):
        _publication_bar_ax(ax, recording_df, condition_col,
                            conditions, palette, col, label)

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(f'Network metrics  |  {method}', fontsize=12, y=1.02)
    fig.tight_layout()
    return _save(fig, output_dir, f'metric_barplots_{method}.pdf')


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
    t   = np.arange(S_bin.shape[1]) / fs
    pop = S_bin.mean(axis=0)
    min_dist = max(1, int(min_peak_distance_s * fs))
    peaks, _ = find_peaks(pop, height=participation_thr, distance=min_dist)
    palette       = _get_palette(cell_df)
    condition_col = next(
        (c for c in ('stimulation', 'condition') if c in recording_df.columns), None
    )
    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # [0,0] population activity
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.plot(t, pop * 100, lw=0.7, color='steelblue')
    ax00.axhline(participation_thr * 100, color='tomato', lw=0.8, ls='--')
    for p in peaks:
        ax00.axvline(t[p], color='tomato', lw=0.5, alpha=0.4)
    ax00.set_xlabel('Time (s)', fontsize=9)
    ax00.set_ylabel('Active cells (%)', fontsize=9)
    ax00.set_title('Population activity (all recordings)', fontsize=10)

    # [0,1] synchrony matrix
    ax01 = fig.add_subplot(gs[0, 1])
    active_idx = np.where(S_bin.sum(axis=1) > 0)[0]
    if len(active_idx) >= 2:
        show = active_idx[:100]
        corr = np.corrcoef(S_bin[show])
        im   = ax01.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
        plt.colorbar(im, ax=ax01, shrink=0.85, label='r')
    ax01.set_title('Synchrony matrix', fontsize=10)

    # [0,2] ISI
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

    bottom_metrics = [
        ('pct_active_neurons',    '% active neurons'),
        ('network_event_rate_hz', 'Network event rate (Hz)'),
        ('synchrony_index',       'Synchrony index'),
    ]

    for col_i, (metric, label) in enumerate(bottom_metrics):
        ax = fig.add_subplot(gs[1, col_i])
        if condition_col and metric in recording_df.columns:
            conds = list(recording_df[condition_col].unique())
            _publication_bar_ax(ax, recording_df, condition_col,
                                conds, palette, metric, label)
        else:
            ax.set_ylabel(label, fontsize=9)
            sns.despine(ax=ax)
        ax.set_title(label, fontsize=10)

    fig.suptitle(f'Network activity summary  |  {method}', fontsize=13)
    return _save(fig, output_dir, f'summary_dashboard_{method}.pdf')


# ── preview figure (unchanged, kept in full) ──────────────────────────────────

def network_event_overlay_preview(
    S_bin: np.ndarray,
    fs: float,
    participation_thr: float = 0.10,
    min_peak_distance_s: float = 0.5,
    recording_label: str = '',
    method: str = '',
    output_dir: str | None = None,
) -> plt.Figure:
    n_neurons, n_frames = S_bin.shape
    t = np.arange(n_frames) / fs
    pop = S_bin.mean(axis=0)
    min_dist = max(1, int(min_peak_distance_s * fs))
    peaks, props = find_peaks(pop, height=participation_thr, distance=min_dist)
    n_events = len(peaks)
    half_win = max(1, int(0.25 * fs))
    try:
        isort, sort_label = _sort_neurons(S_bin)
    except Exception:
        first_spike = np.argmax(S_bin > 0, axis=1)
        isort = np.argsort(first_spike)
        sort_label = 'first-spike order'
    n_show = min(n_neurons, 150)
    step   = max(1, n_neurons // n_show)
    show   = isort[::step]
    S_show = S_bin[show]
    mean_rate_sorted = S_bin[isort].sum(axis=1) / (n_frames / fs)

    fig = plt.figure(figsize=(15, 10), facecolor='#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[3, 0.35, 1.2],
        width_ratios=[5, 1],
        hspace=0.06, wspace=0.04,
    )
    ax_raster   = fig.add_subplot(gs[0, 0])
    ax_rate_bar = fig.add_subplot(gs[0, 1], sharey=None)
    ax_heatmap  = fig.add_subplot(gs[1, 0], sharex=ax_raster)
    ax_pop      = fig.add_subplot(gs[2, 0], sharex=ax_raster)
    ax_cbar     = fig.add_subplot(gs[1, 1])
    ax_cbar.axis('off')

    DARK_BG  = '#1a1a2e'
    TICK_CLR = '#c8c8d8'
    SPIKE_CLR = '#e0e0f0'
    POP_CLR  = '#64b5f6'
    EVT_CLR  = '#ef5350'
    HEAT_MAP = 'YlOrRd'

    for ax in (ax_raster, ax_heatmap, ax_pop, ax_rate_bar):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TICK_CLR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    for row_i, vec in enumerate(S_show):
        spike_t = t[vec > 0]
        if len(spike_t):
            ax_raster.scatter(spike_t, np.full_like(spike_t, row_i),
                              s=0.8, c=SPIKE_CLR, linewidths=0, alpha=0.85, rasterized=True)
    for ev_i, pk in enumerate(peaks):
        t0 = t[max(0, pk - half_win)]
        t1 = t[min(n_frames - 1, pk + half_win)]
        ax_raster.axvspan(t0, t1, color=EVT_CLR, alpha=0.12, lw=0)
        ax_raster.axvline(t[pk], color=EVT_CLR, lw=0.9, ls='--', alpha=0.7)
        ax_raster.text(t[pk], S_show.shape[0] + 0.5, f'#{ev_i+1}',
                       ha='center', va='bottom', fontsize=6.5,
                       color=EVT_CLR, fontweight='bold')
    ax_raster.set_xlim(0, t[-1])
    ax_raster.set_ylim(-1, S_show.shape[0] + 2)
    ax_raster.set_ylabel('Cell (sorted)', color=TICK_CLR, fontsize=9)
    ax_raster.tick_params(labelbottom=False)
    ax_raster.set_title(
        f'Network event preview  ·  {recording_label}  ·  {sort_label}  ·  {method}\n'
        f'{n_neurons} cells  ·  {n_events} network events detected  '
        f'(thr = {participation_thr*100:.0f} %)',
        color='#e8e8ff', fontsize=10, pad=6,
    )

    y_pos = np.arange(len(mean_rate_sorted))
    ax_rate_bar.barh(y_pos, mean_rate_sorted, height=0.8, color=POP_CLR, alpha=0.7)
    ax_rate_bar.set_xlabel('Rate (Hz)', color=TICK_CLR, fontsize=7)
    ax_rate_bar.set_facecolor(DARK_BG)
    ax_rate_bar.tick_params(colors=TICK_CLR, labelsize=7)
    ax_rate_bar.tick_params(labelleft=False)
    for spine in ax_rate_bar.spines.values():
        spine.set_edgecolor('#444466')
    ax_rate_bar.set_title('Firing\nrate', color=TICK_CLR, fontsize=7, pad=4)
    ax_rate_bar.set_ylim(-1, len(mean_rate_sorted) + 2)

    im = ax_heatmap.imshow(
        pop[np.newaxis, :], aspect='auto', extent=[0, t[-1], 0, 1],
        cmap=HEAT_MAP, vmin=0, vmax=max(pop.max(), participation_thr * 1.5),
        interpolation='nearest',
    )
    for pk in peaks:
        ax_heatmap.axvline(t[pk], color=EVT_CLR, lw=0.9, ls='--', alpha=0.8)
    ax_heatmap.set_yticks([])
    ax_heatmap.set_ylabel('Act.', color=TICK_CLR, fontsize=7, labelpad=2)
    ax_heatmap.tick_params(labelbottom=False)
    cbar = fig.colorbar(im, ax=ax_cbar, location='left', shrink=0.9, pad=0.05)
    cbar.ax.tick_params(colors=TICK_CLR, labelsize=6)
    cbar.set_label('Frac.\nactive', color=TICK_CLR, fontsize=6)

    ax_pop.plot(t, pop * 100, lw=0.9, color=POP_CLR, label='Population activity')
    ax_pop.axhline(participation_thr * 100, color=EVT_CLR, lw=0.9, ls='--',
                   label=f'Threshold ({participation_thr*100:.0f} %)')
    for ev_i, pk in enumerate(peaks):
        t0 = t[max(0, pk - half_win)]
        t1 = t[min(n_frames - 1, pk + half_win)]
        ax_pop.axvspan(t0, t1, color=EVT_CLR, alpha=0.10, lw=0)
        ax_pop.plot(t[pk], pop[pk] * 100, marker='v', ms=5, color=EVT_CLR, zorder=5)
        ax_pop.annotate(f'#{ev_i+1}',
                        xy=(t[pk], pop[pk] * 100),
                        xytext=(t[pk], pop[pk] * 100 + 2.5),
                        ha='center', va='bottom', fontsize=6.5,
                        color=EVT_CLR, fontweight='bold')
    ax_pop.set_xlabel('Time (s)', color=TICK_CLR, fontsize=9)
    ax_pop.set_ylabel('Active cells (%)', color=TICK_CLR, fontsize=9)
    ax_pop.set_xlim(0, t[-1])
    ax_pop.set_ylim(bottom=0)
    ax_pop.legend(fontsize=8, frameon=False, labelcolor=TICK_CLR)

    if n_events > 0:
        event_rate = n_events / (n_frames / fs)
        mean_part  = float(pop[peaks].mean()) * 100
        summary = (
            f'Events : {n_events}\n'
            f'Rate   : {event_rate:.3f} Hz\n'
            f'Mean participation : {mean_part:.1f} %'
        )
    else:
        summary = 'No network events detected\n(lower threshold?)'
    ax_pop.text(0.99, 0.97, summary, transform=ax_pop.transAxes,
                ha='right', va='top', fontsize=8, color='#e0e0f0',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#2a2a4a',
                          edgecolor='#555577', alpha=0.85))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = f'event_preview_{_safe_filename(recording_label)}_{method}.pdf'
        fig.savefig(os.path.join(output_dir, fname),
                    bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig


# ── runner ────────────────────────────────────────────────────────────────────

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
    base_plot_dir = os.path.join(project_dir, 'Network_activity', 'Plots')
    os.makedirs(base_plot_dir, exist_ok=True)

    group_col = next(
        (c for c in ('image_id', 'filename') if c in cell_df.columns), None
    )
    if group_col:
        for rec_id, rec_cell_df in cell_df.groupby(group_col, observed=True):
            indices = rec_cell_df['cell_index'].values
            S_rec   = S_bin[indices]
            label   = _recording_label(rec_cell_df)
            rec_dir = os.path.join(base_plot_dir, _safe_filename(label))
            os.makedirs(rec_dir, exist_ok=True)
            print(f'  Plotting recording: {label}  ({len(indices)} cells)')
            figs = [
                raster_plot(S_rec, fs, recording_label=label, output_dir=rec_dir, method=method),
                population_activity_plot(S_rec, fs, participation_thr, min_peak_distance_s,
                                         recording_label=label, output_dir=rec_dir, method=method),
                network_event_overlay(S_rec, fs, participation_thr, min_peak_distance_s,
                                      recording_label=label, output_dir=rec_dir, method=method),
                synchrony_matrix_plot(S_rec, output_dir=rec_dir, method=method,
                                      recording_label=label),
            ]
            for fig in figs:
                plt.close(fig)
    else:
        figs = [
            raster_plot(S_bin, fs, output_dir=base_plot_dir, method=method),
            population_activity_plot(S_bin, fs, participation_thr, min_peak_distance_s,
                                     output_dir=base_plot_dir, method=method),
            network_event_overlay(S_bin, fs, participation_thr, min_peak_distance_s,
                                  output_dir=base_plot_dir, method=method),
            synchrony_matrix_plot(S_bin, output_dir=base_plot_dir, method=method),
        ]
        for fig in figs:
            plt.close(fig)

    pooled_figs = [
        isi_distribution_plot(S_bin, fs, cell_df, output_dir=base_plot_dir, method=method),
        metric_barplots(recording_df, condition_df, output_dir=base_plot_dir, method=method),
        summary_dashboard(S_bin, fs, cell_df, recording_df,
                          participation_thr=participation_thr,
                          min_peak_distance_s=min_peak_distance_s,
                          output_dir=base_plot_dir, method=method),
    ]
    for fig in pooled_figs:
        plt.close(fig)

    print(f'Network plots saved → {base_plot_dir}')