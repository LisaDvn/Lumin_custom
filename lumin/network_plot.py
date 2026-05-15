"""
Visualisations for network-level spike-train analysis.

1. raster_plot:           spike raster per cell per recording
2. network_event_overlay: recording spike raster + detected networkevent markers
3. population_activity:   fraction of active cells per frame
#TODO: 4. synchrony_matrix:      pairwise Pearson correlation heatmap
#TODO: 5. isi_distribution:      ISI histogram (pooled or per condition)
#TODO: 6. metric_barplots:       per-condition bar + swarm for each metric

summary_file
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

#rastermap imports
import numpy as np
from scipy.stats import zscore

# preview imports
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


# colour palette helper

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


# 1. Spike raster plot
def raster_plot(S_bin, fs, cell_df=None, max_cells=200,
                output_dir=None, method='', use_rastermap=True):
    """
    Copyright (C) 2023 Howard Hughes Medical Institute Janelia Research Campus, the labs of Carsen Stringer and Marius Pachitariu.
    """
    n_neurons, n_frames = S_bin.shape
    t = np.arange(n_frames) / fs

    if use_rastermap and n_neurons >= 10:
        try:
            from rastermap import Rastermap
            from scipy.stats import zscore

            # Rastermap expects (n_neurons × n_frames), z-scored
            S_z = zscore(S_bin.astype(np.float32), axis=1)
            # replace NaNs that zscore produces for zero-variance rows
            S_z = np.nan_to_num(S_z, nan=0.0)

            model  = Rastermap(
                n_clusters     = min(100, n_neurons // 2),
                n_PCs          = min(200, n_neurons - 1),
                locality       = 0.75,   # good balance between local/global
                time_lag_window= 5,      # helps find sequences
                verbose        = False,
            ).fit(S_z)

            isort   = model.isort          # sorted neuron order
            S_plot  = S_bin[isort]
            sort_label = 'Rastermap order'

        except Exception as e:
            print(f'Rastermap failed ({e}), falling back to first-spike sort.')
            use_rastermap = False

    if not use_rastermap or n_neurons < 10:
        # fallback: sort by first spike time
        first_spike = np.argmax(S_bin > 0, axis=1)
        isort   = np.argsort(first_spike)
        S_plot  = S_bin[isort]
        sort_label = 'first spike order'

    # subsample for display if needed
    if n_neurons > max_cells:
        step   = n_neurons // max_cells
        S_plot = S_plot[::step]
        subtitle = f'(1 in {step} cells shown, {sort_label})'
    else:
        subtitle = f'({n_neurons} cells, {sort_label})'

    fig, ax = plt.subplots(figsize=(12, max(4, S_plot.shape[0] * 0.06)))
    for row_i, vec in enumerate(S_plot):
        spike_t = t[vec > 0]
        ax.scatter(spike_t, np.full_like(spike_t, row_i),
                   s=1.2, c='black', linewidths=0)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Cell (sorted)', fontsize=11)
    ax.set_title(f'Spike raster  {subtitle}\n{method}', fontsize=12)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-1, S_plot.shape[0])
    fig.tight_layout()
    return _save(fig, output_dir, f'raster_{method}.pdf')


 
def network_event_overlay_preview(
    S_bin: np.ndarray,
    fs: float,
    participation_thr: float = 0.10,
    min_peak_distance_s: float = 0.5,
    recording_label: str = '',
    method: str = '',
    output_dir: str | None = None,
) -> plt.Figure:
    """
    Preview figure shown during test runs BEFORE full analysis.
 
    Layout (4 panels, shared x-axis)
    ──────────────────────────────────
    [0]  Rastermap-sorted spike raster
         • each spike = black dot
         • detected network events = vertical red dashed lines + numbered labels
         • event windows shaded in semi-transparent red
 
    [1]  Participation heatmap strip
         • colour = fraction of cells active per frame
         • quick visual for burst width / shape
 
    [2]  Population activity trace
         • black line = fraction active
         • red dashed threshold line
         • event peaks marked with inverted triangles
         • event numbers annotated above each marker
 
    [3]  Per-cell spike-rate bar (horizontal)
         • mean firing rate per cell (Hz), sorted same as raster
         • lets you spot hyper-active or dead cells at a glance
 
    Returns the Figure (caller is responsible for plt.close).
    """
    n_neurons, n_frames = S_bin.shape
    t = np.arange(n_frames) / fs
 
    # ── detect network events ────────────────────────────────────────────────
    pop = S_bin.mean(axis=0)                           # fraction active per frame
    min_dist = max(1, int(min_peak_distance_s * fs))
    peaks, props = find_peaks(pop, height=participation_thr, distance=min_dist)
    n_events = len(peaks)
 
    # event half-width in frames (for shading) — use 0.5 s window around peak
    half_win = max(1, int(0.25 * fs))
 
    # ── sort neurons ─────────────────────────────────────────────────────────
    try:
        from lumin.Z_network_plots import _sort_neurons   # reuse existing helper
        isort, sort_label = _sort_neurons(S_bin)
    except Exception:
        first_spike = np.argmax(S_bin > 0, axis=1)
        isort = np.argsort(first_spike)
        sort_label = 'first-spike order'
 
    # subsample rows for raster display only
    n_show = min(n_neurons, 150)
    step   = max(1, n_neurons // n_show)
    show   = isort[::step]
    S_show = S_bin[show]
 
    # per-cell mean rate (Hz) in sorted order — for the side panel
    mean_rate_sorted = S_bin[isort].sum(axis=1) / (n_frames / fs)
 
    # ── figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10), facecolor='#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
 
    # 4-row grid: raster | heatmap strip | pop activity | (side panel via twin)
    gs = gridspec.GridSpec(
        3, 2,
        figure       = fig,
        height_ratios= [3, 0.35, 1.2],
        width_ratios = [5, 1],
        hspace       = 0.06,
        wspace       = 0.04,
    )
 
    ax_raster   = fig.add_subplot(gs[0, 0])
    ax_rate_bar = fig.add_subplot(gs[0, 1], sharey=None)   # horizontal rate bar
    ax_heatmap  = fig.add_subplot(gs[1, 0], sharex=ax_raster)
    ax_pop      = fig.add_subplot(gs[2, 0], sharex=ax_raster)
    ax_cbar     = fig.add_subplot(gs[1, 1])                # heatmap colourbar
    ax_cbar.axis('off')
 
    DARK_BG   = '#1a1a2e'
    TICK_CLR  = '#c8c8d8'
    SPIKE_CLR = '#e0e0f0'
    POP_CLR   = '#64b5f6'
    EVT_CLR   = '#ef5350'
    HEAT_MAP  = 'YlOrRd'
 
    for ax in (ax_raster, ax_heatmap, ax_pop, ax_rate_bar):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TICK_CLR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')
 
    # ── [0] raster ────────────────────────────────────────────────────────────
    for row_i, vec in enumerate(S_show):
        spike_t = t[vec > 0]
        if len(spike_t):
            ax_raster.scatter(
                spike_t, np.full_like(spike_t, row_i),
                s=0.8, c=SPIKE_CLR, linewidths=0, alpha=0.85, rasterized=True,
            )
 
    # shade event windows + label
    for ev_i, pk in enumerate(peaks):
        t0 = t[max(0, pk - half_win)]
        t1 = t[min(n_frames - 1, pk + half_win)]
        ax_raster.axvspan(t0, t1, color=EVT_CLR, alpha=0.12, lw=0)
        ax_raster.axvline(t[pk], color=EVT_CLR, lw=0.9, ls='--', alpha=0.7)
        ax_raster.text(
            t[pk], S_show.shape[0] + 0.5, f'#{ev_i+1}',
            ha='center', va='bottom', fontsize=6.5,
            color=EVT_CLR, fontweight='bold',
        )
 
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
 
    # ── [side] per-cell mean rate bar ─────────────────────────────────────────
    y_pos = np.arange(len(mean_rate_sorted))
    ax_rate_bar.barh(
        y_pos, mean_rate_sorted,
        height=0.8, color=POP_CLR, alpha=0.7,
    )
    ax_rate_bar.set_xlabel('Rate (Hz)', color=TICK_CLR, fontsize=7)
    ax_rate_bar.set_facecolor(DARK_BG)
    ax_rate_bar.tick_params(colors=TICK_CLR, labelsize=7)
    ax_rate_bar.tick_params(labelleft=False)
    for spine in ax_rate_bar.spines.values():
        spine.set_edgecolor('#444466')
    ax_rate_bar.set_title('Firing\nrate', color=TICK_CLR, fontsize=7, pad=4)
    # align y with raster (show subsample rows, not all sorted neurons)
    ax_rate_bar.set_ylim(-1, len(mean_rate_sorted) + 2)
 
    # ── [1] heatmap strip ─────────────────────────────────────────────────────
    # bin population activity into 2-D strip (1 row × n_frames)
    im = ax_heatmap.imshow(
        pop[np.newaxis, :],
        aspect='auto', extent=[0, t[-1], 0, 1],
        cmap=HEAT_MAP, vmin=0, vmax=max(pop.max(), participation_thr * 1.5),
        interpolation='nearest',
    )
    for pk in peaks:
        ax_heatmap.axvline(t[pk], color=EVT_CLR, lw=0.9, ls='--', alpha=0.8)
    ax_heatmap.set_yticks([])
    ax_heatmap.set_ylabel('Act.', color=TICK_CLR, fontsize=7, labelpad=2)
    ax_heatmap.tick_params(labelbottom=False)
 
    # small colourbar next to heatmap
    cbar = fig.colorbar(im, ax=ax_cbar, location='left', shrink=0.9, pad=0.05)
    cbar.ax.tick_params(colors=TICK_CLR, labelsize=6)
    cbar.set_label('Frac.\nactive', color=TICK_CLR, fontsize=6)
 
    # ── [2] population activity trace ────────────────────────────────────────
    ax_pop.plot(t, pop * 100, lw=0.9, color=POP_CLR, label='Population activity')
    ax_pop.axhline(
        participation_thr * 100, color=EVT_CLR, lw=0.9, ls='--',
        label=f'Threshold ({participation_thr*100:.0f} %)',
    )
 
    for ev_i, pk in enumerate(peaks):
        t0 = t[max(0, pk - half_win)]
        t1 = t[min(n_frames - 1, pk + half_win)]
        ax_pop.axvspan(t0, t1, color=EVT_CLR, alpha=0.10, lw=0)
        # inverted triangle at peak
        ax_pop.plot(
            t[pk], pop[pk] * 100,
            marker='v', ms=5, color=EVT_CLR, zorder=5,
        )
        ax_pop.annotate(
            f'#{ev_i+1}',
            xy    = (t[pk], pop[pk] * 100),
            xytext= (t[pk], pop[pk] * 100 + 2.5),
            ha='center', va='bottom', fontsize=6.5,
            color=EVT_CLR, fontweight='bold',
        )
 
    ax_pop.set_xlabel('Time (s)', color=TICK_CLR, fontsize=9)
    ax_pop.set_ylabel('Active cells (%)', color=TICK_CLR, fontsize=9)
    ax_pop.set_xlim(0, t[-1])
    ax_pop.set_ylim(bottom=0)
    ax_pop.legend(fontsize=8, frameon=False, labelcolor=TICK_CLR)
 
    # ── summary text box ─────────────────────────────────────────────────────
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
 
    ax_pop.text(
        0.99, 0.97, summary,
        transform=ax_pop.transAxes,
        ha='right', va='top', fontsize=8,
        color='#e0e0f0',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#2a2a4a',
                  edgecolor='#555577', alpha=0.85),
    )
 
    # ── save ─────────────────────────────────────────────────────────────────
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        from lumin.Z_network_plots import _safe_filename
        fname = f'event_preview_{_safe_filename(recording_label)}_{method}.pdf'
        fig.savefig(
            os.path.join(output_dir, fname),
            bbox_inches='tight', facecolor=fig.get_facecolor(),
        )
 
    return fig
 
 
"""