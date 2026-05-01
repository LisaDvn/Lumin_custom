"""
Compute spike-train metrics at three levels:
  - per cell
  - per recording (image)
  - per condition (stimulation / cell_line)

Return CSV + PKL to <project_dir>/Network_activity/Tables/.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def binarise(S: np.ndarray, method: str, threshold: float) -> np.ndarray:
    """
    Convert continuous spike-rate array to binary spike train.

    Parameters
    ----------
    S : np.ndarray  (n_neurons × n_frames)
    method : 'OASIS' | 'CASCADE'
    threshold : float
        For CASCADE: user-defined spike-rate threshold.
        For OASIS : small positive value to remove float noise.
    """
    thr = threshold if method == 'CASCADE' else 1e-3
    return (S > thr).astype(np.float32)


def synchrony_index(S_bin: np.ndarray) -> float:
    """
    Mean pairwise Pearson correlation of binary spike trains,
    computed only over active cells (≥1 spike).

    Returns NaN when fewer than 2 active cells are present.
    """
    if S_bin.shape[0] < 2:
        return np.nan
    active = S_bin[S_bin.sum(axis=1) > 0]
    if active.shape[0] < 2:
        return np.nan
    corr = np.corrcoef(active)
    idx  = np.triu_indices(corr.shape[0], k=1)
    return float(np.nanmean(corr[idx]))


def network_event_rate(
    S_bin: np.ndarray,
    fs: float,
    participation_thr: float = 0.10,
    min_peak_distance_s: float = 0.5,
) -> float:
    """
    Rate of network burst events per second.

    A network event is defined as a local peak in the population
    activity trace where ≥ `participation_thr` fraction of neurons
    fire within the same frame.

    Parameters
    ----------
    S_bin : np.ndarray  (n_neurons × n_frames)
    fs : float          Sampling rate in Hz.
    participation_thr : float
        Minimum fraction of co-active neurons to count as a network event.
    min_peak_distance_s : float
        Minimum time between successive network events (seconds).
    """
    if S_bin.shape[0] == 0:
        return np.nan
    pop = S_bin.mean(axis=0)
    min_dist = max(1, int(min_peak_distance_s * fs))
    peaks, _ = find_peaks(pop, height=participation_thr, distance=min_dist)
    duration = S_bin.shape[1] / fs
    return float(len(peaks) / duration)


def population_activity(S_bin: np.ndarray) -> np.ndarray:
    """Fraction of active neurons per frame. Shape (n_frames,)."""
    if S_bin.shape[0] == 0:
        return np.zeros(S_bin.shape[1])
    return S_bin.mean(axis=0)


def compute_cell_metrics(
    S: np.ndarray,
    S_bin: np.ndarray,
    cell_properties_df: pd.DataFrame,
    fs: float,
    method: str,
) -> pd.DataFrame:
    """
    One row per cell.

    Columns
    -------
    cell_index, n_events, frequency_hz, mean_amplitude, mean_isi_s,
    cv_isi, is_active  +  all grouping columns from cell_properties_df.
    """
    n_neurons, n_frames = S.shape
    duration_s = n_frames / fs
    rows = []

    for i in range(n_neurons):
        amp_vec   = S[i]
        spike_vec = S_bin[i]

        spike_frames, _ = find_peaks(spike_vec, height=0.5)
        n_events  = len(spike_frames)
        frequency = n_events / duration_s

        amplitudes = amp_vec[spike_frames] if n_events > 0 else np.array([np.nan])
        mean_amp   = float(np.nanmean(amplitudes))

        if n_events > 1:
            isis     = np.diff(spike_frames) / fs
            mean_isi = float(np.mean(isis))
            cv_isi   = float(np.std(isis) / mean_isi) if mean_isi > 0 else np.nan
        else:
            mean_isi = np.nan
            cv_isi   = np.nan

        row = dict(
            cell_index     = i,
            n_events       = n_events,
            frequency_hz   = round(frequency, 4),
            mean_amplitude = round(mean_amp, 4),
            mean_isi_s     = round(mean_isi, 4) if not np.isnan(mean_isi) else np.nan,
            cv_isi         = round(cv_isi,   4) if not np.isnan(cv_isi)   else np.nan,
            is_active      = int(n_events > 0),
            deconv_method  = method,
        )

        for col in ('image_id', 'filename', 'plate_id', 'cell_line',
                    'stimulation', 'condition', 'label'):
            if col in cell_properties_df.columns:
                row[col] = cell_properties_df.iloc[i][col]

        rows.append(row)

    return pd.DataFrame(rows)


def compute_recording_metrics(
    S_bin: np.ndarray,
    cell_metrics_df: pd.DataFrame,
    fs: float,
    method: str,
    participation_thr: float,
    min_peak_distance_s: float,
) -> pd.DataFrame:
    """
    One row per recording (image_id / filename grouping).
    Includes synchrony index and network event rate.
    """
    n_neurons, n_frames = S_bin.shape
    duration_s = n_frames / fs

    group_cols = [c for c in ('image_id', 'filename', 'plate_id',
                               'cell_line', 'stimulation', 'condition')
                  if c in cell_metrics_df.columns]

    numeric_cols = ['n_events', 'frequency_hz', 'mean_amplitude', 'mean_isi_s', 'cv_isi']
    rows = []

    if group_cols:
        for keys, grp in cell_metrics_df.groupby(group_cols, observed=True):
            indices = grp['cell_index'].values
            S_grp   = S_bin[indices]

            row = {}
            if isinstance(keys, tuple):
                for k, v in zip(group_cols, keys):
                    row[k] = v
            else:
                row[group_cols[0]] = keys

            row['n_cells']               = len(indices)
            row['n_active_cells']        = int(grp['is_active'].sum())
            row['pct_active_neurons']    = round(100 * grp['is_active'].mean(), 2)
            for c in numeric_cols:
                row[f'mean_{c}'] = round(grp[c].mean(), 4)
            row['synchrony_index']       = round(synchrony_index(S_grp), 4)
            row['network_event_rate_hz'] = round(
                network_event_rate(S_grp, fs, participation_thr, min_peak_distance_s), 4
            )
            row['recording_duration_s']  = round(duration_s, 2)
            row['sampling_rate_hz']      = fs
            row['deconv_method']         = method
            rows.append(row)

    else:
        row = dict(
            n_cells               = n_neurons,
            n_active_cells        = int(cell_metrics_df['is_active'].sum()),
            pct_active_neurons    = round(100 * cell_metrics_df['is_active'].mean(), 2),
            synchrony_index       = round(synchrony_index(S_bin), 4),
            network_event_rate_hz = round(
                network_event_rate(S_bin, fs, participation_thr, min_peak_distance_s), 4
            ),
            recording_duration_s  = round(duration_s, 2),
            sampling_rate_hz      = fs,
            deconv_method         = method,
        )
        for c in numeric_cols:
            row[f'mean_{c}'] = round(cell_metrics_df[c].mean(), 4)
        rows.append(row)

    return pd.DataFrame(rows)


def compute_condition_metrics(recording_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per condition (stimulation × cell_line).
    Reports mean ± SEM across recordings.
    """
    condition_cols = [c for c in ('stimulation', 'condition', 'cell_line')
                      if c in recording_metrics_df.columns]
    numeric_metrics = [
        'pct_active_neurons', 'mean_n_events', 'mean_frequency_hz',
        'mean_mean_amplitude', 'mean_mean_isi_s',
        'synchrony_index', 'network_event_rate_hz',
    ]

    if not condition_cols:
        return pd.DataFrame()

    rows = []
    for keys, grp in recording_metrics_df.groupby(condition_cols, observed=True):
        row = {}
        if isinstance(keys, tuple):
            for k, v in zip(condition_cols, keys):
                row[k] = v
        else:
            row[condition_cols[0]] = keys

        row['n_recordings'] = len(grp)
        for m in numeric_metrics:
            if m in grp.columns:
                vals = grp[m].dropna()
                row[f'{m}_mean'] = round(float(vals.mean()), 4) if len(vals) else np.nan
                row[f'{m}_sem']  = round(float(vals.sem()),  4) if len(vals) > 1 else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# main function to compute all metrics and save to CSV + PKL
def compute_and_save_all(
    S: np.ndarray,
    cell_properties_df: pd.DataFrame,
    project_dir: str,
    method: str,
    fs: float,
    participation_thr: float = 0.10,
    min_peak_distance_s: float = 0.5,
    spike_threshold: float = 0.5,
) -> dict:
    """
    Compute all metrics and save to CSV + pickle.

    Returns
    -------
    dict with keys: 'cell', 'recording', 'condition', 'S_bin'
    """
    table_dir = os.path.join(project_dir, 'Network_activity', 'Tables')
    os.makedirs(table_dir, exist_ok=True)

    S_bin = binarise(S, method, spike_threshold)

    cell_df = compute_cell_metrics(S, S_bin, cell_properties_df, fs, method)
    rec_df  = compute_recording_metrics(
        S_bin, cell_df, fs, method, participation_thr, min_peak_distance_s
    )
    cond_df = compute_condition_metrics(rec_df)

    for name, df in [
        ('cell_metrics',      cell_df),
        ('recording_metrics', rec_df),
        ('condition_metrics', cond_df),
    ]:
        if df.empty:
            continue
        df.to_csv(   os.path.join(table_dir, f'{name}_{method}.csv'), sep=';', index=False)
        df.to_pickle(os.path.join(table_dir, f'{name}_{method}.pkl'))
        print(f'Saved {name} → {table_dir}')

    return dict(cell=cell_df, recording=rec_df, condition=cond_df, S_bin=S_bin)