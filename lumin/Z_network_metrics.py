"""
Compute spike-train metrics at three levels:
  - per cell
  - per recording (image)
  - per condition (stimulation / cell_line)

Return CSV + PKL to /Network_activity/Tables/.
"""

import os
import threading

import numpy as np
import pandas as pd
from numba import njit
from scipy.signal import find_peaks

# ── thread lock for numba JIT compilation (prevents concurrent compilation) ──
_NUMBA_LOCK = threading.Lock()


# =============================================================================
# Numba-optimised helpers (ported from cali._fov_metrics)
# =============================================================================

@njit(cache=True, parallel=True)
def _compute_jitter_synchrony_matrix_numba(
    peak_array: np.ndarray, jitter_window: int
) -> np.ndarray:  # pragma: no cover
    n_rois = peak_array.shape[0]
    synchrony_matrix = np.zeros((n_rois, n_rois), dtype=np.float64)
    for i in range(n_rois):
        synchrony_matrix[i, i] = 1.0
        for j in range(i + 1, n_rois):
            events_i = peak_array[i]
            events_j = peak_array[j]
            sync_value = _jitter_window_synchrony_numba(events_i, events_j, jitter_window)
            synchrony_matrix[i, j] = sync_value
            synchrony_matrix[j, i] = sync_value
    return synchrony_matrix


@njit(cache=True)
def _jitter_window_synchrony_numba(
    events_i: np.ndarray, events_j: np.ndarray, jitter_window: int
) -> float:  # pragma: no cover
    peaks_i = np.where(events_i > 0)[0]
    peaks_j = np.where(events_j > 0)[0]
    n_peaks_i = len(peaks_i)
    n_peaks_j = len(peaks_j)
    if n_peaks_i == 0 or n_peaks_j == 0:
        return 0.0
    coincidences_i_to_j = 0
    for i in range(n_peaks_i):
        peak_i = peaks_i[i]
        for j in range(n_peaks_j):
            if abs(peaks_j[j] - peak_i) <= jitter_window:
                coincidences_i_to_j += 1
                break
    coincidences_j_to_i = 0
    for j in range(n_peaks_j):
        peak_j = peaks_j[j]
        for i in range(n_peaks_i):
            if abs(peaks_i[i] - peak_j) <= jitter_window:
                coincidences_j_to_i += 1
                break
    total_peaks = n_peaks_i + n_peaks_j
    total_coincidences = coincidences_i_to_j + coincidences_j_to_i
    return total_coincidences / total_peaks if total_peaks > 0 else 0.0




# =============================================================================
# Core metric functions (ported from cali._fov_metrics)
# =============================================================================

def _compute_zero_lag_corr_matrix(traces: list[np.ndarray]) -> np.ndarray | None:
    """Zero-lag Pearson correlation matrix on a list of 1-D trace arrays."""
    if len(traces) < 2:
        return None
    lengths = [len(t) for t in traces]
    if len(set(lengths)) > 1:
        raise ValueError(f"All traces must have same length. Got: {set(lengths)}")
    traces_array = np.vstack(traces)
    means = traces_array.mean(axis=1, keepdims=True)
    stds = traces_array.std(axis=1, keepdims=True, ddof=1)
    stds[stds == 0] = 1.0
    dff_zero_mean = (traces_array - means) / stds
    n_rois = len(traces)
    correlation_matrix = np.zeros((n_rois, n_rois), dtype=float)
    norms = np.linalg.norm(dff_zero_mean, axis=1)
    norms[norms == 0] = np.finfo(float).eps
    np.fill_diagonal(correlation_matrix, 1.0)
    for i in range(n_rois):
        x = dff_zero_mean[i]
        for j in range(i + 1, n_rois):
            y = dff_zero_mean[j]
            r0 = np.dot(x, y) / (norms[i] * norms[j])
            r0 = np.clip(r0, -1.0, 1.0)
            correlation_matrix[i, j] = r0
            correlation_matrix[j, i] = r0
    return correlation_matrix


def _get_global_pairwise_score(pairwise_matrix: np.ndarray | None) -> float | None:
    """Median of row-means of off-diagonal elements of a pairwise NxN matrix."""
    if pairwise_matrix is None or pairwise_matrix.size == 0:
        return None
    if pairwise_matrix.shape[0] < 2 or pairwise_matrix.shape[0] != pairwise_matrix.shape[1]:
        return None
    n = pairwise_matrix.shape[0]
    off_diagonal_sum = np.sum(pairwise_matrix, axis=1) - np.diag(pairwise_matrix)
    mean_per_row = off_diagonal_sum / (n - 1)
    return float(np.median(mean_per_row))



def _get_spike_correlations_matrix(
    spike_data_dict: dict[str, list[float]],
    jitter_window: int = 5,
) -> np.ndarray | None:
    """Jitter-window synchrony matrix for binary spike trains."""
    active_rois = list(spike_data_dict.keys())
    if len(active_rois) < 2:
        return None
    try:
        binary_spikes = np.array(
            [spike_data_dict[roi] for roi in active_rois], dtype=np.float32
        )
    except ValueError:
        return None
    if binary_spikes.shape[0] < 2:
        return None
    if not np.all(np.isin(binary_spikes, [0, 1])):
        raise ValueError("Spike data contains non-binary values.")
    with _NUMBA_LOCK:
        synchrony_matrix = _compute_jitter_synchrony_matrix_numba(
            binary_spikes, jitter_window
        )
    return synchrony_matrix



def binarise(S: np.ndarray, method: str, threshold: float) -> np.ndarray:
    thr = threshold if method == "CASCADE" else 1e-3
    return (S > thr).astype(np.float32)


# =============================================================================
# Cell-level metrics (unchanged)
# =============================================================================

def compute_cell_metrics(
    S: np.ndarray,
    S_bin: np.ndarray,
    cell_properties_df: pd.DataFrame,
    fs: float,
    method: str,
) -> pd.DataFrame:
    n_neurons, n_frames = S.shape
    duration_s = n_frames / fs
    rows = []
    for i in range(n_neurons):
        amp_vec = S[i]
        spike_vec = S_bin[i]
        spike_frames, _ = find_peaks(spike_vec, height=0.5)
        n_events = len(spike_frames)
        frequency = n_events / duration_s
        amplitudes = amp_vec[spike_frames] if n_events > 0 else np.array([np.nan])
        mean_amp = float(np.nanmean(amplitudes))
        if n_events > 1:
            isis = np.diff(spike_frames) / fs
            mean_isi = float(np.mean(isis))
            cv_isi = float(np.std(isis) / mean_isi) if mean_isi > 0 else np.nan
        else:
            mean_isi = np.nan
            cv_isi = np.nan
        row = dict(
            cell_index=i,
            n_events=n_events,
            frequency_hz=round(frequency, 4),
            mean_amplitude=round(mean_amp, 4),
            mean_isi_s=round(mean_isi, 4) if not np.isnan(mean_isi) else np.nan,
            cv_isi=round(cv_isi, 4) if not np.isnan(cv_isi) else np.nan,
            is_active=int(n_events > 0),
            deconv_method=method,
        )
        for col in ("image_id", "filename", "plate_id", "cell_line",
                    "stimulation", "condition", "label"):
            if col in cell_properties_df.columns:
                row[col] = cell_properties_df.iloc[i][col]
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Recording-level metrics
# — synchrony_index replaced by _get_global_pairwise_score + jitter matrix
# — network_event_rate replaced by _detect_spikes_population_bursts
# — fraction_significant_ccg_pairs added
# =============================================================================

def compute_recording_metrics(
    S_bin: np.ndarray,
    cell_metrics_df: pd.DataFrame,
    fs: float,
    method: str,
    participation_thr: float,
    min_peak_distance_s: float,
    jitter_window: int = 5,
) -> pd.DataFrame:
    """Compute per-recording metrics — mirrors the per-recording loop in
    generate_all_plots so that synchrony, burst, and CCG metrics are always
    computed on exactly the cells that belong to each recording."""

    group_cols = [c for c in ("image_id", "filename", "plate_id",
                               "cell_line", "stimulation", "condition")
                  if c in cell_metrics_df.columns]
    numeric_cols = ["n_events", "frequency_hz", "mean_amplitude", "mean_isi_s", "cv_isi"]
    rows = []

    def _metrics_for_group(indices, grp):
        # ── subset S_bin to this recording's cells only ───────────────────────
        S_grp = S_bin[indices]
        n_rec_frames = S_grp.shape[1]
        duration_s = n_rec_frames / fs                     # per-recording duration

        spike_trains = [S_grp[i] for i in range(S_grp.shape[0])]
        spike_dict   = {str(k): spike_trains[k].tolist() for k in range(len(spike_trains))}

        # synchrony: compute jitter-window synchrony matrix and global pairwise score
        jitter_matrix = _get_spike_correlations_matrix(
            spike_dict, jitter_window=jitter_window
        )
        synchrony = (
            _get_global_pairwise_score(jitter_matrix)
            if jitter_matrix is not None else np.nan
        )

        # network event rate: find peaks in population activity (fraction of active cells) above participation_thr, with min distance between peaks
        pop = S_grp.mean(axis=0)
        min_dist = max(1, int(min_peak_distance_s * fs))
        peaks, _ = find_peaks(pop, height=participation_thr, distance=min_dist)
        network_event_rate_hz = float(len(peaks) / duration_s)

        row = {}
        row["n_cells"]            = len(indices)
        row["n_active_cells"]     = int(grp["is_active"].sum())
        row["pct_active_neurons"] = round(100 * grp["is_active"].mean(), 2)
        for c in numeric_cols:
            row[f"mean_{c}"] = round(grp[c].mean(), 4)
        row["synchrony_index"]       = round(synchrony, 4) if not np.isnan(synchrony) else np.nan
        row["network_event_rate_hz"] = round(network_event_rate_hz, 4)
        row["recording_duration_s"]  = round(duration_s, 2)
        row["sampling_rate_hz"]              = fs
        row["deconv_method"]                 = method
        return row

    if group_cols:
        label_col = next((c for c in ("filename", "image_id") if c in cell_metrics_df.columns), None)
        groups = list(cell_metrics_df.groupby(group_cols, observed=True))
        total  = len(groups)
        for current, (keys, grp) in enumerate(groups, start=1):
            label = str(grp[label_col].iloc[0]) if label_col else str(keys)
            print(f"  Computing metrics: {label}  ({len(grp)} cells)  [{current}/{total}]")
            indices = grp["cell_index"].values
            row = _metrics_for_group(indices, grp)
            if isinstance(keys, tuple):
                for k, v in zip(group_cols, keys):
                    row[k] = v
            else:
                row[group_cols[0]] = keys
            rows.append(row)
    else:
        print("  Computing metrics: all cells")
        indices = cell_metrics_df["cell_index"].values
        row = _metrics_for_group(indices, cell_metrics_df)
        rows.append(row)

    return pd.DataFrame(rows)


# condition-level metrics: group by condition_cols and compute mean + sem of recording-level metrics
def compute_condition_metrics(recording_metrics_df: pd.DataFrame) -> pd.DataFrame:
    condition_cols = [c for c in ("stimulation", "condition", "cell_line")
                      if c in recording_metrics_df.columns]
    numeric_metrics = [
        "pct_active_neurons", "mean_n_events", "mean_frequency_hz",
        "mean_mean_amplitude", "mean_mean_isi_s",
        "synchrony_index", "network_event_rate_hz",
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
        row["n_recordings"] = len(grp)
        for m in numeric_metrics:
            if m in grp.columns:
                vals = grp[m].dropna()
                row[f"{m}_mean"] = round(float(vals.mean()), 4) if len(vals) else np.nan
                row[f"{m}_sem"] = round(float(vals.sem()), 4) if len(vals) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)



def compute_and_save_all(
    S: np.ndarray,
    cell_properties_df: pd.DataFrame,
    project_dir: str,
    method: str,
    fs: float,
    participation_thr: float = 0.10,
    min_peak_distance_s: float = 0.5,
    spike_threshold: float = 0.5,
    jitter_window: int = 5,
) -> dict:
    """Compute all metrics and save to CSV + pickle.

    Returns
    -------
    dict with keys: 'cell', 'recording', 'condition', 'S_bin'
    """
    table_dir = os.path.join(project_dir, "Network_activity", "Tables")
    os.makedirs(table_dir, exist_ok=True)

    S_bin = binarise(S, method, spike_threshold)

    cell_df = compute_cell_metrics(S, S_bin, cell_properties_df, fs, method)
    rec_df = compute_recording_metrics(
        S_bin, cell_df, fs, method, participation_thr, min_peak_distance_s,
        jitter_window=jitter_window,
    )
    cond_df = compute_condition_metrics(rec_df)

    for name, df in [
        ("cell_metrics", cell_df),
        ("recording_metrics", rec_df),
        ("condition_metrics", cond_df),
    ]:
        if df.empty:
            continue
        df.to_csv(os.path.join(table_dir, f"{name}_{method}.csv"), sep=";", index=False)
        df.to_pickle(os.path.join(table_dir, f"{name}_{method}.pkl"))
        print(f"Saved {name} → {table_dir}")

    return dict(cell=cell_df, recording=rec_df, condition=cond_df, S_bin=S_bin)