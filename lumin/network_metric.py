"""
Compute spike-train metrics at three levels:
  - per cell
  - per recording (well)
  - per condition (stimulation / cell_line)

Return CSV + PKL to /Network_activity/Tables/
"""

import os
import threading
from typing import Literal

import numpy as np
import pandas as pd
from numba import njit
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CELL_GROUP_COLS   = ("image_id", "filename", "plate_id", "cell_line", "stimulation", "condition", "label")
_RECORD_GROUP_COLS = ("image_id", "filename", "plate_id", "cell_line", "stimulation", "condition")
_COND_GROUP_COLS   = ("stimulation", "condition", "cell_line")

# Columns produced by compute_recording_metrics that condition-level aggregation reads.
# Keep this in sync with the column names written in _metrics_for_group.
_CONDITION_NUMERIC_METRICS = [
    "pct_active_neurons",
    "mean_n_events",
    "mean_frequency",
    "mean_mean_amplitude",
    "mean_mean_isi_s",
    "synchrony_index",
    "network_event_rate_hz",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _present_cols(candidates: tuple[str, ...], df: pd.DataFrame) -> list[str]:
    """Return only those column names that actually exist in *df*."""
    return [c for c in candidates if c in df.columns]


# ---------------------------------------------------------------------------
# Binarisation
# ---------------------------------------------------------------------------

def binarise(
    S: np.ndarray,
    method: Literal["CASCADE", "OASIS", "other"],
    threshold: float,
) -> np.ndarray:
    """Convert a continuous deconvolved array to a binary spike array.

    For CASCADE the supplied *threshold* is used directly.
    For all other methods a conservative floor of 1e-3 is applied so that
    near-zero floating-point noise is not counted as a spike.
    """
    thr = threshold if method == "CASCADE" else 1e-3
    return (S > thr).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-cell metrics
# ---------------------------------------------------------------------------

def compute_cell_metrics(
    S: np.ndarray,
    S_bin: np.ndarray,
    cell_properties_df: pd.DataFrame,
    fs: float,
    method: str,
) -> pd.DataFrame:
    """Compute single-cell spike-train metrics.

    Parameters
    ----------
    S:
        Continuous deconvolved signal, shape (n_neurons, n_frames).
    S_bin:
        Binary spike array, same shape as *S*.
    cell_properties_df:
        Per-cell metadata DataFrame (must have one row per neuron in the same
        order as the first axis of *S*).
    fs:
        Sampling rate in Hz.
    method:
        Deconvolution method label stored in output (e.g. ``"CASCADE"``).

    Returns
    -------
    pd.DataFrame
        One row per neuron with spike-train statistics.
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

        row: dict = {}
        for col in _CELL_GROUP_COLS:
            if col in cell_properties_df.columns:
                row[col] = cell_properties_df.iloc[i][col]

        row.update(
            cell_index    = i,
            n_events      = n_events,
            frequency_hz  = round(frequency, 4),
            mean_amplitude= round(mean_amp, 4),
            mean_isi_s    = round(mean_isi, 4) if not np.isnan(mean_isi) else np.nan,
            cv_isi        = round(cv_isi,   4) if not np.isnan(cv_isi)   else np.nan,
            is_active     = int(n_events > 0),
            deconv_method = method,
        )
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Numba-accelerated synchrony helpers
# ---------------------------------------------------------------------------

# Thread lock prevents concurrent Numba JIT compilation across threads.
_NUMBA_LOCK = threading.Lock()


@njit(cache=True, parallel=True)
def _compute_jitter_synchrony_matrix_numba(
    peak_array: np.ndarray,
    jitter_window: int,
) -> np.ndarray:  # pragma: no cover
    """Compute the pairwise jitter-window synchrony matrix for binary spike arrays."""
    n_rois = peak_array.shape[0]
    synchrony_matrix = np.zeros((n_rois, n_rois), dtype=np.float64)

    for i in range(n_rois):
        synchrony_matrix[i, i] = 1.0
        for j in range(i + 1, n_rois):
            sync_value = _jitter_window_synchrony_numba(
                peak_array[i], peak_array[j], jitter_window
            )
            synchrony_matrix[i, j] = sync_value
            synchrony_matrix[j, i] = sync_value

    return synchrony_matrix


@njit(cache=True)
def _jitter_window_synchrony_numba(
    events_i: np.ndarray,
    events_j: np.ndarray,
    jitter_window: int,
) -> float:  # pragma: no cover
    """Jitter-window synchrony for a single ROI pair.

    For each peak in ROI *i*, checks whether a peak in ROI *j* falls within
    ±*jitter_window* frames, and vice-versa.
    """
    peaks_i = np.where(events_i > 0)[0]
    peaks_j = np.where(events_j > 0)[0]
    n_peaks_i = len(peaks_i)
    n_peaks_j = len(peaks_j)

    if n_peaks_i == 0 or n_peaks_j == 0:
        return 0.0

    coincidences_i_to_j = 0
    for i in range(n_peaks_i):
        for j in range(n_peaks_j):
            if abs(peaks_j[j] - peaks_i[i]) <= jitter_window:
                coincidences_i_to_j += 1
                break

    coincidences_j_to_i = 0
    for j in range(n_peaks_j):
        for i in range(n_peaks_i):
            if abs(peaks_i[i] - peaks_j[j]) <= jitter_window:
                coincidences_j_to_i += 1
                break

    total_peaks       = n_peaks_i + n_peaks_j
    total_coincidences = coincidences_i_to_j + coincidences_j_to_i
    return total_coincidences / total_peaks if total_peaks > 0 else 0.0


# ---------------------------------------------------------------------------
# Pairwise correlation / synchrony matrix helpers
# ---------------------------------------------------------------------------

def _compute_zero_lag_corr_matrix(traces: list[np.ndarray]) -> np.ndarray | None:
    """Zero-lag Pearson correlation matrix from a list of 1-D trace arrays.

    Returns ``None`` when fewer than two traces are supplied.

    Raises
    ------
    ValueError
        If traces have unequal lengths.
    """
    if len(traces) < 2:
        return None

    lengths = [len(t) for t in traces]
    if len(set(lengths)) > 1:
        raise ValueError(f"All traces must have the same length. Got: {set(lengths)}")

    traces_array = np.vstack(traces)
    means        = traces_array.mean(axis=1, keepdims=True)
    stds         = traces_array.std(axis=1, keepdims=True, ddof=1)
    stds[stds == 0] = 1.0

    z = (traces_array - means) / stds
    norms = np.linalg.norm(z, axis=1)
    norms[norms == 0] = np.finfo(float).eps

    n_rois             = len(traces)
    correlation_matrix = np.zeros((n_rois, n_rois), dtype=float)
    np.fill_diagonal(correlation_matrix, 1.0)

    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            r0 = np.clip(np.dot(z[i], z[j]) / (norms[i] * norms[j]), -1.0, 1.0)
            correlation_matrix[i, j] = r0
            correlation_matrix[j, i] = r0

    return correlation_matrix


def _get_global_pairwise_score(pairwise_matrix: np.ndarray | None) -> float | None:
    """Median of per-row mean off-diagonal values of a square pairwise matrix.

    Returns ``None`` for degenerate inputs (fewer than 2 ROIs, non-square, empty).
    """
    if pairwise_matrix is None or pairwise_matrix.size == 0:
        return None
    n = pairwise_matrix.shape[0]
    if n < 2 or pairwise_matrix.shape[0] != pairwise_matrix.shape[1]:
        return None

    off_diag_sum = np.sum(pairwise_matrix, axis=1) - np.diag(pairwise_matrix)
    mean_per_row = off_diag_sum / (n - 1)
    return float(np.median(mean_per_row))


def _get_spike_correlations_matrix(
    spike_data_dict: dict[str, list[float]],
    *,
    jitter_window: int = 5,
) -> np.ndarray | None:
    """Pairwise spike similarity matrix via jitter-window cross-correlogram.

    Returns ``None`` when fewer than two active ROIs are present.

    Raises
    ------
    ValueError
        If spike data contains non-binary values.
    """
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


# ---------------------------------------------------------------------------
# Per-recording (well) metrics
# ---------------------------------------------------------------------------

def compute_recording_metrics(
    S_bin: np.ndarray,
    cell_metrics_df: pd.DataFrame,
    fs: float,
    method: str,
    participation_thr: float,
    min_peak_distance_s: float,
    *,
    jitter_window: int = 5,
) -> pd.DataFrame:
    """Compute recording-level (well/FOV) metrics aggregated over all cells.

    Parameters
    ----------
    S_bin:
        Binary spike array, shape (n_neurons, n_frames).
    cell_metrics_df:
        Output of :func:`compute_cell_metrics`.
    fs:
        Sampling rate in Hz.
    method:
        Deconvolution method label.
    participation_thr:
        Minimum fraction of simultaneously active cells required to count a
        frame as belonging to a network event.
    min_peak_distance_s:
        Minimum distance between network-event peaks in seconds.
    jitter_window:
        Half-width (frames) of the coincidence window for synchrony.

    Returns
    -------
    pd.DataFrame
        One row per recording with population-level statistics.
    """
    group_cols   = _present_cols(_RECORD_GROUP_COLS, cell_metrics_df)
    numeric_cols = ["n_events", "frequency_hz", "mean_amplitude", "mean_isi_s", "cv_isi"]
    rows: list[dict] = []

    def _metrics_for_group(indices: np.ndarray, grp: pd.DataFrame) -> dict:
        S_grp      = S_bin[indices]
        n_frames   = S_grp.shape[1]
        duration_s = n_frames / fs

        spike_trains = [S_grp[i] for i in range(S_grp.shape[0])]
        spike_dict   = {str(k): spike_trains[k].tolist() for k in range(len(spike_trains))}

        # ── Activity ────────────────────────────────────────────────────────
        spike_counts  = S_grp.sum(axis=1)
        silent_cells  = int(np.sum(spike_counts == 0))
        active_mask   = spike_counts > 0
        mean_frequency = (
            float(np.mean(spike_counts[active_mask] / duration_s))
            if np.any(active_mask) else np.nan
        )

        # ── ISI ─────────────────────────────────────────────────────────────
        all_isi: list[float] = []
        for vec in S_grp:
            spk = np.where(vec > 0)[0]
            if len(spk) > 1:
                all_isi.extend((np.diff(spk) / fs).tolist())
        isi_arr = np.array(all_isi)

        mean_isi = float(np.mean(isi_arr))   if isi_arr.size else np.nan
        cov_isi  = (float(np.std(isi_arr) / np.mean(isi_arr))
                    if isi_arr.size and np.mean(isi_arr) > 0 else np.nan)

        # ── Synchrony ───────────────────────────────────────────────────────
        jitter_matrix = _get_spike_correlations_matrix(spike_dict, jitter_window=jitter_window)
        synchrony     = (
            _get_global_pairwise_score(jitter_matrix)
            if jitter_matrix is not None else np.nan
        )

        # ── Network events ───────────────────────────────────────────────────
        pop      = S_grp.mean(axis=0)
        min_dist = max(1, int(min_peak_distance_s * fs))
        peaks, _ = find_peaks(pop, height=participation_thr, distance=min_dist)

        network_event_rate_hz = float(len(peaks) / duration_s)
        mean_event_size_pct   = float(pop[peaks].mean() * 100) if len(peaks) > 0 else np.nan

        # ── Isolated spikes ──────────────────────────────────────────────────
        total_spikes     = int(np.sum(S_grp))
        isolated_spikes  = int(np.sum(S_grp[:, pop < participation_thr]))
        mean_isolated    = isolated_spikes / total_spikes if total_spikes > 0 else np.nan

        # ── Assemble row ─────────────────────────────────────────────────────
        row: dict = {}
        for c in group_cols:
            if c in grp.columns:
                row[c] = grp[c].iloc[0]

        row["recording_duration_s"]  = round(duration_s, 2)
        row["sampling_rate_hz"]      = fs
        row["deconv_method"]         = method

        row["n_cells"]            = len(indices)
        row["n_active_cells"]     = int(grp["is_active"].sum())
        row["pct_active_neurons"] = round(100 * grp["is_active"].mean(), 2)
        row["silent_cells"]       = silent_cells

        # Named to match _CONDITION_NUMERIC_METRICS keys exactly
        row["mean_frequency"]     = round(mean_frequency, 4) if not np.isnan(mean_frequency) else np.nan
        row["mean_isi"]           = round(mean_isi, 4)       if not np.isnan(mean_isi)        else np.nan
        row["CoV_isi"]            = round(cov_isi, 4)        if not np.isnan(cov_isi)         else np.nan

        for c in numeric_cols:
            row[f"mean_{c}"] = round(grp[c].mean(), 4)

        row["synchrony_index"]       = round(synchrony, 4)          if not np.isnan(synchrony)           else np.nan
        row["mean_isolated"]         = round(mean_isolated, 4)       if not np.isnan(mean_isolated)        else np.nan
        row["network_event_rate_hz"] = round(network_event_rate_hz, 4)
        row["mean_event_size_pct"]   = round(mean_event_size_pct, 2) if not np.isnan(mean_event_size_pct)  else np.nan

        return row

    # ── Group iteration ──────────────────────────────────────────────────────
    label_col = next(
        (c for c in ("filename", "image_id") if c in cell_metrics_df.columns), None
    )

    if group_cols:
        groups = list(cell_metrics_df.groupby(group_cols, observed=True))
        total  = len(groups)
        for current, (keys, grp) in enumerate(groups, start=1):
            label = str(grp[label_col].iloc[0]) if label_col else str(keys)
            print(f"  Computing metrics: {label}  ({len(grp)} cells)  [{current}/{total}]")

            indices = grp["cell_index"].values
            row     = _metrics_for_group(indices, grp)

            if isinstance(keys, tuple):
                for k, v in zip(group_cols, keys):
                    row[k] = v
            else:
                row[group_cols[0]] = keys

            rows.append(row)
    else:
        print("  Computing metrics: all cells")
        rows.append(_metrics_for_group(cell_metrics_df["cell_index"].values, cell_metrics_df))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-condition metrics
# ---------------------------------------------------------------------------

def compute_condition_metrics(recording_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate recording-level metrics to condition level (mean ± SEM).

    Parameters
    ----------
    recording_metrics_df:
        Output of :func:`compute_recording_metrics`.

    Returns
    -------
    pd.DataFrame
        One row per condition with mean and SEM for each numeric metric.
        Empty DataFrame when no condition columns are present.
    """
    condition_cols = _present_cols(_COND_GROUP_COLS, recording_metrics_df)
    if not condition_cols:
        return pd.DataFrame()

    rows: list[dict] = []
    for keys, grp in recording_metrics_df.groupby(condition_cols, observed=True):
        row: dict = {}
        if isinstance(keys, tuple):
            for k, v in zip(condition_cols, keys):
                row[k] = v
        else:
            row[condition_cols[0]] = keys

        row["n_recordings"] = len(grp)
        for m in _CONDITION_NUMERIC_METRICS:
            if m in grp.columns:
                vals           = grp[m].dropna()
                row[f"{m}_mean"] = round(float(vals.mean()), 4) if len(vals)      else np.nan
                row[f"{m}_sem"]  = round(float(vals.sem()),  4) if len(vals) > 1  else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_and_save_all(
    S: np.ndarray,
    cell_properties_df: pd.DataFrame,
    project_dir: str,
    method: Literal["CASCADE", "OASIS", "other"],
    fs: float,
    *,
    participation_thr: float = 0.10,
    min_peak_distance_s: float = 0.5,
    spike_threshold: float = 0.5,
    jitter_window: int = 5,
) -> dict:
    """Compute all metrics and save results to CSV and pickle.

    Parameters
    ----------
    S:
        Continuous deconvolved array, shape (n_neurons, n_frames).
    cell_properties_df:
        Per-cell metadata with one row per neuron.
    project_dir:
        Root project directory. Outputs go to
        ``<project_dir>/Network_activity/Tables/``.
    method:
        Deconvolution method — controls binarisation threshold and output
        file naming.
    fs:
        Sampling rate in Hz.
    participation_thr:
        Fraction of active cells needed to define a network event frame.
    min_peak_distance_s:
        Minimum distance between network event peaks in seconds.
    spike_threshold:
        Spike detection threshold used by :func:`binarise` (CASCADE only).
    jitter_window:
        Half-width (frames) of the coincidence window for synchrony.

    Returns
    -------
    dict
        Keys: ``'cell'``, ``'recording'``, ``'condition'``, ``'S_bin'``.
    """
    table_dir = os.path.join(project_dir, "Network_activity", "Tables")
    os.makedirs(table_dir, exist_ok=True)

    S_bin = binarise(S, method, spike_threshold)

    cell_df = compute_cell_metrics(S, S_bin, cell_properties_df, fs, method)
    rec_df  = compute_recording_metrics(
        S_bin, cell_df, fs, method, participation_thr, min_peak_distance_s,
        jitter_window=jitter_window,
    )
    cond_df = compute_condition_metrics(rec_df)

    for name, df in [
        ("networkmetrics_per_cell",      cell_df),
        ("networkmetrics_per_well",      rec_df),
        ("networkmetrics_per_condition", cond_df),
    ]:
        if df.empty:
            continue
        base = os.path.join(table_dir, f"{name}_{method}")
        df.to_csv(f"{base}.csv", sep=";", index=False)
        df.to_pickle(f"{base}.pkl")
        print(f"Saved {name} → {table_dir}")

    return dict(cell=cell_df, recording=rec_df, condition=cond_df, S_bin=S_bin)