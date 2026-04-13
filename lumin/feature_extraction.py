import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

# -------------------- NORMALIZATION --------------------

def sliding_window(cell_properties_df, window_size, percentile):
    def normalize(trace):
        trace = np.array(trace)
        norm = np.zeros_like(trace)

        for i in range(len(trace)):
            start = max(0, i - window_size)
            window = trace[start:i+1]
            threshold = np.percentile(window, percentile)
            norm[i] = (trace[i] - threshold)

        return norm

    df = cell_properties_df.copy()
    df['dff'] = df['dff'].apply(normalize)
    return df


def pre_stimulation(cell_properties_df, stimulation_frame):
    df = cell_properties_df.copy()

    def normalize(trace):
        trace = np.array(trace)
        baseline = np.mean(trace[:stimulation_frame])
        return (trace - baseline) / (baseline + 1e-9)

    df['dff'] = df['dff'].apply(normalize)
    return df


# -------------------- AUC --------------------

def compute_auc(cell_properties_df, start_frame, column='AUC'):
    df = cell_properties_df.copy()

    def auc(trace):
        trace = np.array(trace)
        return np.trapz(trace[start_frame:])

    df[column] = df['dff'].apply(auc)
    return df


# -------------------- SPIKE DETECTION --------------------

def detect_spikes(cell_properties_df, threshold=2.0):
    df = cell_properties_df.copy()

    def spikes(trace):
        trace = np.array(trace)
        return np.where(trace > threshold * np.std(trace))[0].tolist()

    df['peak_location'] = df['dff'].apply(spikes)
    return df


# -------------------- FEATURE MATRIX --------------------

def build_feature_matrix(cell_properties_df, feature_cols):
    return cell_properties_df[feature_cols].select_dtypes(include=[np.number])
