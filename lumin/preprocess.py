# -------------------- IMPORTS --------------------
import numpy as np          # numerical operations (arrays, mean, etc.)
import statistics           # basic statistics (used for mean here)
import pandas as pd         # dataframe handling
from scipy.signal import savgol_filter  # for optional smoothing of deconvolved trace
import lumin.Z_deconv_oasis as oasis  # OASIS deconvolution function

# -------------------- METHOD 1: PRE-STIMULATION BASELINE --------------------
def pre_stimulation(cell_properties_df: pd.DataFrame = None,
                    stimulation_frame: int = None):

    """
    Calculate ΔF/F using a fixed baseline.
    
    Baseline (F0) = mean fluorescence BEFORE stimulation.
    
    For each cell:
    - Take raw trace
    - Compute baseline from pre-stimulation frames
    - Calculate ΔF/F = (F - F0) / F0
    """

    dff_traces_list = []   # will store ΔF/F traces for all cells
    f0_list = []           # will store baseline (F0) traces

    # Loop over each cell (row in dataframe)
    for _, cell in cell_properties_df.iterrows():

        raw_trace = cell['raw']  # fluorescence over time

        # Compute baseline as mean BEFORE stimulation
        mean_baseline = np.mean(raw_trace[:stimulation_frame])

        # Compute ΔF/F for entire trace
        dff_traces_list.append([
            (F - mean_baseline) / mean_baseline for F in raw_trace
        ])

        # Store baseline as a flat line (same value for all frames)
        f0_list.append([mean_baseline] * len(raw_trace))

    # Add results to dataframe
    cell_properties_df['dff'] = dff_traces_list
    cell_properties_df['baseline'] = f0_list

    return cell_properties_df


# -------------------- METHOD 2: SLIDING WINDOW BASELINE --------------------
def sliding_window(cell_properties_df: pd.DataFrame = None,
                   sliding_window_size: int = None,
                   percentile_threshold: int = None):

    """
    Calculate ΔF/F using a dynamic baseline (sliding window).

    Idea:
    - Baseline (F0) changes over time
    - At each timepoint:
        - Look at previous window
        - Take lower percentile (to avoid spikes)
        - Compute mean of values BELOW that threshold
    """

    dff_traces_list = []   # store ΔF/F traces
    f0_list = []           # store dynamic baseline traces

    # Loop over each cell
    for _, cell in cell_properties_df.iterrows():

        raw_trace = cell['raw']   # fluorescence signal
        f0_trace = []             # dynamic baseline per frame

        # initial window (for first few frames, use the first 'sliding_window_size' frames)
        sliding_window = raw_trace[:sliding_window_size]

        # Threshold to exclude high values (spikes)
        pctl_thresh = np.percentile(sliding_window, percentile_threshold)

        # Compute baseline as mean of LOWER values only
        f0 = statistics.mean([i for i in sliding_window if i < pctl_thresh])

        # Fill initial baseline for first window
        f0_trace.extend([f0] * sliding_window_size)

        # Compute ΔF/F for initial window
        dff = [(value - f0) / f0 for value in sliding_window]

        # sliding window for the rest of the trace
        for location, value in enumerate(raw_trace[sliding_window_size:], 
                                         start=sliding_window_size + 1):

            # Take previous window (moving window)
            sliding_window = raw_trace[location - sliding_window_size - 1 : location - 1]

            # NOTE: here threshold is hardcoded to 10 (could be a bug/inconsistency)
            pctl_thresh = np.percentile(sliding_window, 10)

            # Compute baseline again from low values
            f0 = statistics.mean([i for i in sliding_window if i < pctl_thresh])

            # Store baseline for this frame
            f0_trace.append(f0)

            # Compute ΔF/F for current value
            dff.append((value - f0) / f0)

        # Store results for this cell
        dff_traces_list.append(dff)
        f0_list.append(f0_trace)

    # Add results to dataframe
    cell_properties_df['baseline'] = f0_list
    cell_properties_df['dff'] = dff_traces_list

    return cell_properties_df

# --------------------  METHOD 3: DECONVOLUTION-BASED BASELINE --------------------
def deconvolution(cell_properties_df: pd.DataFrame = None, tau_d = None, frame_rate = None):
    """
    Calculate ΔF/F using a dynamic baseline based on OASIS deconvolution.

    Idea:
    - Deconvolve fluorescence signal to estimate underlying neural activity
    - Use deconvolved trace as a dynamic baseline for ΔF/F calculation
    """

    dff_traces_list = []   # store ΔF/F traces

    # Loop over each cell
    for _, cell in cell_properties_df.iterrows():

        raw_trace = cell['raw']   # fluorescence signal

        # Reshape trace for OASIS (expects 2D array)
        F = np.expand_dims(raw_trace, axis=0)

        # Run OASIS deconvolution to get spike estimates
        S = oasis.oasis(F, batch_size=1, tau=tau_d, fs=frame_rate)

        spike_trace = S[0]  # get the deconvolved trace for this cell

        # Compute ΔF/F using deconvolved trace as baseline
        dff = [(value - spike) / spike if spike != 0 else 0 
               for value, spike in zip(raw_trace, spike_trace)]

        dff_traces_list.append(dff)

    # Add results to dataframe
    cell_properties_df['dff'] = dff_traces_list

    return cell_properties_df