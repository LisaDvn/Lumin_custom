# Parameter guide

---

## Segmentation parameters

---

### Segmentation model
**What it does:** Selects the Cellpose model used to detect and outline individual cells (ROIs) in the mean fluorescence image. The model determines how cell boundaries are drawn before signal extraction begins.

| Model | Best for |
|-------|---------|
| **cyto3** | General-purpose cytoplasm segmentation. Works well for most neuronal cell cultures with visible cell bodies. |
| **cyto2** | Older cytoplasm model. Use if cyto3 over-segments your data. |
| **nuclei** | Nuclear staining. Use when imaging a nuclear marker rather than cytoplasmic GCaMP. |

**Default:** `cyto3` — recommended as the starting point for most calcium imaging datasets.

---

### Cell diameter (px)
**What it does:** The expected diameter of a single cell in pixels. Cellpose uses this to set the scale of the segmentation. Getting this right is the single most important parameter for good segmentation results.

**How to set it:** Measure a representative cell in your image using the Napari measurement tool. Over- or underestimating by more than ~30% will cause missed cells or merged ROIs.

* Set too small → large cells are split into multiple ROIs
* Set too large → nearby cells are merged into a single ROI

**Default:** `30 px` — a reasonable starting point for 10× or 20× objectives with standard magnification. Always verify on your data.

---

### Cell probability threshold
**What it does:** Minimum predicted probability that a pixel belongs to a cell before it is included in an ROI mask. Raises or lowers the overall sensitivity of detection.

* Lower → larger ROIs, more background included
* Higher → smaller, tighter ROIs, some dim cells may be missed

**Default:** `0.0` — the Cellpose default. For noisy or low-contrast recordings, try values between `−1.0` and `−2.0` to recover more cells.

---

### Flow threshold
**What it does:** Controls how strictly Cellpose enforces its internal cell boundary predictions. Lower values accept weaker boundaries and detect more cells; higher values are more conservative and reject ambiguous detections.

**Rule of thumb:** Increase if you see many spurious small detections; decrease if cells are being missed.

**Default:** `0.4` — the Cellpose default. Adjust in small steps (±0.1).

---

## Event detection parameters

---

### Smoothing
**What it does:** Applies a Savitzky–Golay filter (window = 3 frames, polynomial order = 1) to the ΔF/F₀ trace before peak detection. This reduces high-frequency noise and can improve peak detection on noisy recordings, at the cost of slightly reduced temporal precision.

**When to use:** Enable for low SNR recordings or high frame-rate data where noise causes peak splitting. Disable when temporal precision of spike timing matters (e.g. before OASIS or CASCADE deconvolution, which have their own smoothing).

---

### Prominence threshold
**What it does:** The minimum prominence of a peak to be counted as a calcium event. Prominence measures how much a peak stands out from the surrounding signal baseline — it is the height of the peak above the highest trough between it and any higher neighbouring peak. This makes it more robust than a simple height threshold because it accounts for the local signal context.

* Set too low → noise fluctuations and small artefacts are counted as events
* Set too high → real but small calcium transients are missed

**How to set it:** Inspect the ΔF/F₀ traces for a representative cell and identify the smallest real transient you want to detect, set the threshold just below that value. Or increase gradually while inspecting the cellwise trace plots.

---

### Amplitude/width ratio
**What it does:** A quality control filter that removes broad, slow artefacts (e.g. baseline drift bumps, photobleaching steps) that pass the prominence threshold. For each detected peak, the ratio of its amplitude to its width is computed; peaks below the threshold are classified as low quality and excluded.

* High ratio → sharp, fast transient (typical calcium spike)
* Low ratio → broad, slow event (likely artefact or drift)

**How to set it:** Set to `0` to disable filtering entirely. Increase gradually while inspecting the cellwise trace plots.

---

## OASIS parameters

### Indicator decay τ (s)
**What it does:** The indicator decay constant (τ) describes how quickly fluorescence returns to baseline after a calcium transient. During OASIS deconvolution, each inferred spike is modelled as an exponentially decaying signal: `F(t) = exp(-t/τ)`. 

Accurate estimation of τ is critical for reliable spike inference.

* If τ is set too low, slow calcium events may be split into multiple spikes
* If τ is set too high, nearby spikes may merge together or false events may appear

**How to set it:** Use literature values or manufacturer specifications for the calcium indicator used in your experiment. The optimal value may also depend on imaging speed, cell type, temperature, and expression level.

The following commonly used defaults were summarised from published datasets (Chen et al., 2013; Dana et al., 2019; Ohkura et al., 2012; Zhang et al., 2023):

| Indicator | Typical τ |
|-----------|-----------|
| GCaMP6f | 0.4 – 0.6 s |
| GCaMP6s | 1.5 – 2.5 s |
| GCaMP7f | 0.2 – 0.4 s |
| jGCaMP8f | 0.1 – 0.2 s |
| jGCaMP8s | 0.3 – 0.5 s |
| RCaMP1 | 1.0 – 2.0 s |

!!! note
    In vitro vs in vivo: Temperature, intracellular calcium buffering, and indicator expression level all affect decay kinetics. 
    Published values are usually from in vivo mouse cortex at 37°C. For in vitro cell culture preparations, decay can be somewhat slower. 
    It is worth empirically checking τ by looking at isolated single-transient events in your test recordings and measuring how long the tail lasts.

---

### Imaging interval (s)
**What it does:** The time between consecutive frames. Used to convert τ from seconds to frames (`fs = 1 / interval`), and to scale all time axes in plots and metric outputs (e.g. spike frequency in Hz, ISI in seconds).

**How to set it:** Read from your microscope acquisition software. Common values: 0.033 s (30 Hz), 0.1 s (10 Hz), 0.133 s (7.5 Hz).

---

### Baseline method
**What it does:** Before spike inference, OASIS subtracts a slowly-varying baseline from the raw fluorescence to remove drift from photobleaching, focus changes, and neuropil contamination.

| Method | How it works | When to use |
|--------|-------------|-------------|
| **maximin** | Smooths trace with Gaussian → rolling minimum → rolling maximum. Tracks slow drift while preserving transient peaks. | Default. Works well for most recordings. |
| **constant** | Subtracts the global minimum of the entire trace. | Only for very short, stable recordings with no drift. |
| **prctile** | Subtracts the 8th percentile of the trace. | When the recording is too short for a meaningful rolling window. |

**Default:** `maximin` — the same default used by Suite2p.

---

### Baseline window (s)
**What it does:** The rolling window size for the maximin filter. Determines how quickly the estimated baseline is allowed to change over time. A window that is too short will follow real calcium transients and over-subtract them; a window that is too long will not track genuine slow drift.

**Rule of thumb:** Should be at least 5–10× the duration of your longest expected calcium transient, and no more than ~20% of the total recording duration.

**Default:** 60 s — inherited from Suite2p defaults, assumes recordings of several minutes. Adjust downward for short recordings.

---

### Baseline σ (frames)
**What it does:** The standard deviation of the Gaussian smoothing kernel applied to the raw trace before the rolling min/max filters. Smoothing removes high-frequency noise so the baseline estimate tracks only slow trends.

**Rule of thumb:** Should be roughly 1–3× your expected spike width in frames. At 30 Hz with τ ≈ 1 s, a transient lasts ~30 frames, so σ = 10 frames is reasonable.

**Default:** 10 frames — reasonable for 30 Hz recordings with medium-speed indicators. Reduce for faster frame rates or faster indicators; increase for very noisy recordings.

---

## CASCADE parameters

### Model
**What it does:** Selects the pre-trained neural network. Each model was trained on ground truth data recorded at a specific frame rate with a specific amount of temporal smoothing of the spike labels.

**How to set it:** Match the model frame rate to your imaging interval. The smoothing parameter controls the temporal precision of spike localisation — 200 ms smoothing gives firing rate estimates, 50 ms causal smoothing gives sharper spike timing.

| Model name | Frame rate | Smoothing | Use when |
|------------|-----------|-----------|----------|
| Global_EXC_7.5Hz_smoothing200ms | 7.5 Hz | 200 ms | Slow imaging, firing rate focus |
| Global_EXC_30Hz_smoothing200ms | 30 Hz | 200 ms | Fast imaging, firing rate focus |
| Global_EXC_30Hz_smoothing50ms_causal | 30 Hz | 50 ms | Fast imaging, precise spike timing |
| Global_EXC_7.5Hz_smoothing50ms_causal | 7.5 Hz | 50 ms | Slow imaging, precise spike timing |

**Default:** `Global_EXC_7.5Hz_smoothing200ms` — conservative choice. Always verify against your actual frame rate.

---

### Imaging interval (s)
Same meaning as for OASIS. See above.

---

### Spike threshold
**What it does:** CASCADE outputs a continuous spike-rate estimate in spikes/frame for every time point. This threshold binarises it: frames where the output exceeds the threshold are counted as spike events.

**How to set it:** Lower values detect more spikes but increase false positives. Higher values are more conservative. Inspect the spike train output on a test recording and adjust until the detected events match the visible calcium transients.

**Default:** 0.5 spikes/frame — a starting point from the CASCADE paper, but should be verified on your data.

---

## Network analysis parameters

### Network event threshold
**What it does:** At each frame, the population activity is the fraction of cells with a spike. A "network event" (population burst) is a local peak in this trace that exceeds this threshold. Used to compute network event rate.

**How to set it:** Depends on how synchronous your network is. A very synchronous culture might show 50–80% co-activation during bursts; sparse spontaneous activity might only reach 10–20%. Set lower to detect smaller coordinated events, higher to detect only large bursts.

**Default:** 0.10 (10% of cells co-active) — a common threshold in the in vitro network activity literature, but highly data-dependent.

---

### Min. event distance (s)
**What it does:** Minimum time that must separate two detected network events. Prevents a single broad burst with a bumpy peak from being counted as multiple events.

**How to set it:** Should be roughly equal to the typical duration of a network burst in your preparation. For in vitro cortical cultures this is typically 0.5–2 s.

**Default:** 0.5 s — a conservative lower bound. Increase if you see the same burst being counted multiple times.

---

### Imaging interval (s)
Used to convert network event counts to rates (events/second) and to scale time axes in network plots. Should match the value used during deconvolution. Auto-filled when switching from Temporal to Network analysis mode.