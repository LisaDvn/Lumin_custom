# 3. Event detection

This step detects calcium transients directly from fluorescence traces using ΔF/F normalization and peak-based event detection. It produces single-cell activity metrics for downstream analysis.

---

Open the **Trace quantification** widget and select the project directory generated after segmentation and signal extraction.

---

## Analysis modes

| Mode | Use when |
|---|---|
| **Compound-evoked activity** | Recordings include a defined stimulation event (e.g. drug addition) |
| **Spontaneous activity** | No external stimulus; analysing baseline calcium dynamics |

---

## Activity types (compound-evoked only)

| Type | Description |
|---|---|
| **Baseline shift** | Sustained fluorescence change after stimulation (AUC-based) |
| **Spontaneous** | Discrete calcium spikes within the recording window |

---

## Normalization

Fluorescence traces are normalized before event detection:

| Method | Description |
|---|---|
| **Sliding window** | Rolling percentile-based baseline (general use) |
| **Pre-stimulus window** | Baseline estimated from pre-stimulation frames |

---

### Sliding window parameters
- **Window size** — number of frames used for baseline estimation  
- **Percentile threshold** — defines baseline level (higher = stricter baseline)  

---

## Event detection (peak-based)

Events are detected directly from ΔF/F traces using threshold-based peak detection.

### Key parameters

| Parameter | Description |
|---|---|
| Smoothing | Apply smoothing before peak detection |
| Prominence | Minimum peak prominence for event detection |
| Amplitude/width ratio | Filters broad, low-amplitude events |
| Imaging interval (s) | Converts frames to time |
| Analysis window | Frame range used for analysis |

---

## Parameter optimisation

Use the interactive testing tools to tune detection settings:

### Test settings on random recording
- Samples a random dataset entry  
- Displays:
  - Baseline estimation  
  - Detected events  
  - Calcium trace with event overlays  
- Allows real-time parameter adjustment  

### Test settings on same recording
- Re-runs analysis on the same sample for iterative tuning  

!!! note
    This step does not save results and is intended for parameter tuning only.

---

## Output

After running event detection, the following outputs are generated:
```bash
Quantification/
├── Tables/
│ ├── cell_properties_event_detection.pkl
│ ├── cell_properties_event_detection.csv
│ └── event_detection_parameters.txt
├── Plots/
│ ├── traces_with_events.pdf
│ ├── event_frequency_distribution.pdf
│ ├── amplitude_distribution.pdf
│ └── condition_comparisons.pdf
```

---

## Notes

- Event detection is **trace-based and independent of network analysis**
- Results depend strongly on parameter selection (use test mode before running full analysis)
- Recommended to validate on multiple recordings per condition
- ΔF/F normalization is required for all event detection workflows

---

## Demo

<video controls style="width: 100%;">
  <source src="../videos/demo_eventdetection.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---
