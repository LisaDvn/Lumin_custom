# 3. Event detection

This step detects calcium activity events directly from extracted fluorescence traces using ΔF/F normalization and peak-based signal analysis. It generates single-cell activity metrics that can be used for downstream phenotyping, condition comparison, and network-level analysis.

---

Open the **Trace quantification** widget and select the project directory generated during segmentation and signal extraction.

---

## Analysis modes

LUMIN supports two activity modes depending on the experiment:

- **Compound-evoked activity** — Recordings contain a defined stimulation event such as drug addition, optogenetic stimulation, or media exchange 
- **Spontaneous activity**     — Recordings without external perturbation, used to analyse intrinsic calcium dynamics

---

## ΔF/F normalization

Before event detection, fluorescence traces are normalized to ΔF/F to reduce baseline variability and improve comparability between cells.

Available normalization methods:

- **Sliding window** — Rolling percentile-based baseline estimation across the recording (default) 
- **Pre-stimulus window** — Baseline estimated only from frames preceding stimulation

---

### Sliding window parameters

| Parameter                | Description                                                                                            |
| ------------------------ | ------------------------------------------------------------------------------------------------------ |
| **Window size**          | Number of frames used for baseline estimation                                                          |
| **Percentile threshold** | Defines the baseline level used for normalization (higher values produce a stricter baseline estimate) |

---

## Activity classification (compound-evoked mode only)

For stimulation experiments, LUMIN can distinguish between different response types:

- **Baseline shift** — Sustained fluorescence increase or decrease after stimulation, quantified using area-under-the-curve (AUC) metrics
- **Spontaneous events** — Discrete calcium transients occurring throughout the recording window

---

## Event detection

Events are identified directly from ΔF/F traces using threshold-based peak detection.

### Event detection parameters

| Parameter                 | Description                                                              |
| ------------------------- | ------------------------------------------------------------------------ |
| **Smoothing**             | Applies optional smoothing before peak detection to reduce noise         |
| **Prominence**            | Minimum peak prominence required to classify an event                    |
| **Amplitude/width ratio** | Filters broad low-amplitude fluctuations and non-specific signal changes |
| **Imaging interval (s)**  | Converts frame indices into real time                                    |
| **Analysis window**       | Restricts analysis to a selected frame range                             |

---

## Parameter optimisation

Use the testing tools before running full analysis:

- **Test settings on random image** → quick preview on a random recording  

The following visualisations will be shown for a random recording:

* ΔF/F normalization
* baseline estimation
* calcium traces with event overlays

### Test settings on same recording

Re-runs analysis on the same sample to allow iterative parameter refinement and direct comparison between settings.

!!! note
    Testing mode is intended for parameter optimisation only and does not save analysis outputs.

---

## Generated outputs

Depending on the selected analysis mode, LUMIN generates:

* normalized ΔF/F traces
* detected event timestamps
* event amplitudes and durations
* firing frequency metrics
* AUC-based response measurements
* per-cell activity summaries

These outputs are used in downstream modules including phenotype comparison, clustering, and network analysis.

---

## Notes

* Accurate segmentation and signal extraction are required for reliable results
* Detection sensitivity depends strongly on parameter selection
* It is recommended to validate settings across multiple recordings and conditions before large-scale analysis

---

## Demo

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe
    src="https://www.youtube.com/embed/Fkmgo1ZjrRo"
    title="Demo video"
    frameborder="0"
    allowfullscreen
    style="position: absolute; top:0; left:0; width:100%; height:100%;">
  </iframe>
</div>
---
