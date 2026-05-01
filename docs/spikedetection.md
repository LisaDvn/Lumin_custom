# 4. Network activity analysis

This step quantifies coordinated activity across cells by analysing spike trains and computing network-level metrics such as synchrony and population events.

---

Open the **Network activity** widget and select the project directory containing the processed event or spike data.

---

## Analysis modes

| Mode | Description |
|---|---|
| **Temporal (spike trains)** | Analyse temporal spike structure across cells |
| **Network analysis** | Compute network metrics from pre-computed spike trains |
| **Combined** | Perform spike inference and network analysis in a single step |

---

## Spike inference methods

### OASIS
Model-based deconvolution of calcium traces into spike trains.

- **Indicator decay (τ)** — calcium indicator decay time constant (s)  
- **Sampling rate** — recording frame rate (Hz)  
- **Baseline method** — `maximin` (recommended), `constant`, or `prctile`  
- **Baseline window** — rolling baseline window size (s)  
- **Baseline σ** — Gaussian smoothing applied before baseline estimation  

---

### CASCADE
Deep learning-based spike inference method.

- Select a pre-trained model matching your frame rate (auto-downloaded on first use)  
- **Spike threshold** — threshold applied to spike-rate output (0 = no threshold)  

---

## Network analysis parameters

| Parameter | Description |
|---|---|
| Network event threshold | Fraction of co-active neurons required to define a population burst (e.g. 0.10 = 10%) |
| Min. event distance (s) | Minimum time between successive network events |
| Sampling rate (Hz) | Frame rate used for metric calculations |
| Spike binarisation threshold | Threshold to convert spike probabilities into binary events |

---

## Parameter optimisation

Use small-scale testing before running full analysis:

- Run on selected recordings to validate spike inference quality  
- Adjust thresholds to control network event sensitivity  
- Compare outputs across conditions for consistency  

!!! note
    This step does not save results and is intended for parameter tuning only.
---

## Output

After running network analysis, the following outputs are generated:

```bash
Network/
├── Spike_trains/
│ ├── spike_trains.pkl
│ └── spike_trains.csv
├── Metrics/
│ ├── network_events.csv
│ ├── synchrony_metrics.csv
│ ├── burst_analysis.csv
│ └── participation_scores.csv
├── Plots/
│ ├── raster_plot.pdf
│ ├── network_activity_overview.pdf
│ ├── synchrony_heatmap.pdf
│ └── population_bursts.pdf
```

---

## Notes

- Network metrics depend on quality of prior segmentation and event detection  
- Spike inference method strongly influences network structure (OASIS vs CASCADE)  
- Recommended to keep parameters consistent across datasets for comparability  
- Combined mode is computationally heavier but fully automated  

---

## Demo

<video controls style="width: 100%;">
  <source src="../videos/demo_spikedetection.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---
