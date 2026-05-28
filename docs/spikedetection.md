# 4. Network activity analysis

This step quantifies coordinated activity across cells by analysing spike trains and computing network-level metrics such as synchrony and population events.

---

Open the **Network activity** widget and select the project directory containing the processed event or spike data.

---

## Analysis modes

- **Temporal (spike trains)** — Estimate action potentials (spike inference) by deconvolving the calcium trace to spike trains
- **Network analysis** — Compute network metrics from pre-computed spike trains 
- **Combined** — Perform spike inference and network analysis in a single step 

---

## Spike inference methods

### OASIS [\[Pachitariu et al., 2019\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC12044035/#B47)

OASIS is a model-based deconvolution algorithm that estimates action potentials from calcium fluorescence traces using a first-order autoregressive model.

It is:

* Fast and lightweight
* Well-suited for large datasets
* Recommended for most standard calcium imaging experiments

**OASIS parameters**

| Parameter               | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| **Indicator decay (τ)** | Calcium indicator decay constant in seconds                  |
| **Sampling rate (Hz)**  | Imaging frame rate                                           |
| **Baseline method**     | Drift correction strategy (`maximin`, `constant`, `prctile`) |
| **Baseline window**     | Rolling baseline estimation window                           |
| **Baseline σ**          | Gaussian smoothing applied before baseline estimation        |

---

### CASCADE [\[Rupprecht et al., 2021\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC12044035/#B55)

CASCADE is a deep learning-based spike inference framework trained on simultaneous electrophysiology and calcium imaging recordings.

It is:

* More sensitive to complex calcium dynamics
* Better at recovering dense firing patterns

**CASCADE parameters**

| Parameter              | Description                                       |
| ---------------------- | ------------------------------------------------- |
| **Model**              | Pre-trained model matching the imaging frame rate |
| **Spike threshold**    | Threshold applied to spike probability output     |
| **Sampling rate (Hz)** | Recording frame rate                              |

!!! note
CASCADE models are automatically downloaded the first time they are used.
 

---

### Network analysis parameters

| Parameter                        | Description                                                                     |
| -------------------------------- | ------------------------------------------------------------------------------- |
| **Network event threshold**      | Fraction of simultaneously active neurons required to define a population event |
| **Min. event distance (s)**      | Minimum time between detected network bursts                                    |
| **Sampling rate (Hz)**           | Imaging frame rate used for temporal scaling                                    |
| **Spike binarisation threshold** | Threshold used to convert spike probabilities into binary events                |

---

## Extracted network metrics

Depending on the selected workflow, LUMIN computes:

* Spike frequency
* Inter-spike interval (ISI)
* Population firing rate
* Network burst frequency
* Fraction of active neurons
* Synchrony/co-activity measures
* Temporal activity profiles
* Network event statistics

---

## Parameter optimisation

Use the interactive testing tools before running full analysis:

* **Test settings on random recording** → preview analysis on a randomly selected recording
* **Test settings on same recording** → repeatedly test parameters on the same sample for direct comparison

The following visualisations are shown during testing:

* Deconvolved signal
* Inferred spike train

This allows rapid optimisation of spike inference and network detection parameters before batch processing the full dataset.

!!! note
    Testing mode is intended for parameter optimisation only and does not save analysis outputs.

---

## Notes

- Network metrics depend on quality of prior segmentation and event detection  
- Spike inference method strongly influences network structure  
- Recommended to keep parameters consistent across datasets for comparability  

---

## Demo

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe
    src="https://www.youtube.com/embed/NWh1Gor_-WM"
    title="Demo video"
    frameborder="0"
    allowfullscreen
    style="position: absolute; top:0; left:0; width:100%; height:100%;">
  </iframe>
</div>


---

