# Workflow Overview

This pipeline provides an end-to-end workflow for calcium imaging analysis, from raw recordings to single-cell and network-level insights. Each step generates structured outputs, allowing inspection and flexible re-analysis.

---

## Pipeline overview

| Step | Input | Output |
|------|------|--------|
| **1. Input** | ND2 / TIFF recordings + metadata | `_input.csv` + organised project structure |
| **2. Segmentation & signal extraction** | `_input.csv` + recordings (`.tiff`) | ROI masks + fluorescence traces (`cell_properties_signal_extraction.pkl`) |
| **3. Event detection** | Fluorescence traces | Detected events + single-cell activity metrics |
| **4. Spike inference** | Fluorescence traces | Inferred activity + network activity metrics|

!!! tip
    You can inspect results after each step before continuing the pipeline.

---

## Pipeline steps

### 1. Input

**Input:**

- Calcium imaging recordings (`.nd2` or `.tiff`)
- Metadata (plate ID, cell line, condition, etc.)

**Output:**

- Structured `_input.csv`
- Organised project directory with converted `.tiff` files (if needed)

---

### 2. Segmentation & signal extraction

Cells are detected and fluorescence signals are extracted per ROI.

**Modes:**

- Automated — ROI detection using Cellpose  
- **Hybrid — automated detection with manual refinement  
- Manual — user-defined ROI selection  

**Input:**

- `_input.csv`
- Image recordings

**Output:**

- ROI masks (`.tiff`)
- Fluorescence traces per cell
- Cell properties table (`.csv` / `.pkl`)

---

### 3. Event detection (ΔF/F)

Fluorescence traces are normalised and analysed for activity.

**Input:**

- Fluorescence traces (`cell_properties_signal_extraction.pkl`)

**Processing:**

- ΔF/F normalisation
- Peak detection

**Output:**

- Detected events per cell
- Single-cell activity metrics

---

### 4. Spike inference

Model-based estimation of action potentials.

**Input:**
- Fluorescence traces (`cell_properties_signal_extraction.pkl`)

**Processing:**
- Deconvolution using OASIS

**Output:**
- Inferred spike trains
- Network activity metrics

---


## Key features

- Modular workflow — run steps independently  
- Model-based spike inference (OASIS)  
- Intermediate outputs for validation  
- Scalable across datasets and experiments  
