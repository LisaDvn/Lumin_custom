# Installation & getting Started

This page explains how to install LUMIN and run your first analysis.

---

## 1. Requirements

Python 3.10+ via Anaconda ([Miniconda installation page](https://www.anaconda.com/docs/getting-started/miniconda/install).) 

---

## 2. Installation

Open **Anaconda Prompt** and run:

```bash
git clone https://github.com/LisaDvn/Lumin_custom.git
cd Lumin_custom
conda create -n lumin_env python=3.10 cudatoolkit=11.2 cudnn=8.1.0 -c conda-forge -y
conda activate lumin_env
pip install -e .
```

---

## 3. Launch the application

```bash
napari
```

LUMIN is available as a Napari plugin.

---

## 4. First analysis workflow

Once installed, follow the full pipeline:

### Step 1 — Input
Prepare your recordings and metadata.

→ [Go to Input module](input.md)

---

### Step 2 — Segmentation & signal extraction
Detect cells and extract fluorescence traces.

→ [Go to Segmentation module](segmentation.md)

---

### Step 3 — Event detection & features
Detect calcium events and extract single-cell metrics.

→ See: [Go to Event detection module](eventdetection.md)

---

### Step 4 — Network analysis (optional)
Infer spike trains and compute network metrics.

→ See: [Go to Spike detection module](spikedetection.md)

---

