# Workflow Overview

## Pipeline steps

### 1. Input

* Calcium imaging recordings
* Metadata
---
### 2. Segmentation

* ROI detection using Cellpose
* Manual refinement possible
---
### 3. Signal extraction

* Fluorescence traces per ROI
* ΔF/F calculation
---
### 4. Event detection

* Peak detection on ΔF/F traces
* Optional spike inference
---
### 5. Feature extraction

* Single-cell metrics
* Network-level metrics
---
### 6. Output

* Excel feature table
* PDF report
