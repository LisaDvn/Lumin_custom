# 2. Segmentation and signal extraction

This step detects cells in your recordings and extracts fluorescence signals per region of interest (ROI). It generates all data required for downstream event detection and analysis.

---

## Required inputs

Open the **Cell segmentation** widget and configure the following inputs.

- **Input file** — Path to `_input.csv` (generated during preprocessing).
- **Project directory** — Root folder where all segmentation and analysis outputs will be stored.


### Optional inputs

- **Metadata file** — Optional CSV file containing scaling parameters:
  - `nuclear_area_scaler`
  - `cell_area_scaler`
  - `intensity_scaler`

These improve ROI filtering and normalization but are not required.

---

## Supported segmentation modes

LUMIN supports three segmentation strategies:

| Mode | Description |
|---|---|
| **Automated** | Fully automated segmentation using StarDist and/or Cellpose |
| **Hybrid** | Automated segmentation followed by manual refinement in napari |
| **Manual selection** | Fully manual ROI annotation inside napari |

---

## Segmentation methods

Depending on whether a nuclear channel is present:

| Nuclear stain | Methods |
|---|---|
| None | Cytoplasmic segmentation (Cellpose) |
| First frame contains nucleus | StarDist (nuclear) + Cellpose (cytoplasmic) |

---

### Cellpose parameters
- **Model** — `cyto3`, `cyto2`, `cyto`, or custom trained model  
- **Diameter** — estimated cell size in pixels  
- **Cell probability threshold** — sensitivity of detection  
- **Flow threshold** — segmentation strictness (0 disables filtering)  

### StarDist parameters
- **Probability threshold** — higher = fewer but more confident ROIs  
- **Overlap threshold** — controls allowed ROI overlap  

---

## Parameter optimisation

Use the testing tools before running full analysis:

- **Test settings on random image** → quick preview on a random recording  
- **Test settings on same image** → iterate on identical sample  

The napari viewer shows segmentation overlays and raw images.

### Post-filtering (after testing)

After generating an initial segmentation preview, you can further refine the detected ROIs using post-filtering controls. These filters help remove incorrectly segmented objects such as debris or abnormally large structures.

The effect of each filter is shown immediately in the napari viewer, allowing rapid visual optimisation before running the full dataset.

- **Cell area** — Removes ROIs that are too small or too large based on the selected size range
- **Fluorescence intensity** — Excludes ROIs with unusually low or high signal intensity


!!! tip
    Post-filtering is intended for interactive parameter optimisation only. No results are saved during testing until the full segmentation pipeline is executed.

---

## Hybrid segmentation

Hybrid mode combines automated segmentation with manual refinement inside the napari viewer. This approach is recommended when fully automated segmentation is insufficient or when additional quality control is required.

During hybrid segmentation:

1. ROIs are first generated automatically using Cellpose and/or StarDist
2. The resulting segmentation masks are loaded into napari as editable layers
3. Users can manually refine the masks before finalizing the analysis

Available refinement actions include:

* adding missing ROIs
* deleting incorrect ROIs
* splitting merged cells

This workflow allows automated models to provide an initial segmentation while still giving full user control over the final ROI set.

---

## Manual segmentation

- Enable **Co-stain** for labelled subpopulations  
- Assign a **Marker name** for the population  
- Adjust **Point size** for annotation visibility  
- Already annotated images are automatically skipped on reruns  

---

## Notes

- Segmentation results are required for all downstream steps  
- Hybrid mode is recommended for best accuracy  
- Each run overwrites only the selected recording outputs  

---

## Demo

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe
    src="https://www.youtube.com/embed/kQwwagh8vb4"
    title="Demo video"
    frameborder="0"
    allowfullscreen
    style="position: absolute; top:0; left:0; width:100%; height:100%;">
  </iframe>
</div>

---

