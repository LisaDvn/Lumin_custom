# 2. Segmentation and signal extraction

This step detects cells in your recordings and extracts fluorescence signals per region of interest (ROI). It generates all data required for downstream event detection and analysis.

---

## Required inputs

Open the **Cell segmentation** widget and configure the following inputs.

### Input file
Path to `_input.csv` (generated during preprocessing).

### Project directory
Root folder where all segmentation and analysis outputs will be stored.

---

## Optional inputs

### Metadata file
Optional CSV file containing scaling parameters:

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

## Segmentation methods (Automated / Hybrid only)

Depending on whether a nuclear channel is present:

| Nuclear stain | Methods |
|---|---|
| None | Cytoplasmic segmentation (Cellpose) |
| First frame contains nucleus | StarDist (nuclear) + Cellpose (cytoplasmic) |

---

### StarDist parameters
- **Probability threshold** — higher = fewer but more confident ROIs  
- **Overlap threshold** — controls allowed ROI overlap  

---

### Cellpose parameters
- **Model** — `cyto3`, `cyto2`, `cyto`, or custom model  
- **Diameter** — estimated cell size in pixels  
- **Cell probability threshold** — sensitivity of detection  
- **Flow threshold** — segmentation strictness (0 disables filtering)  

---

## Parameter optimisation

Use the testing tools before running full analysis:

- **Test settings on random image** → quick preview on a random recording  
- **Test settings on same image** → iterate on identical sample  

The napari viewer shows segmentation overlays and raw images.

!!! note
    This step does not save results and is intended for parameter tuning only.
---

## Post-filtering (after testing)

| Slider | Description |
|---|---|
| Nuclear overlap | Minimum overlap between nucleus and cytoplasm |
| Nuclear area | Allowed size range for nuclear ROIs |
| Cell area | Allowed size range for cytoplasmic ROIs |
| Fluorescence intensity | Intensity range filter for ROIs |

Enable **Apply filters** to compare raw vs filtered results.

---

## Manual segmentation

- Enable **Co-stain** for labelled subpopulations  
- Assign a **Marker name** for the population  
- Adjust **Point size** for annotation visibility  
- Already annotated images are automatically skipped on reruns  

---

## Output

After running, the following structure is created in your project directory:
```bash
Segmentation/
├── Masks/
│ └── <plate_id>/
│ ├── <filename>_final_mask.tiff
│ └── <filename>_projected.tiff
├── Plots/
│ └── <plate_id>/<filename>/
│ ├── nuclear_mask.pdf
│ ├── calcium_mask.pdf
│ ├── calcium_mask_filtered.pdf
│ ├── labelled_mask_filtered.pdf
│ └── raw_traces.pdf
└── Tables/
├── cell_properties_signal_extraction.pkl
├── cell_properties_signal_extraction.csv
├── parameters.csv
└── segmentation_signal_extraction_config.txt
```
---

## Notes

- Segmentation results are required for all downstream steps  
- Hybrid mode is recommended for best accuracy  
- Each run overwrites only the selected recording outputs  

---

## Demo

<video controls style="width: 100%;">
  <source src="../videos/demo_segmentation.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

