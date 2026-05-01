# 1. Data input & metadata

This step prepares your raw imaging data and generates the required project structure for downstream analysis.

---

## Supported input formats

LUMIN accepts the following raw file formats:

- `.tiff` / `.tif`
- `.nd2` (Nikon)

All files with the same metadata should be placed inside a single input folder.

---

## Required inputs

In the **Preprocessing widget**, you must provide:

### Input folder (ND2/TIFF)
Select the folder containing your raw imaging files.

### Project directory
Choose or create a directory where all pipeline output will be stored.

### Plate ID
Identifier for the imaging plate (e.g. `Plate_01`).

### Cell line
Biological replicate or cell type (e.g. `Striatal`, `iPSC-derived neurons`).

### Condition
Experimental condition (e.g. `Control`, `Drug_A`, `HD50`).

---

## Optional metadata

You can optionally include additional metadata for better data tracking and FAIR compliance:

- **Researcher ID** — name or initials of the experimenter  
- **Research institute** — lab or institution  
- **Experiment ID** — unique experiment identifier  
- **Replicate number** — replicate index (e.g. 1, 2, 3)  
- **Microscope** — imaging system used  
- **Days in vitro (DIV)** — cell maturation stage  
- **Calcium indicator** — e.g. GCaMP6, Fluo-4  
- **Cell density** — plating density or estimate  

These fields are not required for analysis but are stored in the metadata output.

---

## What happens during preprocessing?

After clicking **Run**, LUMIN will:

1. Scan the input folder for ND2/TIFF files  
2. Extract metadata from file names and user input  
3. Convert files (if needed) into a standardized format  
4. Generate a structured project directory  
5. Create an `_input.csv` file used in downstream steps  

---

### `_input.csv`
This file contains all required information for segmentation and analysis:

- `plate_id`
- `filename`
- `filepath`
- `cell_line`
- `condition`

---

## Notes

- Each run appends metadata consistently across all files in the selected folder  
- Ensure that all files in the folder belong to the same experimental context  
- If working with multiple conditions or plates, run preprocessing separately per dataset  

---

## Demo

<video controls style="width: 100%;">
  <source src="../videos/demo_preprocessing.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
```