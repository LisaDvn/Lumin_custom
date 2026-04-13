# Data input & metadata

## Supported formats

* TIFF 
* Nikon nd2 files (optional with preprocessing)
* AVI  files (will be optional with preprocessing)


## Required files

Input file (_input_data.csv) including:
* plate_id
* filename
* filepath
* biological_replicate
* stimulation


## Optional files
Metadata (.csv) including:
* nuclear_area_scaler
* cell_area_scaler
* intensity_scaler
* fast_mode
* tau 
* frame rate (ms)
 
These parameters are used for correct signal interpretation and feature extraction.
