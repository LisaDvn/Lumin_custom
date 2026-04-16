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

an example of this format is as follows:
plate_id,filename,filepath,biological_replicate,stimulation
1,_CTRL_WIC5_01.tif,C:\Users\name\data\STRIATAL_CTRL_DIV5_01.tif,Striatal,Control
2,_CTRL_WIC5_01.tif,C:\Users\name\data\STRIATAL_HD50_DIV5_02.tif,Striatal,HD50


## Optional files
Metadata (.csv) including:
* nuclear_area_scaler
* cell_area_scaler
* intensity_scaler
* fast_mode
* tau 
* frame rate (ms)
 
These parameters are used for correct signal interpretation and feature extraction.
