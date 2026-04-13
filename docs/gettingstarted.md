# Getting Started

## Step 1: Input format

LUMIN requires a .csv files including the following columns: plate_id, filename, filepath, biological_replicate, and stimulation.
See an example of this format [here](input.md#dbnpy)
The recordings should be in .tiff format.

## Optional step: Preparing your data
To automatically compute the desired format from recordings, use the preprocessing step. 

Select the folder including all the tiffs/nd2 files, LUMIN will extract the metadata and include the additional input about:

* Plate ID
* Biological replicate
* Stimulation

---

## Step 2: Run the segmentation and signal extraction

After selecting the Segmentation and signal ectraction option, select the _input.csv file and the accompanying folder with .tiff files for segmentation.

### Step 2.1: Inspect results

Select the automated or hybrid segmentation option, and optionally test settings on a random image. The tool will generate a first version of a segmentation mask. 

When satisfied, run the analysis. 

If the hybrid selection was selected, you will see the second version of the segmented images. This you can refine by adding or deleting ROIs, making the final ROI mask. 

After segmenting the all the recordings, the fluorescence signals will be extracted and the following folders will be conducted: 
* Masks -> projected and final mask per recording (.tiff)
* Plots -> initial and final mask + labelled mask and raw fluorescence traces per recording (.pdf)
* tables -> cell properties and signal extraction (.pkl)

---

## Step 3: Single cell analysis



The tool will generate:

features.xlsx
analysis_report.pdf
