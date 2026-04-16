# Segmentation & ROI Detection

## Methods

The deep learning method Cellpose was selected due to:

* High segmentation accuracy
* Low parameter tuning
* Good performance on dense cultures

## Output

The segmentation step generates ROI-based outputs stored per image in a `.csv` file:

| filename | condition | replicate | image_id | input_filepath | roi_mask_path | segmentation_overlay_path | centroid_coordinates | roi_labels | overlay_plot_path |
|----------|----------|-----------|----------|----------------|---------------|----------------------------|----------------------|------------|-------------------|
| img001   | control  | 1         | 001      | /data/img001.tiff | /output/mask_img001.tiff | /output/segmented_img001.tiff | [(x1,y1), (x2,y2)] | [1,2,3,...] | /output/Plots/overlay_img001.pdf |

## Manual refinement

Users can:

* Remove false positives
* Add missing cells
