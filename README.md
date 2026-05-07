# Calcium imaging analysis tool (based on LUMIN)

Automated pipeline for calcium imaging data analysis from iPSC-derived neurons.
Includes segmentation, signal extraction, event detection and feature extraction.

## What it does

- Automated cell segmentation (Cellpose)
- ΔF/F signal extraction
- Event and spike detection (peak detection + OASIS deconvolution)
- Single-cell feature extraction
- Excel and PDF outputs

For full documentation, installation and usage, please see the[documentation site.](https://lisadvn.github.io/Lumin_custom/)

## Acknowledgements

This pipeline is based on and extends the functionality of LUMIN:

> Winter, M. et al. LUMIN: a framework for label-free calcium imaging analysis.
> Licensed under CC BY 4.0.

Original publication:
https://doi.org/10.1038/s41598-026-40269-0

LUMIN is distributed under the Creative Commons Attribution 4.0 International License:
http://creativecommons.org/licenses/by/4.0/

This project includes modifications and extensions for:
- preprocessing workflows
- modular network analysis
- spike inference
- metadata integration
- extended GUI functionality