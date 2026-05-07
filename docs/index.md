# **Calcium imaging analysis tool**

Welcome to the NAPARI-based calcium imaging analysis pipeline for in vitro (hiPSC) recordings.

This tool provides an end-to-end workflow for:

- Cell segmentation
- Signal extraction
- Event and spike detection
- Single-cell analysis
- Network analysis

Because the pipeline is modular, each analysis step generates intermediate outputs that can be inspected, validated, and re-used independently, allowing users to choose which steps to run and when.

## Why this tool?

Current calcium imaging workflows are often fragmented across FIJI/ImageJ, MATLAB, spreadsheets, and custom scripts, making analysis time-consuming, difficult to reproduce, and dependent on manual intervention. This pipeline builds upon the open-source LUMIN framework: a modular, end-to-end environment for in vitro calcium imaging analysis of (iPSC-derived) neuronal cultures. LUMIN is a plugin for napari, providing an interactive graphical interface without requiring coding experience.

This pipeline extends LUMIN with combining automated preprocessing, manual segmentation refinement options and model-based methods such as OASIS spike inference. The workflow supports scalable, reproducible, and user-friendly calcium imaging analysis across experiments and datasets.