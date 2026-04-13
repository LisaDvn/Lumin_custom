# GUI Usage

## Design goals
The graphical user interface (GUI) was selected and modified with the following objectives:

* User-friendly
* No coding required
* Interactive analysis

## Tool selection

Several existing platforms were evaluated as potential foundations for the tool, including Suite2p and Lumin.

Lumin was selected as the primary platform due to:

* Its intuitive and accessible interface for biological end users
* Built-in support for visualization of calcium imaging data
* Flexibility to integrate custom analysis modules
* Compatibility with Python-based workflows
* Reduced need for manual configuration compared to alternative tools
* Focus on analysis of in vitro calcium imaging analysis

This makes Lumin particularly suitable for rapid analysis and interactive exploration of calcium imaging datasets.

---

## Features
The GUI provides the following functionality:

* Folder selection
Easy loading of input data and project directories
* Parameter adjustment
Interactive tuning of analysis parameters (e.g. segmentation, event detection)
* Visualization of results
Fluorescence traces (ΔF/F)
Detected events and spikes
ROI overlays on imaging data
* Interactive inspection
Ability to explore individual neurons and validate analysis results
