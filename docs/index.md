# **Calcium imaging analysis tool**

Welcome to the NAPARI-based calcium imaging analysis pipeline for in vitro (hiPSC) recordings.

This tool provides an end-to-end workflow for:

- Cell segmentation
- Signal extraction
- Event and spike detection
- Single-cell analysis
- Network analysis


## Why this tool?

Current workflows rely on fragmented pipelines across tools such as FIJI, MATLAB, and spreadsheets, often requiring manual ROI selection and extensive user intervention. These approaches are time-consuming, error-prone, and difficult to reproduce or scale across experiments. 

This tool addresses these limitations by providing an integrated, model-based workflow that standardizes segmentation, signal extraction, and downstream analysis. Each analysis step produces structured outputs, allowing users to inspect intermediate results and flexibly run or repeat specific stages of the pipeline. The result is a more automated, scalable, and reproducible approach that supports both single-cell and network-level analysis with minimal manual effort.

## Workflow

```text
Input → Segmentation + signal extraction → Activity detection → Features
```
## Example output

* ROI masks
* Calcium traces
* Event detection plots
* Feature tables

## Getting started

See the [Installation](installation.md) page.
