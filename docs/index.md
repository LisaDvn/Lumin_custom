# **Calcium imaging analysis tool**

## Overview

This tool provides an automated pipeline for the analysis of calcium imaging data from iPSC-derived neurons. It integrates segmentation, signal extraction, event detection and feature extraction into a single workflow.

## Why this tool?

Current workflows rely on manual ROI selection and spreadsheet-based analysis, which are time-consuming and not reproducible. This tool standardizes the analysis and reduces user intervention.

## Key features

* Automated cell segmentation (Cellpose)
* ΔF/F signal extraction
* Event and spike detection (Peak detection + OASIS deconvolution)
* Single-cell feature extraction
* Network-level analysis (coming)
* Excel and PDF outputs

## Workflow

Input → Segmentation → Signal extraction → Events → Features → Output

## Example output

* ROI masks
* Calcium traces
* Event detection plots
* Feature tables (.xlsx)

## Getting started

See the [Installation](installation.md) and [Getting started](gettingstarted.md) pages.
