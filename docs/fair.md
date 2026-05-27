# FAIR Principles

To ensure that the developed calcium imaging analysis tool is reusable and aligned with best practices in scientific data management, the pipeline was adapted to meet the ([FAIR principles](https://www.go-fair.org/fair-principles/)): **Findable, Accessible, Interoperable, and Reusable**.

## 1. Findable

The analysis workflow, codebase, and documentation are structured and version-controlled to ensure that they can be easily located and referenced.

* The full analysis pipeline is documented using structured documentation (MkDocs), including installation instructions, a step-by-step workflow description, and parameter explanations.
* Code and documentation are maintained in a version-controlled repository (GitHub), enabling traceability of changes.
* Standardized file naming conventions are used for both input and output data, including method-tagged output filenames (e.g., `networkmetrics_per_well_OASIS.csv`).
* The modular structure of the LUMIN pipeline ensures that each processing step (segmentation, event detection, analysis) is clearly defined and can be independently identified and referenced.

---

## 2. Accessible

The tool and its outputs are designed to be easily accessible to users with varying technical backgrounds.

* The pipeline includes a graphical user interface (GUI), eliminating the need for programming knowledge.
* Documentation is publicly available and written for non-technical users.
* Input and output files can be accessed using standard file systems without proprietary software.
* The tool is developed in Python and distributed as open-source software, avoiding licensing barriers (e.g., MATLAB).
* Napari offers an interactive, user-friendly interface for image visualization and manual refinement of segmentation results.

---

## 3. Interoperable

To ensure compatibility with other tools and workflows, the pipeline uses open and widely supported file formats.

### Input formats

* Imaging data:
  * `.tiff` (standard format for microscopy data)
  * `.nd2` (converted to `.tiff` using Bio-Formats)

* Metadata:
  * `.csv` (structured tabular format)

### Output formats

* Quantitative data:
  * `.csv` (feature tables)
  * `.pkl` (serialized Python DataFrames for intermediate results and downstream analysis within the pipeline)

* Imaging data:
  * `.tiff` (ROI masks)

* Experiment configuration and metadata:

  * `.json` (per-run parameter exports, enabling exact reproduction of analysis settings)

* Logging:

  * `.log` (structured run logs recording processing steps, parameter values, warnings, and errors per analysis session)

* Reports and visualization:
  * `.pdf`

The integration of Bio-Formats ensures that proprietary microscopy formats can be converted into open, standardized formats without loss of essential metadata. Additionally, the Napari environment supports visualization of multidimensional imaging data across formats, and its plugin architecture allows customization and extension.

Interoperability is further supported by embedding experimental metadata (e.g., `plate_id`, `cell_line`, `stimulation`, `deconv_method`) directly in all output tables, making outputs self-describing and linkable across datasets without reliance on external databases.

---

## 4. Reusable

The pipeline is designed to support reproducibility and reuse across experiments and research groups.

* A standardized workflow ensures consistent processing of calcium imaging data.
* Metadata integration enables reproducible parameter selection across datasets.
* All analysis parameters are exported as `.json` files at runtime, allowing exact reproduction of any analysis session.
* Run logs (`.log`) provide a traceable record of each session, including parameter values, processing steps, and any warnings or errors encountered.
* The deconvolution method used is stored explicitly in every output table (`deconv_method` column), ensuring that results remain interpretable and comparable across runs.
* All analysis steps (segmentation, signal extraction, feature extraction) are documented and transparent.
* The modular design in Napari allows replacement of individual components (e.g., segmentation method) and extension with new analysis modules.
* Output files are structured and labeled, enabling downstream analysis and sharing.

### Limitations

* `.pkl` files are Python-specific and cannot be opened without a Python environment, limiting their interoperability with non-Python tools.
* The semicolon-delimited `.csv` format used in outputs may require delimiter configuration in tools that expect comma-separated values by default.
* The GUI is desktop-based and requires local installation; no browser-based or remote access is currently supported.
* The current validation is based on a limited number of datasets (proof-of-concept), which may affect generalizability.
* Future work should include validation on larger and more diverse datasets to further improve reusability.