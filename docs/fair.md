# FAIR Principles

To ensure that the developed calcium imaging analysis tool is reusable and aligned with best practices in scientific data management, the pipeline was adapted to meet the FAIR principles: **Findable, Accessible, Interoperable, and Reusable**.

## 1. Findable

The analysis workflow, codebase, and documentation are structured ~and version-controlled~ to ensure that they can be easily located and referenced.

* The full analysis pipeline is documented using structured documentation (MkDocs), including:

  * Installation instructions
  * Step-by-step workflow description
  * Parameter explanations
* Code and documentation are maintained in a version-controlled repository (GitHub), enabling traceability of changes.
* Standardized file naming conventions are used for both input and output data.
* The modular structure of the LUMIN pipeline, ensures that each processing step (segmentation, event detection, analysis) is clearly defined and can be independently identified and referenced.

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
  * ~`.avi` (converted to `.tiff` using Bio-Formats)~
* Metadata:

  * `.csv` files (structured tabular format)

### Output formats

* Quantitative data:

  * `.csv` / `.xlsx` (feature tables)
* Imaging data:

  * `.tiff` (ROI masks)
* Reports and visualization:

  * `.pdf`

The integration of Bio-Formats ensures that proprietary microscopy formats can be converted into open, standardized formats without loss of essential metadata. 
Additionally, the Napari environment supports visualization of multidimensional imaging data across formats and the support for plugins allows customization and extending.
---

## 4. Reusable

The pipeline is designed to support reproducibility and reuse across experiments and research groups.

* A standardized workflow ensures consistent processing of calcium imaging data.
* Metadata integration enables reproducible parameter selection across datasets.
* All analysis steps (segmentation, signal extraction, feature extraction) are documented and transparent.
* The modular design in Napari allows:

  * Replacement of individual components (e.g., segmentation method)
  * Extension with new analysis modules
* Output files are structured and labeled, enabling downstream analysis and sharing.

### Limitations

* The current validation is based on a limited number of datasets (proof-of-concept), which may affect generalizability.
* Future work should include validation on larger and more diverse datasets to further improve reusability.

