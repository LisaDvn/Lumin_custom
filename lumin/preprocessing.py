import os
import csv
import numpy as np
import tifffile as tiff
from datetime import datetime
from pathlib import Path


#  Optional heavy imports – only loaded when needed
def _import_nd2reader():
    try:
        from nd2reader import ND2Reader
        return ND2Reader
    except ImportError as e:
        raise ImportError("nd2reader is not installed. Run: pip install nd2reader") from e


def _init_bioformats():
    """Initialise ImageJ + Bio-Formats"""
    try:
        import imagej
        from scyjava import jimport
        print("Initializing Bio-Formats (this may take a moment)...")
        ij = imagej.init("sc.fiji:fiji", mode="headless")
        BF = jimport("loci.plugins.BF")
        return ij, BF
    except Exception as e:
        raise RuntimeError(f"Failed to initialise Bio-Formats / ImageJ: {e}") from e


#  CSV helpers

def _load_existing_csv(csv_path: str) -> tuple[list[dict], set]:
    """
    Read an existing CSV and return (rows, processed_filenames).
    Returns empty structures when the file does not exist yet.
    """
    if not os.path.exists(csv_path):
        return [], set()

    rows = []
    processed = set()
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                rows.append(dict(row))
                if "filename" in row and row["filename"]:
                    processed.add(row["filename"])
        print(f"Found existing CSV with {len(rows)} record(s). "
              f"Skipping already-processed files.")
    except Exception as e:
        print(f"WARNING: Could not read existing CSV ({e}). Starting fresh.")
        rows, processed = [], set()

    return rows, processed


def _write_csv(csv_path: str, rows: list[dict]) -> None:
    """Write all rows to CSV, collecting every key across all rows as fieldnames."""
    if not rows:
        print("No data to write.")
        return

    fieldnames = []
    seen = set()
    # Preserve insertion order and put 'core' columns first
    priority = ["plate_id", "filename", "filepath", "cell_line",
                "condition", "format", "converted"]
    for col in priority:
        if col not in seen:
            fieldnames.append(col)
            seen.add(col)
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved → {csv_path}  ({len(rows)} total record(s))")


#  ND2 conversion and metadata extraction

def _convert_nd2(fpath: str, out_path: str, ij, BF) -> np.ndarray:
    """
    Open an ND2 file with Bio-Formats, convert to numpy array,
    save as ImageJ-compatible TIFF, and verify the write.

    Returns the numpy array so the caller can inspect shape/dtype.
    Raises on any failure.
    """
    img = BF.openImagePlus(fpath)
    data = np.array(ij.py.from_java(img[0]))

    tiff.imwrite(out_path, data, imagej=True)

    # ── verification ──────────────────────────────────
    written = tiff.imread(out_path)
    if written.shape != data.shape:
        raise ValueError(
            f"Shape mismatch after writing {out_path}: "
            f"expected {data.shape}, got {written.shape}"
        )

    return data


def extract_nd2_metadata(file_path: str) -> dict:
    """Extract acquisition metadata from an ND2 file."""
    ND2Reader = _import_nd2reader()
    metadata = {}

    try:
        with ND2Reader(file_path) as images:
            metadata["X_pixels"]   = images.metadata.get("width")
            metadata["Y_pixels"]   = images.metadata.get("height")
            metadata["Z_planes"]   = images.metadata.get("z_levels", 1)
            metadata["Timepoints"] = images.metadata.get("num_frames", 1)
            metadata["Channels"]   = images.metadata.get("num_channels", 1)
            metadata["BitDepth"]   = images.metadata.get("bits_per_component")

            try:
                px = images.metadata["pixel_microns"]
                metadata["PixelSize_X_um"] = px
                metadata["PixelSize_Y_um"] = px
            except KeyError:
                metadata["PixelSize_X_um"] = None
                metadata["PixelSize_Y_um"] = None

    except FileNotFoundError:
        print(f"  WARNING: ND2 file not found for metadata extraction: {file_path}")
    except Exception as e:
        print(f"  WARNING: Could not extract ND2 metadata from {file_path}: {e}")

    return metadata


# tiff file reading and metadata extraction

def _read_tiff_shape(fpath: str) -> tuple | None:
    """Return the shape of the first TIFF series, or None on failure."""
    try:
        with tiff.TiffFile(fpath) as tf:
            return tf.series[0].shape
    except tiff.TiffFileError as e:
        print(f"  ERROR: Corrupt TIFF file {fpath}: {e}")
    except Exception as e:
        print(f"  ERROR: Could not read TIFF {fpath}: {e}")
    return None


# build metadata row

def _build_row(base: dict, extra_metadata: dict) -> dict:
    """Merge a base row dict with optional user-supplied metadata."""
    row = dict(base)
    for k, v in extra_metadata.items():
        if v not in (None, ""):
            row[k] = v
    return row


# main pipeline

def run_preprocessing_pipeline(
    input_dir,
    project_dir,
    plate_id,
    cell_line,
    condition,
    extra_metadata: dict | None = None,
    progress_callback=None,
) -> None:
    """
    Preprocessing pipeline:
      • ND2  → TIFF conversion + metadata row
      • TIFF → metadata row only  (no conversion)

    Already-processed files (present in the existing CSV) are skipped
    automatically so the CSV is *appended to* rather than overwritten.
    """

    print("=" * 60)
    print("Preprocessing pipeline started")
    print("=" * 60)

    extra_metadata = extra_metadata or {}

    # ── paths ──────────────────────────────────────────────
    timestamp  = datetime.now().strftime("%Y%m%d")
    safe_name  = f"{timestamp}_{plate_id}_{cell_line}".replace(" ", "_")
    csv_path   = os.path.join(project_dir, safe_name + "_input_data.csv")
    output_dir = os.path.join(project_dir, "Preprocessing", "TIFF")
    os.makedirs(output_dir, exist_ok=True)

    # ── load what we already have ───────────────────────────
    existing_rows, already_processed = _load_existing_csv(csv_path)
    new_rows: list[dict] = []
    failed_files: list[str] = []

    # ── discover input files ────────────────────────────────
    try:
        all_files = os.listdir(input_dir)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    nd2_files  = [f for f in all_files if f.lower().endswith(".nd2")]
    tiff_files = [f for f in all_files if f.lower().endswith((".tif", ".tiff"))]
    total = len(nd2_files) + len(tiff_files)
    done  = 0

    print(f"ND2  files found : {len(nd2_files)}")
    print(f"TIFF files found : {len(tiff_files)}")

    # ── ND2 conversion ──────────────────────────────────────
    ij, BF = None, None
    try:
        nd2_to_process = [f for f in nd2_files
                          if os.path.splitext(f)[0] + ".tif" not in already_processed
                          and f not in already_processed]

        if nd2_to_process:
            ij, BF = _init_bioformats()

        for i, fname in enumerate(nd2_to_process, start=1):
            fpath    = os.path.join(input_dir, fname)
            out_name = os.path.splitext(fname)[0] + ".tif"
            out_path = os.path.join(output_dir, out_name)

            # avoid silent overwrite of a same-named converted file
            if os.path.exists(out_path):
                stem     = os.path.splitext(out_name)[0]
                out_name = f"{stem}_duplicate_{i}.tif"
                out_path = os.path.join(output_dir, out_name)
                print(f"  WARNING: duplicate output name, saving as {out_name}")

            print(f"[ND2 {i}/{len(nd2_to_process)}] Converting {fname} ...")

            try:
                _convert_nd2(fpath, out_path, ij, BF)
                nd2_meta = extract_nd2_metadata(fpath)

                row = _build_row(
                    {
                        "plate_id"  : plate_id,
                        "filename"  : out_name,
                        "filepath"  : out_path,
                        "cell_line" : cell_line,
                        "condition" : condition,
                        "format"    : "nd2",
                        "converted" : True,
                        **{k: v for k, v in nd2_meta.items()
                           if v not in (None, "")},
                    },
                    extra_metadata,
                )
                new_rows.append(row)

            except FileNotFoundError:
                print(f"  ERROR: File not found – {fname}")
                failed_files.append(fname)
            except ValueError as e:
                print(f"  ERROR: Verification failed for {fname}: {e}")
                failed_files.append(fname)
            except Exception as e:
                print(f"  ERROR: Unexpected error processing {fname}: {e}")
                failed_files.append(fname)
            
            done += 1
            if progress_callback:
                progress_callback(done, total, f"ND2: {fname}")
                
    finally:
        # Always release the JVM, even if an exception occurred mid-loop
        if ij is not None:
            try:
                ij.dispose()
                print("Bio-Formats / ImageJ disposed.")
            except Exception:
                pass

    # ── TIFF registration (no conversion) ──────────────────
    tiff_to_process = [f for f in tiff_files if f not in already_processed]
    skipped_tiff    = len(tiff_files) - len(tiff_to_process)

    if skipped_tiff:
        print(f"Skipping {skipped_tiff} already-registered TIFF file(s).")

    for i, fname in enumerate(tiff_to_process, start=1):
        fpath = os.path.join(input_dir, fname)
        print(f"[TIFF {i}/{len(tiff_to_process)}] Registering {fname} ...")

        shape = _read_tiff_shape(fpath)
        if shape is None:
            failed_files.append(fname)
            continue

        row = _build_row(
            {
                "plate_id"  : plate_id,
                "filename"  : fname,
                "filepath"  : fpath,
                "cell_line" : cell_line,
                "condition" : condition,
                "format"    : "tiff",
                "converted" : False,
                "shape"     : str(shape),
            },
            extra_metadata,
        )
        new_rows.append(row)

        done += 1
        if progress_callback:
            progress_callback(done, total, f"TIFF: {fname}")

    # ── write combined CSV ──────────────────────────────────
    all_rows = existing_rows + new_rows
    _write_csv(csv_path, all_rows)

    # ── summary ────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Preprocessing complete.")
    print(f"  New records added  : {len(new_rows)}")
    print(f"  Total in CSV       : {len(all_rows)}")
    if failed_files:
        print(f"  Failed files ({len(failed_files)}):")
        for ff in failed_files:
            print(f"    • {ff}")
    print("=" * 60)