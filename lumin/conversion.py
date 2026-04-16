import os
import csv
import numpy as np
import tifffile as tiff
import imagej
from scyjava import jimport

def run_conversion_pipeline(
    input_dir,
    project_dir,
    plate_id,
    cell_line,
    condition,
    extra_metadata=None,   # <- optional dict
):
    """
    Pipeline:
    - ND2 → TIFF + CSV
    - TIFF → CSV only
    """

    print("Running conversion pipeline...")

    csv_path = os.path.join(project_dir, "_input_data.csv")
    output_dir = os.path.join(project_dir, "Preprocessing", "TIFF")
    os.makedirs(output_dir, exist_ok=True)

    extra_metadata = extra_metadata or {}

    # --------------------------
    # FILE LISTS
    # --------------------------
    all_files = os.listdir(input_dir)

    nd2_files = [f for f in all_files if f.lower().endswith(".nd2")]
    tiff_files = [f for f in all_files if f.lower().endswith((".tif", ".tiff"))]

    print(f"ND2 files: {len(nd2_files)}")
    print(f"TIFF files: {len(tiff_files)}")

    # --------------------------
    # INIT BIOFORMATS ONLY IF NEEDED
    # --------------------------
    ij, BF = None, None
    if len(nd2_files) > 0:
        print("Initializing Bio-Formats...")
        ij = imagej.init('sc.fiji:fiji', mode='headless')
        BF = jimport('loci.plugins.BF')

    # --------------------------
    # CSV WRITER
    # --------------------------
    def append_row(row):
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # ==========================
    # 1. ND2 CONVERSION ONLY
    # ==========================
    for i, fname in enumerate(nd2_files):
        fpath = os.path.join(input_dir, fname)

        print(f"[ND2 {i+1}/{len(nd2_files)}] {fname}")

        try:
            img = BF.openImagePlus(fpath)
            data = ij.py.from_java(img[0])
            data = np.array(data)

            out_name = os.path.splitext(fname)[0] + ".tif"
            out_path = os.path.join(output_dir, out_name)

            tiff.imwrite(out_path, data, imagej=True)

            row = {
                "plate_id": plate_id,
                "filename": out_name,
                "filepath": out_path,
                "cell_line": cell_line,
                "condition": condition,
                "format": "nd2",
                "converted": True,
                **extra_metadata,   # optional fields
            }

            append_row(row)

        except Exception as e:
            print(f"ERROR ND2 {fname}: {e}")

    # ==========================
    # 2. TIFF ONLY (NO CONVERSION)
    # ==========================
    for i, fname in enumerate(tiff_files):
        fpath = os.path.join(input_dir, fname)

        print(f"[TIFF {i+1}/{len(tiff_files)}] {fname}")

        try:
            with tiff.TiffFile(fpath) as tif:
                shape = tif.series[0].shape

            row = {
                "plate_id": plate_id,
                "filename": fname,
                "filepath": fpath,
                "cell_line": cell_line,
                "condition": condition,
                "format": "tiff",
                "converted": False,
                "shape": str(shape),
                **extra_metadata,   # optional fields
            }

            append_row(row)

        except Exception as e:
            print(f"ERROR TIFF {fname}: {e}")

    print(f"\nDone. CSV saved at: {csv_path}")