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
):
    """
    Full conversion pipeline:
    ND2/LIF → TIFF + CSV
    """

    #output paths
    output_dir = os.path.join(project_dir, "Preprocessing", "TIFF")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(project_dir, "_input_data.csv")

    #initialize Bio-Formats
    print("Initializing Bio-Formats...")
    ij = imagej.init('sc.fiji:fiji', mode='headless')
    BF = jimport('loci.plugins.BF')

    # helper to append a row to the CSV
    def append_row(row):
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # find all ND2/LIF/AVI files in the input directory
    files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".nd2", ".lif", ".avi"))
    ]

    print(f"Found {len(files)} files")

    # process each file
    for i, fname in enumerate(files):
        fpath = os.path.join(input_dir, fname)

        print(f"[{i+1}/{len(files)}] Processing {fname}...")

        ext = os.path.splitext(fname)[1].lower()

        try:
            if ext in [".nd2", ".lif"]:
                img = BF.openImagePlus(fpath)
                data = ij.py.from_java(img[0])

            elif ext == ".avi":
                import imageio.v2 as imageio

                data = imageio.mimread(fpath)
                data = np.stack(data)

                if data.ndim == 4:
                    data = data[..., 0]

            else:
                print(f"Skipping unsupported file: {fname}")
                continue

            # save TIFF and csv
            out_name = os.path.splitext(fname)[0] + ".tif"
            out_path = os.path.join(output_dir, out_name)

            tiff.imwrite(out_path, np.asarray(data), imagej=True)

            row = {
                "plate_id": plate_id,
                "filename": out_name,
                "filepath": out_path,
                "cell_line": cell_line,
                "condition": condition,
                "format": ext,
            }

            append_row(row)

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    print(f"\nDone. CSV saved at: {csv_path}")