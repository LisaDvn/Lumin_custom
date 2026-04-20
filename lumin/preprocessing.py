import os
import csv
import numpy as np
import tifffile as tiff
#from PIL import Image
#import av <-- not possible in this environment...
#from tqdm import tqdm
import imagej
from scyjava import jimport
from nd2reader import ND2Reader
import tifffile as tiff
from datetime import datetime


def run_preprocessing_pipeline(
    input_dir,
    project_dir,
    plate_id,
    cell_line,
    condition,
    extra_metadata=None,
):
    """
    - ND2 → TIFF + CSV
    - TIFF → CSV only
    """

    print("Running preprocessing pipeline...")
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"{timestamp}_{plate_id}_{cell_line}"
    filename = filename.replace(" ", "_")

    csv_path = os.path.join(project_dir, filename +"_input_data.csv")
    output_dir = os.path.join(project_dir, "Preprocessing", "TIFF")
    os.makedirs(output_dir, exist_ok=True)

    extra_metadata = extra_metadata or {}
    rows = []
    # --------------------------
    # FILE LISTS
    # --------------------------
    all_files = os.listdir(input_dir)

    nd2_files = [f for f in all_files if f.lower().endswith(".nd2")]
    tiff_files = [f for f in all_files if f.lower().endswith((".tif", ".tiff"))]

    print(f"ND2 files: {len(nd2_files)}")
    print(f"TIFF files: {len(tiff_files)}")

    # initialize Bio-Formats if there are ND2 files
    ij, BF = None, None
    if len(nd2_files) > 0:
        print("Initializing Bio-Formats...")
        ij = imagej.init('sc.fiji:fiji', mode='headless')
        BF = jimport('loci.plugins.BF')

    """
    # avi conversion 
    
    for i, fname in enumerate(avi_files):
        print(f"[AVI {i+1}/{len(avi_files)}] {fname}")

        fpath = os.path.join(input_dir, fname)
        
        try:
            container = av.open(fpath)
            frames = []
            
            for frame in tqdm(container.decode(video=0)):
                # Convert frame to grayscale numpy array
                frame_array = frame.to_ndarray(format='gray')
                frames.append(frame_array)

            container.close()

            if len(frames) == 0:
                print(f"WARNING: No frames in {fname}")
            elif len(frames) < 5000:
                print(f"WARNING: Large video, may take time to process: {fname}")
                continue
            
            out_name = os.path.splitext(fname)[0] + ".tif"
            out_path = os.path.join(output_dir, out_name)
            #proj_path = os.path.join(output_dir, out_name + "_projection.tif")

            # Save as multi-layer TIFF
            pil_frames = [Image.fromarray(frame, mode='L') for frame in frames]

            pil_frames[0].save(
                out_path,
                save_all=True,
                append_images=pil_frames[1:],
                compression='tiff_deflate'
            )

            #mean projection for sanity check
            #frames_stack = np.stack(frames, axis=0)
            #mean_projection = np.mean(frames_stack, axis=0).astype(np.uint8)
            #Image.fromarray(mean_projection, mode='L').save('projection.tiff', compression='tiff_deflate')
            
            # After saving, read back and compare
            with Image.open(out_path) as img:
                for i in range(10):
                    img.seek(i)
                    reloaded = np.array(img)
                    print(f"Frame {i} identical: {np.array_equal(frames[i], reloaded)}")
            
            rows = []
            row = {
                "plate_id": plate_id,
                "filename": out_name,
                "filepath": out_path,
                "cell_line": cell_line,
                "condition": condition,
                "format": "avi",
                "converted": True,
            }
            for k, v in extra_metadata.items():
                if v not in ["", None]:
                    row[k] = v

            append_row(row)

        except Exception as e:
            print(f"ERROR ND2 {fname}: {e}")
    """
    # ==========================
    # ND2 CONVERSION 
    # ==========================
    
    for i, fname in enumerate(nd2_files):
        fpath = os.path.join(input_dir, fname)

        print(f"[ND2 {i+1}/{len(nd2_files)}] {fname}")

        try:
            img = BF.openImagePlus(fpath)
            data = ij.py.from_java(img[0])
            data = np.array(data)

            nd2_metadata = extract_nd2_metadata(fpath)

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
            }
            # add ND2 metadata (only if not None)
            for k, v in nd2_metadata.items():
                if v not in [None, ""]:
                    row[k] = v

            # add optional GUI metadata
            for k, v in extra_metadata.items():
                if v not in [None, ""]:
                    row[k] = v
            rows.append(row)

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
            }

            # add optional GUI metadata
            for k, v in extra_metadata.items():
                if v not in [None, ""]:
                    row[k] = v

            rows.append(row)

        except Exception as e:
            print(f"ERROR TIFF {fname}: {e}")
    # ==========================
    # WRITE FINAL CSV
    # ==========================
    if len(rows) > 0:
        # collect ALL keys across all rows
        fieldnames = set()
        for r in rows:
            fieldnames.update(r.keys())

        fieldnames = sorted(fieldnames)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nDone. CSV saved at: {csv_path}")
    else:
        print("No data to write.")


def extract_nd2_metadata(file_path: str) -> dict:
    metadata = {}

    with ND2Reader(file_path) as images:
        metadata["X_pixels"] = images.metadata.get("width")
        metadata["Y_pixels"] = images.metadata.get("height")

        metadata["Z_planes"] = images.metadata.get("z_levels", 1)
        metadata["Timepoints"] = images.metadata.get("num_frames", 1)
        metadata["Channels"] = images.metadata.get("num_channels", 1)

        try:
            metadata["PixelSize_X_um"] = images.metadata["pixel_microns"]
            metadata["PixelSize_Y_um"] = images.metadata["pixel_microns"]
        except:
            metadata["PixelSize_X_um"] = None
            metadata["PixelSize_Y_um"] = None

        metadata["BitDepth"] = images.metadata.get("bits_per_component", None)

    return metadata

