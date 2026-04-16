import os
import json
import logging
from datetime import datetime
import subprocess
import skimage.io
import os

# -------------------- LOGGING & RUN MANAGER --------------------
class RunManager:
    def __init__(self, base_dir="analysis_runs", parent_run_id=None, stage=None):

        if parent_run_id:
            self.run_id = parent_run_id
            self.run_dir = os.path.join(base_dir, self.run_id, stage)
        else:
            self.run_id = datetime.now().strftime("run_%Y_%m_%d_%H%M%S")
            self.run_dir = os.path.join(base_dir, self.run_id)

        os.makedirs(self.run_dir, exist_ok=True)
        self.output_dir = os.path.join(self.run_dir, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(self.run_id)

        if not logger.handlers:  # prevent duplicates
            logger.setLevel(logging.INFO)

            log_file = os.path.join(self.run_dir, "pipeline.log")
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

            logger.addHandler(fh)

        return logger

    def log(self, msg):
        print(msg)
        self.logger.info(msg)

    def save_json(self, name, data):
        def convert(obj):
            from pathlib import Path
            if isinstance(obj, Path):
                return str(obj)
            return obj

        path = os.path.join(self.run_dir, name)
        with open(path, "w") as f:
            json.dump(data, f, indent=4, default=convert)

        return path

    def get_git_commit(self):
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"]
            ).decode().strip()
        except:
            return "unknown"

# -------------------- METADATA & PARAMETERS --------------------
def build_metadata(run_manager, image_id, condition, replicate, filepath):
    return {
        "dataset": {
            "image_id": image_id,
            "condition": condition,
            "replicate": replicate,
            "input_file": filepath
        },
        "provenance": {
            "run_id": run_manager.run_id,
            "git_commit": run_manager.get_git_commit(),
            "date": datetime.now().isoformat()
        }
    }

def build_parameters(method, segmentation_parameters):
    return {
        "segmentation_method": method,
        "parameters": {
            "diameter": segmentation_parameters[0],
            "cellprob_threshold": segmentation_parameters[1] if len(segmentation_parameters) > 1 else None
        }
    }

# -------------------- EXPORT RESULTS --------------------

def export_results(run, image, mask, traces, metadata, parameters, filename):

    run.log(f"Exporting results for {filename}")

    # save outputs
    img_path = os.path.join(run.output_dir, "image.tiff")
    mask_path = os.path.join(run.output_dir, "mask.tiff")
    trace_path = os.path.join(run.output_dir, "traces.csv")

    skimage.io.imsave(img_path, image)
    skimage.io.imsave(mask_path, mask.astype("uint16"))

    import pandas as pd
    pd.DataFrame(traces).to_csv(trace_path, index=False)

    # add output paths to metadata
    metadata["outputs"] = {
        "image": img_path,
        "mask": mask_path,
        "traces": trace_path
    }

    # save files
    run.save_json("metadata.json", metadata)
    run.save_json("parameters.json", parameters)

    # run manifest 
    run.save_json("run_manifest.json", {
        "run_id": run.run_id,
        "files": list(os.listdir(run.run_dir))
    })

    run.log("Export complete ✔")

    return run.run_dir
