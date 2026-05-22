import os
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# RunManager
# ──────────────────────────────────────────────────────────────────────────────
# Directory layout produced for a project_dir = "/data/my_project":
#
#   /data/my_project/
#   └── logs/
#       ├── run_info.json                   ← {"run_id": "run_2025_…"}
#       ├── pipeline.log                    ← unified log, every stage appends
#       ├── parameters.json                 ← unified params, merged per stage
#       └── runs/
#           └── run_2025_05_21_143000/
#               ├── segmentation/
#               │   ├── pipeline.log        ← stage-specific log
#               │   └── parameters_segmentation.json
#               ├── quantification/
#               │   ├── pipeline.log
#               │   └── parameters_quantification.json
#               └── network/
#                   ├── pipeline.log
#                   └── parameters_network.json
#
# Usage
# ─────
# Stage 1 – segmentation (creates the run_id):
#   run = RunManager(project_dir)
#   json.dump({"run_id": run.run_id}, open(run_info_path, "w"))
#
# Stage 2 – quantification (re-uses the run_id):
#   run_info = json.load(open(run_info_path))
#   run = RunManager(project_dir, parent_run_id=run_info["run_id"],
#                    stage="quantification")
#
# Stage 3 – network (same pattern as stage 2):
#   run = RunManager(project_dir, parent_run_id=run_info["run_id"],
#                    stage="network")
#
# Logging:
#   run.log("some message")        → both stage log AND unified log
#
# Parameters:
#   run.save_json("parameters_segmentation.json", params_dict)
#                                  → stage dir  AND  merged into unified params.json
# ──────────────────────────────────────────────────────────────────────────────


class RunManager:
    """
    Manages logging and parameter persistence across all pipeline stages.

    Key guarantees:
    • Every run.log() call goes to BOTH the stage-specific log and
      the project-wide ``logs/pipeline.log``.
    • Every run.save_json("parameters_<stage>.json", data) call writes to
      the stage directory AND merges the data into ``logs/parameters.json``
      under a key matching the stage name.
    • The unified files persist and grow across stages so the final
      ``logs/parameters.json`` contains segmentation + quantification +
      network sections after all three stages have run.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        project_dir: str,
        parent_run_id: str | None = None,
        stage: str | None = None,
    ):
        self.project_dir = Path(project_dir)
        self.stage = stage or "general"

        # ── unified log directory (project-level) ──────────────────────
        self._unified_dir = self.project_dir / "logs"
        self._unified_dir.mkdir(parents=True, exist_ok=True)

        # ── run identity ───────────────────────────────────────────────
        if parent_run_id:
            self.run_id = parent_run_id
        else:
            self.run_id = datetime.now().strftime("run_%Y_%m_%d_%H%M%S")

        # ── stage-specific directory ───────────────────────────────────
        self.run_dir = self._unified_dir / "runs" / self.run_id / self.stage
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # outputs sub-folder (kept for backward compat with export_results)
        self.output_dir = self.run_dir / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── loggers ───────────────────────────────────────────────────
        self.logger = self._setup_logger()

    # ------------------------------------------------------------------
    # Logger setup
    # ------------------------------------------------------------------

    def _setup_logger(self) -> logging.Logger:
        """
        Returns a logger that writes to:
          • <run_dir>/pipeline.log           (stage-specific)
          • <unified_dir>/pipeline.log       (project-wide, all stages)
        """
        logger_name = f"{self.run_id}.{self.stage}"
        logger = logging.getLogger(logger_name)

        if logger.handlers:          # avoid duplicate handlers on re-init
            return logger

        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")

        for log_path in (
            self.run_dir / "pipeline.log",         # stage log
            self._unified_dir / "pipeline.log",    # unified log
        ):
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

        return logger

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def log(self, msg: str) -> None:
        """Print to stdout and write to both stage and unified log files."""
        print(msg)
        self.logger.info(msg)

    def save_json(self, name: str, data: dict) -> Path:
        """
        Save *data* to ``<run_dir>/<name>`` and merge it into the unified
        ``logs/parameters.json`` under a section key derived from *name*.

        The section key is the file stem with 'parameters_' stripped, e.g.:
          'parameters_segmentation.json' → 'segmentation'
          'parameters_quantification.json' → 'quantification'
          'parameters_network.json' → 'network'
          'parameters.json' → 'general'
          'metadata.json' → 'metadata'

        Merging is additive: existing sections are preserved so that the
        unified file grows across stages rather than being overwritten.
        """
        def _convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

        # ── write stage-specific file ──────────────────────────────────
        stage_path = self.run_dir / name
        with open(stage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, default=_convert)

        # ── derive section key ─────────────────────────────────────────
        stem = Path(name).stem                       # e.g. 'parameters_segmentation'
        if stem.startswith("parameters_"):
            section = stem[len("parameters_"):]      # → 'segmentation'
        elif stem == "parameters":
            section = self.stage
        else:
            section = stem                           # e.g. 'metadata'

        # ── load-merge-save unified parameters.json ────────────────────
        unified_path = self._unified_dir / "parameters.json"
        unified: dict = {}
        if unified_path.exists():
            try:
                with open(unified_path, encoding="utf-8") as f:
                    unified = json.load(f)
            except (json.JSONDecodeError, OSError):
                unified = {}

        # deep-merge: if the section already exists, update it rather
        # than replacing wholesale (handles reruns of the same stage)
        if section in unified and isinstance(unified[section], dict) and isinstance(data, dict):
            unified[section].update(data)
        else:
            unified[section] = data

        # always stamp the last-updated time
        unified.setdefault("_meta", {})["last_updated"] = datetime.now().isoformat()
        unified["_meta"].setdefault("run_id", self.run_id)

        with open(unified_path, "w", encoding="utf-8") as f:
            json.dump(unified, f, indent=4, default=_convert)

        return stage_path

    def get_git_commit(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Metadata / parameter helpers  (unchanged API)
# ──────────────────────────────────────────────────────────────────────────────

def build_metadata(run_manager: RunManager, image_id, condition, replicate, filepath) -> dict:
    return {
        "dataset": {
            "image_id": image_id,
            "condition": condition,
            "replicate": replicate,
            "input_file": str(filepath),
        },
        "provenance": {
            "run_id": run_manager.run_id,
            "git_commit": run_manager.get_git_commit(),
            "date": datetime.now().isoformat(),
        },
    }


def build_parameters(method: str, segmentation_parameters: list) -> dict:
    return {
        "segmentation_method": method,
        "parameters": {
            "diameter": segmentation_parameters[0],
            "cellprob_threshold": (
                segmentation_parameters[1] if len(segmentation_parameters) > 1 else None
            ),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Export helper  (unchanged API)
# ──────────────────────────────────────────────────────────────────────────────

def export_results(run: RunManager, image, mask, traces, metadata, parameters, filename) -> str:
    import skimage.io
    import pandas as pd

    run.log(f"Exporting results for {filename}")

    img_path   = run.output_dir / "image.tiff"
    mask_path  = run.output_dir / "mask.tiff"
    trace_path = run.output_dir / "traces.csv"

    skimage.io.imsave(str(img_path), image)
    skimage.io.imsave(str(mask_path), mask.astype("uint16"))
    pd.DataFrame(traces).to_csv(str(trace_path), index=False)

    metadata["outputs"] = {
        "image":  str(img_path),
        "mask":   str(mask_path),
        "traces": str(trace_path),
    }

    run.save_json("metadata.json",   metadata)
    run.save_json("parameters.json", parameters)
    run.save_json("run_manifest.json", {
        "run_id": run.run_id,
        "files":  list(os.listdir(run.run_dir)),
    })

    run.log("Export complete ✔")
    return str(run.run_dir)