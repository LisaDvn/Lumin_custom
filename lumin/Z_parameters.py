import json
from datetime import datetime
import subprocess

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
    except:
        return "unknown"


def generate_output(
    filename,
    image_condition,
    image_replicate,
    image_id,
    image,
    mask,
    filepath,
    output_folder,
    parameters
):
    os.makedirs(output_folder, exist_ok=True)

    # ---------------- SAVE DATA ----------------
    skimage.io.imsave(f"{output_folder}/image.tiff", image)
    skimage.io.imsave(f"{output_folder}/mask.tiff", mask.astype("uint16"))

    # ---------------- METADATA ----------------
    metadata = {
        "dataset": {
            "image_id": image_id,
            "filename": filename,
            "condition": image_condition,
            "replicate": image_replicate,
            "input_file": filepath
        },
        "analysis": {
            "step": "segmentation",
            "method": parameters.get("method"),
            "date": datetime.now().isoformat()
        },
        "output": {
            "image_path": f"{output_folder}/image.tiff",
            "mask_path": f"{output_folder}/mask.tiff"
        },
        "code": {
            "git_commit": get_git_commit()
        }
    }

    with open(os.path.join(output_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # ---------------- PARAMETERS ----------------
    with open(os.path.join(output_folder, "parameters.json"), "w") as f:
        json.dump(parameters, f, indent=4)

    return output_folder