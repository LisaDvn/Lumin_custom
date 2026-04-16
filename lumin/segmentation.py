# -------------------- IMPORTS --------------------
import warnings
warnings.filterwarnings("ignore")                   # Suppress all warnings
import sys                                          # for exiting program
from skimage import util, segmentation              # image processing tools
import numpy as np                                  # numerical operations
import napari                                       # interactive viewer (GUI)
from napari_blob_detection import points_to_labels  # convert points → mask
from scipy import ndimage as ndi                    # distance transforms
from csbdeep.utils import normalize                 # image normalization (for StarDist)

# -------------------- STARDIST SEGMENTATION --------------------
def run_stardist(image=None, model_sd=None, prob_thresh_sd=None, overlap_thresh_sd=None):

    """
    Run StarDist segmentation on an image.

    Steps:
    1. Normalize image (important for neural networks)
    2. Predict instance segmentation (cells/nuclei)
    3. Return mask with labeled objects
    """

    mask, _ = model_sd.predict_instances(
        normalize(image, 1, 99.8, axis=(0, 1)),   # normalize intensities
        nms_thresh=overlap_thresh_sd,             # overlap suppression
        prob_thresh=prob_thresh_sd               # probability threshold
    )

    del model_sd  # free memory 

    return mask


# -------------------- CELLPOSE SEGMENTATION --------------------
def run_cellpose(image=None, model_cp=None, diameter_cp=None, cellprob_threshold_cp=None, flow_threshold_cp=None):

    """
    Run Cellpose segmentation: Uses a pretrained deep learning model to detect cells.
    """

    segmentation_params = {
        'diameter': int(diameter_cp),                   # expected cell size
        'cellprob_threshold': cellprob_threshold_cp,    # threshold for cell probability map
        'flow_threshold': flow_threshold_cp,            # threshold for flow map (cell boundaries)
        'resample': False,                              # don't resample image (keep original resolution)
        'do_3D': False,                                 # 2D segmentation (not 3D)
        'stitch_threshold': 0.0,                        # no stitching of small objects (keep all detections)
    }

    # Run model
    mask, _, _ = model_cp.eval(image, **segmentation_params)

    del model_cp  # free memory

    return mask

# -------------------- MANUAL REFINEMENT --------------------
def refine_segmentation(image: np.ndarray,
                        mask: np.ndarray):
    viewer = napari.Viewer()

    # originele image
    viewer.add_image(image, name='image')

    # segmentation mask als labels layer (EDITABLE!)
    labels_layer = viewer.add_labels(mask, name='segmentation')
    labels_layer.editable = True

    # start GUI
    viewer.show(block=True)

    # haal aangepaste mask op
    refined_mask = labels_layer.data

    return refined_mask


# -------------------- HYBRID PIPELINE --------------------
def run_hybrid_segmentation(image,
                            method="cellpose",
                            model=None,
                            params=None,
                            refine=True):
    """
    Hybrid pipeline:
    1. automatic segmentation (Cellpose)
    2. manual refinement in Napari
    """

    # ---------------- AUTO SEGMENTATION ----------------
    if method == "cellpose":
        mask = run_cellpose(
            image=image,
            model_cp=model,
            diameter_cp=params.get("diameter"),
            cellprob_threshold_cp=params.get("cellprob_threshold"),
            flow_threshold_cp=params.get("flow_threshold")
        )

    elif method == "stardist":
        mask = run_stardist(
            image=image,
            model_sd=model,
            prob_thresh_sd=params.get("prob_thresh"),
            overlap_thresh_sd=params.get("overlap_thresh")
        )

    else:
        raise ValueError("Unknown method")

    # ---------------- MANUAL REFINEMENT ----------------
    if refine:
        print("Opening Napari for manual refinement...")
        mask = refine_segmentation(image, mask)

    return mask

# -------------------- FULL MANUAL ROI SELECTION --------------------
def run_manual_selection(image: np.ndarray,
                         point_size: int,
                         first_label: int = 0):

    """
    Manual annotation of ROIs using Napari.

    Workflow:
    - User clicks points on cells
    - Points are converted to segmentation mask
    - Watershed expands points into regions
    """

    # Settings for label display
    label_text = {"string": "label", 'size': 6, 'color': '#eb1717'}
    roi_color = '#66abd9'

    all_labels = set()  # keep track of used labels

    viewer, image_layer = napari.imshow(image)

    # Hide unnecessary UI buttons
    viewer.window._qt_viewer.layerButtons.hide()

    # Initialize empty points layer (user clicks here)
    features = {'label': np.empty(0, dtype=int)}

    points_layer = viewer.add_points(
        features=features,
        name='points',
        size=point_size,
        face_color=roi_color,
        symbol='disc',
        text=label_text
    )

    @points_layer.events.data.connect
    def update_points_layer():
        """
        Automatically assign a new label each time user adds a point.
        """
        current_labels = {label for label in points_layer.properties.get('label', [])}
        all_labels.update(current_labels)

        # Assign next label (increment)
        next_label = max(all_labels, default=first_label) + 1
        points_layer.feature_defaults['label'] = next_label

    update_points_layer()  # run once at start

    viewer.show(block=True)  # pauses code until user closes viewer

    if len(vars(points_layer)['_data']) == 0:
        # User didn't annotate anything → stop pipeline
        sys.exit("Stopping program: User saved image without masking.")

    layer = viewer.layers['image']
    m, M = layer.contrast_limits

    # Rescale intensities between 0–1
    rescaled = (layer.data - m) / (M - m)
    image = np.clip(rescaled, 0, 1)

    mask, _, layer_type = points_to_labels(points_layer, image_layer) 

    assert layer_type == 'Labels'  # ensure correct output type

    # Extract point coordinates and labels
    points_layer_data = points_layer.data
    points_layer_label = points_layer.properties["label"]

    # Remove points outside image boundaries
    indices_to_remove = np.where(
        (points_layer_data[:, 0] >= image_layer.data.shape[0]) |
        (points_layer_data[:, 1] >= image_layer.data.shape[1]) |
        (points_layer_data[:, 0] < 0) |
        (points_layer_data[:, 1] < 0)
    )[0]

    points_layer_data = np.delete(points_layer_data, indices_to_remove, axis=0)
    points_layer_label = np.delete(points_layer_label, indices_to_remove)

    # Convert points into seeds
    seeds = util.label_points(points_layer_data, image_layer.data.shape)

    # Distance transform → expand regions from seeds
    distances = -ndi.distance_transform_edt(mask)

    # Watershed grows regions from seeds inside mask
    mask = segmentation.watershed(distances, seeds, mask=mask)

    features = {'label': np.array(points_layer_label, dtype=int)}

    # Map internal labels → user-defined labels
    updated_values = {0: 0}  # background stays 0
    updated_values.update({i+1: value for i, value in enumerate(features['label'])})

    mask = np.vectorize(updated_values.get, otypes=[mask.dtype])(mask)

    return image, mask