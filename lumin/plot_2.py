# -------------------- IMPORTS --------------------
import matplotlib.pyplot as plt                     # plotting
from skimage.segmentation import mark_boundaries    # draw outlines on images
import os                                           # file handling
import numpy as np                                  # numerical operations
from skimage import io                              # image reading
from cv2 import circle                              # draw circles on images
from skimage.color import gray2rgb                  # convert grayscale → RGB
from skimage.util import img_as_ubyte               # convert image format
import pandas as pd                                 # dataframes
from typing import Literal                          # type hints
from sklearn.preprocessing import minmax_scale      # normalize data
import seaborn as sns                               # statistical plots
import matplotlib.lines as mlines                   # custom legend elements
from pca import pca                                 # PCA library
 
import warnings
# Ignore matplotlib warning when RGB values exceed valid range
warnings.filterwarnings("ignore", message="Clipping input data to the valid range for imshow with RGB data")
 
 
# -------------------- COLOR PALETTE --------------------
def create_palette(unique_stimulations):
    """
    Create a dictionary mapping each stimulation condition to a color (hex).
    """
    colors = sns.color_palette(n_colors=len(unique_stimulations))  # generate colors
    # Convert RGB (0-1) → HEX format (#RRGGBB)
    hex_colors = ["#{:02X}{:02X}{:02X}".format(int(color[0]*255),
                                               int(color[1]*255),
                                               int(color[2]*255))
                  for color in colors]
    return dict(zip(unique_stimulations, hex_colors))  # map condition → color
 
 
# -------------------- SEGMENTATION PLOT --------------------
def segmentation(image: np.ndarray, mask: np.ndarray,
                 title_image: str = 'Image',
                 title_mask: str = 'Mask Outline',
                 output_path: str = None,
                 file_name: str = 'segmentation_plot'):
 
    """
    Overlay segmentation mask boundaries on an image and save it.
    """
 
    with plt.rc_context({"figure.dpi": (350), 'figure.figsize':(10, 10)}):
        fig, ax = plt.subplots()
 
        # Draw mask boundaries on top of image (pink color)
        ax.imshow(mark_boundaries(image, mask,
                                 color=(1.0, 0.0, 1.0),
                                 mode='thick'),
                  cmap='gray')
 
        ax.axis("off")  # remove axes
 
        # Save figure if path provided
        plt.savefig(os.path.join(output_path, f'{file_name}.pdf'),
                    bbox_inches='tight')
        plt.close('all')
 
 
# -------------------- OVERLAY LABELS --------------------
def overlay_labels(image: np.ndarray, mask: np.ndarray,
                   cell_properties_df: pd.DataFrame,
                   mask_nuclear: np.ndarray = None,
                   output_path: str = None,
                   file_name: str = 'overlay_plot'):
 
    """
    Overlay segmentation + cell labels (numbers) on image.
    """
 
    image_rgb = gray2rgb(img_as_ubyte(image))  # convert grayscale → RGB
 
    with plt.rc_context({"figure.dpi": (350), 'figure.figsize':(10, 10)}):
 
        # Get centroids (cell centers) and labels from dataframe
        centroids = cell_properties_df[['centroid-0','centroid-1']].values.tolist()
        labels = cell_properties_df['label'].values.tolist()
 
        fig, ax = plt.subplots()
 
        # Color segmented regions
        mask_bool = mask.astype(bool)
        roi_rgb = np.array([102, 171, 217], dtype=np.uint8)  # blue-ish color
        image_rgb[mask_bool] = roi_rgb
 
        # If nuclear mask exists → overlay both masks
        if mask_nuclear is not None:
            img = mark_boundaries(
                mark_boundaries(image_rgb, mask, color=(38, 86, 245), mode='thick'),
                mask_nuclear, color=(1.0, 0.0, 1.0), mode='thick'
            )
        else:
            img = mark_boundaries(image_rgb, mask, color=(38, 86, 245), mode='thick')
 
            # Add cell numbers on top of each ROI
            kwargs = {'horizontalalignment': 'center', 'fontsize': 6}
            for i in range(len(centroids)):
                ax.text(centroids[i][1], centroids[i][0]+6,
                        str(labels[i]),
                        color='#eb1717', **kwargs)
 
        ax.imshow(img, cmap='gray')
        plt.axis('off')
 
        if output_path is not None:
            plt.savefig(os.path.join(output_path, f'{file_name}.pdf'),
                        bbox_inches='tight')
            plt.close('all')
        else:
            return img
 
 
# -------------------- OVERLAID TRACES --------------------
def overlaid_traces(cell_properties_df: pd.DataFrame,
                    trace: str = None,
                    mean: bool = False):
 
    """
    Plot all calcium traces on top of each other.
    Optionally overlay the mean trace.
    """
 
    with plt.rc_context({"figure.dpi": 350}):
        fig, ax = plt.subplots(figsize=(15,10))
 
        # Plot each cell trace
        for _, cell in cell_properties_df.iterrows():
            ax.plot(cell[trace], linewidth=1)
 
        # Plot mean trace if requested
        if mean:
            ax.plot(np.array(cell_properties_df[trace].tolist()).mean(axis=1),
                    color='black', linewidth=3)
 
    return ax
 
 
# -------------------- TWO-GROUP TRACE PLOT --------------------
def overlaid_traces_two_groups(cell_properties_df: pd.DataFrame = None,
                               trace: str = None,
                               output_path: str = None,
                               mean: bool = False,
                               control_condition: str = None,
                               treatment_condition: str = None,
                               start_frame: int = None,
                               end_frame: int = None,
                               stimulation_frame: int = None,
                               ax=None,
                               palette: dict = None,
                               imaging_interval: float = None,
                               kcl_frame: int = None):
 
    """
    Compare traces between control vs treatment groups.
    Shows individual traces + mean traces.
    """
 
    with plt.rc_context({"figure.dpi": 350}):
 
        if ax is None:
            fig, ax = plt.subplots()
 
        # Create color palette if not provided
        unique_stimulations = cell_properties_df["stimulation"].cat.categories
        if palette is None:
            palette = create_palette(unique_stimulations)
 
        # Plot all traces (thin lines)
        ax.plot(np.array(cell_properties_df[cell_properties_df.stimulation == treatment_condition][trace].tolist()).T,
                color=palette[treatment_condition], lw=0.4, alpha=0.5)
 
        ax.plot(np.array(cell_properties_df[cell_properties_df.stimulation == control_condition][trace].tolist()).T,
                color=palette[control_condition], lw=0.4, alpha=0.5)
 
        # Plot mean traces (thicker)
        ax.plot(np.mean(np.array(cell_properties_df[cell_properties_df.stimulation == treatment_condition][trace].tolist()).T, axis=1),
                lw=2)
 
        ax.plot(np.mean(np.array(cell_properties_df[cell_properties_df.stimulation == control_condition][trace].tolist()).T, axis=1),
                lw=2)
 
        # Highlight regions (baseline / stimulation)
        ax.axvspan(start_frame, end_frame, color='#555555', alpha=0.48)
        ax.axvspan(0, stimulation_frame, color='lightgray', alpha=0.5)
 
        # Optional vertical line (e.g. KCl addition)
        if kcl_frame is not None:
            ax.axvline(x=kcl_frame, color='red', linestyle='--')
 
        # Axis labels
        ax.set_ylabel(r"Ca$^{2+}$ signal ($\Delta F/F_0$)")
        ax.set_xlabel('Time (s)')
 
        return ax
 
 
# -------------------- CELLWISE TRACES --------------------
def cellwise_traces(cell_properties_df: pd.DataFrame,
                    trace: str = None,
                    baseline: bool = False,
                    spikes: bool = False,
                    spikes_mode: Literal["all", "filtered"] = "all",
                    output_path: str = None,
                    smoothing: bool = False):
 
    """
    Plot traces per cell (max 20 per figure).
    Optionally show:
    - baseline
    - spikes
    - smoothed trace
    """
 
    plot_list = []
    cell_counter = 0
    total_cells = cell_properties_df.shape[0]
 
    while cell_counter < total_cells:
 
        # Plot up to 20 cells per figure
        cells_in_chunk = min(20, total_cells - cell_counter)
        chunk_df = cell_properties_df.iloc[cell_counter:cell_counter + cells_in_chunk]
 
        fig, ax = plt.subplots(nrows=cells_in_chunk, figsize=(15, 20), sharex=True)
 
        if cells_in_chunk == 1:
            ax = [ax]
 
        for i, (_, cell) in enumerate(chunk_df.iterrows()):
            ax_plot = ax[i]
 
            # Main trace
            ax_plot.plot(cell[trace], color='#1c1fb0', linewidth=2)
 
            # Optional smoothing
            if smoothing:
                ax_plot.plot(cell['dff_smoothed'], color='magenta', linewidth=1)
 
            # Optional baseline
            if baseline and 'baseline' in cell:
                ax_plot.plot(cell['baseline'], color='#fc4a2b', alpha=0.7)
 
            # Label each subplot with cell ID
            ax_plot.set_ylabel(f"{cell['label']}.", rotation=0)
 
            # Optional spike markers
            if spikes and isinstance(cell.get('peak_location'), (list, np.ndarray)):
                for peak in cell['peak_location']:
                    ax_plot.axvline(x=peak, linestyle='--', color='#fca02b')
 
        plot_list.append(fig)
 
        # Save figure if needed
        if output_path is not None:
            plt.savefig(os.path.join(output_path, f"{(cell_counter // 20) + 1}.pdf"),
                        dpi=350, bbox_inches="tight")
            plt.close('all')
 
        cell_counter += cells_in_chunk
 
    return plot_list
 
 
# -------------------- OVERLAY EVENTS --------------------
def overlay_events(cell_properties_df: pd.DataFrame):
    """
    Create a mask showing where spikes/events occur in time + space.
    """
 
    stack = io.imread(cell_properties_df['filepath'].values[0])  # load image stack
    mask = np.zeros_like(stack, dtype=np.uint16)
 
    for _, row in cell_properties_df.iterrows():
        locations = row['peak_location']
        centroid = (int(row['centroid-1']), int(row['centroid-0']))
        area = row.get('area', row.get('cell_area'))
 
        if len(locations) > 0:
            radius = int(np.sqrt(area / np.pi))  # approximate cell radius
 
            for frame_idx in locations:
                # Draw circle around event for a few frames
                for offset in [-1, 0, 1, 2]:
                    idx = frame_idx + offset
                    if 0 <= idx < mask.shape[0]:
                        circle(mask[idx], centroid, int(radius * 1.5),
                               color=100, thickness=2)
 
    return stack, mask
 
# -------------------- BEESWARM --------------------
def beeswarm(cell_properties_df, y, x, palette, hue=None, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
 
    sns.swarmplot(
        data=cell_properties_df,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        dodge=True if hue else False,
        ax=ax
    )
 
    return ax
 
# -------------------- HEATMAP --------------------
def cluster_heatmap(cell_properties_df, cbar=True):
    data = cell_properties_df.select_dtypes(include=[np.number])
 
    ax = sns.clustermap(data, cmap='viridis', cbar=cbar)
    return ax
 
# -------------------- BARPLOT WITH STRIP --------------------
def all_conditions_barplot(dataframe, palette, xcolumn, ycolumn, hue=None):
    plt.figure(figsize=(6, 4))
 
    ax = sns.barplot(
        data=dataframe,
        x=xcolumn,
        y=ycolumn,
        hue=hue,
        palette=palette
    )
 
    sns.stripplot(
        data=dataframe,
        x=xcolumn,
        y=ycolumn,
        hue=hue,
        dodge=True if hue else False,
        color='black',
        size=3,
        ax=ax
    )
 
    handles, labels = ax.get_legend_handles_labels()
 
    if hue:
        ax.legend(handles[:len(palette)], labels[:len(palette)])
    else:
        ax.legend().remove()
 
    return ax
 
def two_conditions_barplot(dataframe, palette, x, y, hue=None):
    plt.figure(figsize=(6, 4))
 
    ax = sns.barplot(
        data=dataframe,
        x=x,
        y=y,
        hue=hue,
        palette=palette
    )
    sns.stripplot(
        data=dataframe,
        x=x,
        y=y,
        hue=hue,
        dodge=True if hue else False,
        color='black',
        size=3,
        ax=ax
    )
 
    return ax
# -------------------- PCA BIPLOT --------------------
def biplot(cell_properties_df, palette):
    features = cell_properties_df.select_dtypes(include=[np.number])
    pca_model = pca(n_components=2)
    pcs = pca_model.fit_transform(features)
 
    df_pca = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
    df_pca['stimulation'] = cell_properties_df['stimulation'].values
 
    fig, ax = plt.subplots(figsize=(5, 5))
 
    sns.scatterplot(
        data=df_pca,
        x='PC1',
        y='PC2',
        hue='stimulation',
        palette=palette,
        ax=ax
    )
 
    return ax, cell_properties_df
 
# -------------------- PCA PROPERTY --------------------
def pca_property(cell_properties_df, color_by):
    features = cell_properties_df.select_dtypes(include=[np.number])
    pca_model = pca(n_components=2)
    pcs = pca_model.fit_transform(features)
 
    df_pca = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
    df_pca[color_by] = cell_properties_df[color_by].values
 
    fig, ax = plt.subplots(figsize=(5, 5))
 
    sns.scatterplot(
        data=df_pca,
        x='PC1',
        y='PC2',
        hue=color_by,
        palette='viridis',
        ax=ax
    )
 
    return ax
 
# -------------------- SYNCHRONIZATION --------------------
def compute_synchronization(dff_array, save_path=None):
    """
    Compute synchronization matrix using correlation.
 
    Parameters
    ----------
    dff_array : list/array of 1D traces (shape: num_cells x timepoints)
    """
    dff_array = np.array(dff_array)
 
    n_cells = dff_array.shape[0]
    sync_matrix = np.zeros((n_cells, n_cells))
 
    for i in range(n_cells):
        for j in range(n_cells):
            if dff_array.shape[1] == dff_array.shape[1]:
                sync_matrix[i, j] = np.corrcoef(dff_array[i], dff_array[j])[0, 1]
 
    sync_matrix = np.nan_to_num(sync_matrix)
 
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.figure()
        sns.heatmap(sync_matrix, cmap='viridis')
        plt.title("Synchronization")
        plt.savefig(os.path.join(save_path, "synchronization_heatmap.png"))
        plt.close()
 
        np.savetxt(os.path.join(save_path, "synchronization_matrix.csv"),
                   sync_matrix, delimiter=",")
 
    return sync_matrix
 
# -------------------- CORRELATION --------------------
def correlation(cell_properties_df, trace_col, output_path):
    detrended_seq = np.array(cell_properties_df[trace_col].tolist())
 
    obs_num, length = detrended_seq.shape
    heat_mat = np.zeros((obs_num, obs_num))
    max_lag = min(50, length // 2)
 
    for i in range(obs_num):
        for j in range(i + 1):
            max_cor = -1
 
            for lag in range(max_lag + 1):
                A = detrended_seq[i, :length - lag]
                B = detrended_seq[j, lag:]
 
                cor = np.sum(
                    (A - np.mean(A)) * (B - np.mean(B)) /
                    (np.std(A) * np.std(B))
                ) / (length - lag - 1)
 
                max_cor = max(cor, max_cor)
 
            for lag in range(1, max_lag + 1):
                A = detrended_seq[i, lag:]
                B = detrended_seq[j, :length - lag]
 
                cor = np.sum(
                    (A - np.mean(A)) * (B - np.mean(B)) /
                    (np.std(A) * np.std(B))
                ) / (length - lag - 1)
 
                max_cor = max(cor, max_cor)
 
            heat_mat[i][j] = max_cor
            heat_mat[j][i] = max_cor
 
    plt.figure()
    sns.heatmap(heat_mat, cmap='jet', vmin=-1, vmax=1)
    plt.title("Correlation")
    plt.savefig(os.path.join(output_path, "correlation.png"))
    plt.close()
 
    np.savetxt(os.path.join(output_path, "correlation.csv"), heat_mat, delimiter=",")
 
    return heat_mat