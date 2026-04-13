# Event & Spike Detection

Activity of the cells can be detected using two methods:

* Peak detection on ΔF/F traces
* Based on deconvolved signals (OASIS)

## Event detection (peak detection)
With peak detection, different thresholding parameters can be used to detect events in the fluorescence trace. 

To facilitate parameter tuning, an interactive testing workflow is provided:

* Test settings on random image
    This option samples a random recording from the input dataset. The user can:
    * Visualize baseline estimation and detected events using line plots
    * Play the calcium imaging video with detected events overlaid
    * Adjust peak detection parameters in real time

This exploratory step allows the user to optimize detection settings without saving results.

* Run

    Executes the full single-cell analysis pipeline using the selected parameters.
    The results are saved in a Quantification folder within the specified project directory and can be accessed via the file system.

## Spike deconvolution
To improve temporal resolution, fluorescence traces can be deconvolved using the OASIS algorithm. This process estimates underlying spike trains from the slower calcium signals.

Deconvolution is particularly useful for:
* Resolving closely spaced events
* Approximating neuronal firing activity
* Enabling downstream network analysis


## Output

The analysis generates the following outputs:

* Event times (per cell)
* Detected spike activity (deconvolved traces)
* Optional visualizations of activity traces and detected events

## Limitations

Calcium imaging signals are indirect proxies of neuronal firing. Event detection depends on parameter selection
Deconvolution provides an estimate of spike activity rather than exact spike timing