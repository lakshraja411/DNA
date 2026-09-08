# Nanopore Analysis — Scientific edition

A compact, white scientific interface for the existing BIC-guided nanopore GMM workflow.

## Install and run

Install the packages in requirements.txt, then run:

    streamlit run app.py

For Streamlit Community Cloud, upload app.py, analysis_core.py, plotting.py,
splitter_core.py, requirements.txt, and .streamlit/config.toml. Set the main
file path to app.py. Keep your previous deployment as a backup until the new
version has been checked with your real measurements.

## Analysis

Only dataset.npz is required. The default mapping is X[:,0] = ΔI (nA) and
X[:,4] = dwell time (seconds). Select the correct dwell unit and ΔI source.
event_data.npz and event_fitting.npz are optional. When supplied, they are
checked against the dataset before analysis.

The original BIC GMM algorithm is preserved: standardized [log10(dwell_ms), ΔI],
full covariance, 10 initializations, random_state=0, reg_covar=1e-6.
Models are compared for K=1 through the selected maximum; any tested K can be
chosen. No separate 1D or forced-K2 analysis is run.

The population table reports arithmetic means and medians of original
hard-assigned events, not transformed Gaussian centres. Clusters are ordered
by median original dwell. The optional companion files are not required for
basic clustering or dataset-only exports.

## Figures

All figures use white backgrounds, black axes, restrained colours, and
Matplotlib. Open Figure settings to edit the title, axis labels, size, font,
legend, grid, and axis limits. Export each figure as PNG (600 dpi), PDF, or
SVG. Figure settings do not change the model, event assignments, or statistics.
Raw and selected-population histograms, KDE curves, 2D density, GMM
classification, and BIC plotting are available. Histogram count, density,
share, bin width, and logarithmic-axis controls are explicit.

## Exports

The ZIP contains cluster summary, BIC results, all original-row assignments,
posterior probabilities, settings, an excluded-event log, and filtered NPZ
files for each cluster. When all three source files are supplied, the original
synchronized export routine is used. Dataset-only exports contain dataset
files and row mappings. Source files are never overwritten.

## Notes

Gaussian components are statistical descriptions, not confirmed molecular
transport mechanisms. A BIC minimum at the largest K tested may require a
wider candidate range. Event-level uncertainty is not a substitute for
replicate-level uncertainty. KDE smoothing and display sampling affect only
the figure.

Only open trusted NPZ archives because NanoSense object arrays require pickle
loading. This app does not make external analysis requests, but files uploaded
to a hosted Streamlit app are processed on that hosting service.
