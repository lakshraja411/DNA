# Nanopore Analysis — Scientific edition v2.2

## Run
Install requirements.txt, then run `streamlit run app.py`.
For Streamlit Community Cloud, upload the project files and set the main file to app.py.

## Changes in v2.2
- Restored the plotting behaviour from app(20260908-040321).py: filled overlapping histograms, original Silverman KDE, original smooth 2D KDE, masked magma density, and selected-K scatter boundaries.
- The density plot offers Classic dark and White appearances. Other figures retain the white scientific style.
- The figure editor still allows custom titles, axis labels, font sizes, dimensions, legends and limits. PNG, PDF and SVG exports remain available.
- Removed the visible covariance/initialization/random-seed caption; fitting settings are unchanged.
- Replaced raw dataset column indices with descriptive names in the selectors and plot labels.

## Dataset columns
The current NanoSense layout is:
0 Current drop, ΔI (nA)
1 Full width at half maximum, FWHM (s)
2 Blockade height at FWHM (nA)
3 Event area (stored units; confirm the area convention in the source)
4 Dwell time (s)
5 Skewness
6 Kurtosis
7 Baseline current (nA)
8 Event time (s)

Unknown extra columns are labelled as unlabelled dataset columns rather than assigned an invented physical meaning. The original X-array indices and values are not changed.

## Analysis
The existing BIC-guided 2D GMM and selected-K workflow is preserved. There is no separate 1D cutoff, forced-K2 or topology section. Dataset is required; event data and event fitting are optional. Complete three-file inputs can be exported as synchronized, reindexed NPZ populations. Keep your previous deployment as a backup until you have checked this release with real data.
