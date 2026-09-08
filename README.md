# Nanopore Studio v2.0

A streamlined Streamlit interface for BIC-guided, two-dimensional Gaussian mixture modelling of nanopore events.

## Start

Install the packages in `requirements.txt`, then run:

    streamlit run app.py

For Streamlit Community Cloud, upload the three Python files, `requirements.txt`, and the `.streamlit` configuration folder. Set `app.py` as the entry point.

## Input files

`dataset.npz` is required. The default NanoSense mapping is X[:,0] = event blockade ΔI (nA) and X[:,4] = dwell time (seconds). Change the dwell unit selector if your file uses another unit.

`event_data.npz` is optional and provides original event IDs and synchronized event exports. `event_fitting.npz` is optional and enables segment/topology analysis. Companion files are checked against the dataset before use.

## Workflow

Inspect the raw event cloud, fit candidate K values using BIC, select a model, and examine its cluster statistics. The original 1D cutoff and separate forced-K=2 analysis have been removed. K=2 remains a valid candidate in the BIC comparison.

The fitted features are standardized log10(dwell_ms) and ΔI. Full-covariance GMMs use random_state=0, n_init=10, and reg_covar=1e-6, preserving the v1.4.8 BIC algorithm.

The main table reports original-event medians and arithmetic means, not GMM centres. Optional topology reclassification uses the preserved exploratory adjacent-segment similarity rule. Its threshold does not affect the GMM assignments.

## Exports

The ZIP contains cluster summary, BIC results, event mapping with posterior probabilities, settings, and excluded-event information. Each cluster folder contains the available synchronized NPZ source types; complete three-file inputs produce complete three-file cluster exports.

The exported source arrays are not changed apart from filtering and reindexing event identifiers. Original row indices and IDs are recorded separately.

## Scientific cautions

Gaussian components are statistical descriptions, not proof of distinct molecular mechanisms. A BIC minimum at the largest tested K does not establish the exact component count. Segment-derived linear-like, folded-like, and complex classes are putative. Report missing segment counts and validate physical interpretations using raw traces and independent recordings.

The similarity metric is 100 × min(|a|, |b|) / max(|a|, |b|), with duration-weighted adjacent merging. It is an exploratory rule retained from v1.4.8 and should not be described as the native NanoSense algorithm without verification.

## Security

Load NPZ files only from trusted sources: NanoSense archives may contain Python object arrays. The app processes data on its hosting Streamlit server and never overwrites uploaded source files.
