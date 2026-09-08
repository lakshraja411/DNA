# 🧬 Nanopore Event Population Splitter

A small Streamlit app for separating **short-dwell** and **long-dwell** nanopore-event populations while keeping three NanoSense-style output files synchronized:

- `*.event_data.npz`
- `*.dataset*.npz`
- `*.event_fitting*.npz`

The app was built for dwell-time population analysis of DNA translocation events. It validates that the three uploaded files refer to the same events before any filtering is performed.

## What it does

For one matched dataset, the app:

1. Loads the three NPZ files.
2. Checks that event counts and event identities are synchronized.
3. Reads dwell time from `dataset['X'][:, 4]` in seconds and converts it to milliseconds for display.
4. Suggests a two-population cutoff using a 2-component Gaussian mixture model in `log10(dwell time)`.
5. Lets you manually change the cutoff after inspecting the dwell-time histogram.
6. Defines:
   - **SHORT:** dwell time ≤ cutoff
   - **LONG:** dwell time > cutoff
7. Exports both populations with synchronized `event_data`, `dataset`, and `event_fitting` files.
8. Re-numbers filtered event IDs from `0...N-1` consistently in all output files.
9. Saves `event_id_mapping.csv` so every filtered event can be traced back to its original event ID and row.

> **Important:** The automatic GMM cutoff is a starting suggestion, not a physical rule. For comparative experiments, inspect each dataset and use a scientifically justified population boundary.

## Repository structure

```text
dna-event-population-splitter/
├── app.py
├── splitter_core.py
├── requirements.txt
├── README.md
└── .gitignore
```

`app.py` contains the Streamlit user interface. `splitter_core.py` contains the file validation, GMM cutoff estimation, filtering, re-indexing, and ZIP-export logic, so the same engine can later be reused for batch processing.

## Run on a Mac / local computer

Open Terminal in the project folder.

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the app

```bash
streamlit run app.py
```

A browser window should open with the event splitter.

## Typical workflow

Upload the matching three files from one experiment, for example:

```text
LiCl_400mV_1kb_....event_data.npz
LiCl_400mV_1kb_....dataset.npz
LiCl_400mV_1kb_....event_fitting.npz
```

After validation, inspect the histogram and set a dwell-time boundary. Then click **Build filtered files** and download the ZIP.

The ZIP contains:

```text
SHORT/
    <dataset>_SHORT.event_data.npz
    <dataset>_SHORT.dataset.npz
    <dataset>_SHORT.event_fitting.npz

LONG/
    <dataset>_LONG.event_data.npz
    <dataset>_LONG.dataset.npz
    <dataset>_LONG.event_fitting.npz

event_id_mapping.csv
split_info.txt
```

## Put the project on GitHub

The `.gitignore` intentionally excludes all `.npz` and `.zip` files so raw/processed experimental data are not accidentally committed.

### Option A — GitHub website

1. Create a new empty repository on GitHub.
2. Extract this project ZIP.
3. Upload the files in the extracted folder to the repository.
4. Commit the files.

### Option B — Terminal

From inside the extracted project folder:

```bash
git init
git add .
git commit -m "Initial nanopore event population splitter"
git branch -M main
git remote add origin <YOUR-GITHUB-REPOSITORY-URL>
git push -u origin main
```

## Data safety

This repository is designed to contain the **analysis code, not the experimental NPZ files**. Before pushing to GitHub, you can check what will be uploaded with:

```bash
git status
```

Your `.npz` files should not appear in the staged files.

## Current scope

Version 1.0 separates **two dwell-time populations** from one dataset at a time.

Useful future extensions include:

- batch processing across salts and voltages;
- three or more event populations;
- 2-D filtering using dwell time plus current blockade/amplitude;
- saving population labels back into a master results table;
- comparison plots of SHORT vs LONG populations across conditions.
