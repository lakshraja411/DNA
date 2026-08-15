"""Core functions for synchronized nanopore event population splitting.

The functions here are intentionally independent of Streamlit so they can be
reused later by a batch-processing script, notebook, CLI, or tests.
"""

from __future__ import annotations

import copy
import io
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


EVENT_KEY_RE = re.compile(r"^(EVENT_DATA|SEGMENT_INFO|EVENT_ANALYSIS)_(\d+)(.*)$")


def load_npz_bytes(data: bytes) -> Dict[str, np.ndarray]:
    """Load an NPZ byte string into a regular dictionary."""
    with np.load(io.BytesIO(data), allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def load_npz_path(path: str | Path) -> Dict[str, np.ndarray]:
    """Load an NPZ file from disk into a regular dictionary."""
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def validate_inputs(
    event_data: Dict[str, np.ndarray],
    dataset: Dict[str, np.ndarray],
    event_fitting: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Validate synchronization among the three NanoSense-style files.

    Returns
    -------
    dwell_s
        Event dwell times in seconds, taken from dataset['X'][:, 4].
    original_ids
        Original event_id values stored in event_data.
    notes
        Human-readable validation messages.
    """
    required_event_data = {"sampling_rate", "events"}
    required_dataset = {"settings", "X"}

    missing_event_data = required_event_data.difference(event_data)
    missing_dataset = required_dataset.difference(dataset)
    if missing_event_data:
        raise ValueError(
            f"event_data file is missing keys: {sorted(missing_event_data)}"
        )
    if missing_dataset:
        raise ValueError(f"dataset file is missing keys: {sorted(missing_dataset)}")

    events = np.asarray(event_data["events"], dtype=object)
    X = np.asarray(dataset["X"])

    if X.ndim != 2 or X.shape[1] < 5:
        raise ValueError(
            "dataset['X'] must be a 2-D array with at least five columns. "
            "Column index 4 is expected to contain event dwell time in seconds."
        )

    n_events = len(events)
    if X.shape[0] != n_events:
        raise ValueError(
            f"Event-count mismatch: event_data has {n_events} events, "
            f"but dataset['X'] has {X.shape[0]} rows."
        )

    original_ids = np.empty(n_events, dtype=int)
    widths_from_event_data = np.empty(n_events, dtype=float)
    for row_index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(
                f"event_data['events'][{row_index}] is not a dictionary."
            )
        if "start_time" not in event or "end_time" not in event:
            raise ValueError(
                f"event_data['events'][{row_index}] lacks start_time/end_time."
            )
        original_ids[row_index] = int(event.get("event_id", row_index))
        widths_from_event_data[row_index] = float(
            event["end_time"] - event["start_time"]
        )

    dwell_s = np.asarray(X[:, 4], dtype=float)
    if np.any(~np.isfinite(dwell_s)) or np.any(dwell_s <= 0):
        raise ValueError(
            "The dwell-time column contains non-finite or non-positive values."
        )

    notes: List[str] = []
    sampling_rate = float(np.asarray(event_data["sampling_rate"]).item())
    atol = max(1e-12, 0.51 / sampling_rate)
    close_widths = np.isclose(
        dwell_s, widths_from_event_data, rtol=1e-7, atol=atol
    )
    if not np.all(close_widths):
        bad = np.flatnonzero(~close_widths)
        raise ValueError(
            "Dwell-time mismatch between dataset and event_data for "
            f"{len(bad)} events (first mismatch at row {int(bad[0])})."
        )
    notes.append("dataset dwell times match event_data start/end widths")

    fitting_ids = sorted(
        {
            int(match.group(2))
            for key in event_fitting
            if (match := EVENT_KEY_RE.match(key)) is not None
        }
    )
    expected_ids = list(range(n_events))
    if fitting_ids != expected_ids:
        raise ValueError(
            "event_fitting event IDs are not the expected contiguous range "
            "0..N-1. The splitter stops here rather than risk mixing events."
        )

    mismatches = []
    for i in range(n_events):
        key = f"EVENT_ANALYSIS_{i}"
        if key not in event_fitting:
            raise ValueError(f"event_fitting is missing {key}.")
        if not np.allclose(
            np.asarray(event_fitting[key]),
            X[i],
            equal_nan=True,
            rtol=1e-7,
            atol=1e-12,
        ):
            mismatches.append(i)
            if len(mismatches) >= 5:
                break

    if mismatches:
        raise ValueError(
            "dataset rows do not match EVENT_ANALYSIS rows in event_fitting. "
            f"First mismatches: {mismatches}."
        )
    notes.append("dataset rows match EVENT_ANALYSIS_i in event_fitting")

    return dwell_s, original_ids, notes


def auto_cutoff_gmm(dwell_ms: np.ndarray) -> Tuple[float, dict]:
    """Suggest a two-population cutoff with a GMM in log10 dwell time.

    The returned cutoff is a mathematical starting point, not a physical claim.
    Users should inspect the plotted distributions and choose a scientifically
    defensible threshold for each experimental dataset.
    """
    dwell_ms = np.asarray(dwell_ms, dtype=float)
    if len(dwell_ms) < 4:
        raise ValueError("At least four events are required for a 2-component fit.")
    if np.any(~np.isfinite(dwell_ms)) or np.any(dwell_ms <= 0):
        raise ValueError("Dwell times must be finite and positive.")

    log_dwell = np.log10(dwell_ms).reshape(-1, 1)
    model = GaussianMixture(n_components=2, random_state=0, n_init=10)
    model.fit(log_dwell)

    component_order = np.argsort(model.means_.ravel())
    grid = np.linspace(log_dwell.min(), log_dwell.max(), 20000).reshape(-1, 1)
    probabilities = model.predict_proba(grid)
    p_short = probabilities[:, component_order[0]]

    mean_lo, mean_hi = sorted(model.means_.ravel())
    between = (grid[:, 0] >= mean_lo) & (grid[:, 0] <= mean_hi)
    candidates = np.flatnonzero(between)
    if len(candidates):
        crossing_index = candidates[
            np.argmin(np.abs(p_short[candidates] - 0.5))
        ]
    else:
        crossing_index = int(np.argmin(np.abs(p_short - 0.5)))

    cutoff_ms = float(10 ** grid[crossing_index, 0])
    means_ms = 10 ** model.means_.ravel()[component_order]
    weights = model.weights_.ravel()[component_order]

    details = {
        "short_geometric_mean_ms": float(means_ms[0]),
        "long_geometric_mean_ms": float(means_ms[1]),
        "short_weight": float(weights[0]),
        "long_weight": float(weights[1]),
    }
    return cutoff_ms, details


def _npz_bytes(arrays: Dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def build_filtered_files(
    selected_indices: np.ndarray,
    label: str,
    event_data: Dict[str, np.ndarray],
    dataset: Dict[str, np.ndarray],
    event_fitting: Dict[str, np.ndarray],
) -> Tuple[Dict[str, bytes], pd.DataFrame]:
    """Build synchronized and re-indexed NPZ outputs for one population."""
    selected_indices = np.asarray(selected_indices, dtype=int)
    selected_indices.sort()

    if selected_indices.size == 0:
        raise ValueError(f"The {label} population contains no events.")

    n_total = len(event_data["events"])
    if selected_indices.min() < 0 or selected_indices.max() >= n_total:
        raise IndexError("Selected event index lies outside the source dataset.")

    old_events = np.asarray(event_data["events"], dtype=object)
    new_events = []
    mapping_rows = []
    row_to_new_id = {}

    for new_id, old_row_index in enumerate(selected_indices):
        old_row_index = int(old_row_index)
        old_event = old_events[old_row_index]
        original_event_id = int(old_event.get("event_id", old_row_index))

        copied_event = copy.deepcopy(old_event)
        copied_event["event_id"] = int(new_id)
        new_events.append(copied_event)
        row_to_new_id[old_row_index] = int(new_id)

        mapping_rows.append(
            {
                "population": label,
                "new_event_id": int(new_id),
                "original_event_id": original_event_id,
                "original_row_index": old_row_index,
                "start_time_s": float(old_event["start_time"]),
                "end_time_s": float(old_event["end_time"]),
                "dwell_time_ms": 1000.0
                * float(old_event["end_time"] - old_event["start_time"]),
            }
        )

    filtered_event_data = {k: v for k, v in event_data.items() if k != "events"}
    filtered_event_data["events"] = np.asarray(new_events, dtype=object)

    filtered_dataset = {k: v for k, v in dataset.items() if k != "X"}
    filtered_dataset["X"] = np.asarray(dataset["X"])[selected_indices].copy()

    filtered_fitting: Dict[str, np.ndarray] = {}
    for key, value in event_fitting.items():
        if EVENT_KEY_RE.match(key) is None:
            filtered_fitting[key] = value

    selected_set = set(map(int, selected_indices))
    for key, value in event_fitting.items():
        match = EVENT_KEY_RE.match(key)
        if match is None:
            continue
        prefix, old_row_str, suffix = match.groups()
        old_row_index = int(old_row_str)
        if old_row_index in selected_set:
            new_id = row_to_new_id[old_row_index]
            filtered_fitting[f"{prefix}_{new_id}{suffix}"] = value

    return (
        {
            "event_data": _npz_bytes(filtered_event_data),
            "dataset": _npz_bytes(filtered_dataset),
            "event_fitting": _npz_bytes(filtered_fitting),
        },
        pd.DataFrame(mapping_rows),
    )


def clean_stem(filename: str) -> str:
    """Remove known NanoSense suffixes from an uploaded filename."""
    name = Path(filename).name
    patterns = [
        r"\.event_data\.npz$",
        r"\.dataset(?:\(\d+\))?\.npz$",
        r"\.event_fitting(?:\(\d+\))?\.npz$",
    ]
    for pattern in patterns:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)
    return name


def make_zip(
    stem: str,
    cutoff_ms: float,
    short_files: Dict[str, bytes],
    long_files: Dict[str, bytes],
    short_map: pd.DataFrame,
    long_map: pd.DataFrame,
) -> bytes:
    """Package both populations and their event-ID mapping into one ZIP."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for population, files in (("SHORT", short_files), ("LONG", long_files)):
            zf.writestr(
                f"{population}/{stem}_{population}.event_data.npz",
                files["event_data"],
            )
            zf.writestr(
                f"{population}/{stem}_{population}.dataset.npz",
                files["dataset"],
            )
            zf.writestr(
                f"{population}/{stem}_{population}.event_fitting.npz",
                files["event_fitting"],
            )

        mapping = pd.concat([short_map, long_map], ignore_index=True)
        zf.writestr("event_id_mapping.csv", mapping.to_csv(index=False).encode("utf-8"))
        zf.writestr(
            "split_info.txt",
            (
                "Nanopore dwell-time population split\n"
                f"Cutoff: {cutoff_ms:.9g} ms\n"
                "SHORT: dwell <= cutoff\n"
                "LONG: dwell > cutoff\n"
                f"Short events: {len(short_map)}\n"
                f"Long events: {len(long_map)}\n\n"
                "Each population contains synchronized event_data, dataset, and "
                "event_fitting NPZ files.\n"
                "Event IDs are re-numbered 0..N-1 within each population.\n"
                "event_id_mapping.csv preserves original event IDs and row indices.\n"
            ).encode("utf-8"),
        )
    return buffer.getvalue()
