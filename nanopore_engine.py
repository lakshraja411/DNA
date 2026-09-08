"""Analysis engine for Nanopore Studio.

The GMM and adjacent-segment reclassification algorithms are preserved
from the user's v1.4.8 release. The interface and optional-file handling
are new. Source NPZ files are never modified.
"""
from __future__ import annotations
import copy
import io
import json
import re
import zipfile
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from splitter_core import (
    EVENT_KEY_RE,
    build_filtered_files,
    load_npz_bytes,
)


def derive_event_metrics(event_fitting: dict[str, np.ndarray], n_events: int):
    """Derive useful per-event plotting metrics from event_fitting."""

    peak_segment_delta_i = np.full(n_events, np.nan, dtype=float)
    weighted_segment_delta_i = np.full(n_events, np.nan, dtype=float)
    number_of_segments = np.full(n_events, np.nan, dtype=float)

    for i in range(n_events):

        diff_key = f"SEGMENT_INFO_{i}_segment_mean_diffs"
        width_key = f"SEGMENT_INFO_{i}_segment_widths_time"
        n_key = f"SEGMENT_INFO_{i}_number_of_segments"

        # ----------------------------------------------------
        # PEAK SEGMENT ΔI
        # ----------------------------------------------------

        if diff_key in event_fitting:

            diffs = np.asarray(
                event_fitting[diff_key],
                dtype=float,
            ).ravel()

            diffs = diffs[np.isfinite(diffs)]

            if diffs.size:
                peak_segment_delta_i[i] = float(np.max(diffs))

        # ----------------------------------------------------
        # TIME-WEIGHTED ΔI
        # ----------------------------------------------------

        if diff_key in event_fitting and width_key in event_fitting:

            diffs = np.asarray(
                event_fitting[diff_key],
                dtype=float,
            ).ravel()

            widths = np.asarray(
                event_fitting[width_key],
                dtype=float,
            ).ravel()

            n_pair = min(
                len(diffs),
                len(widths),
            )

            diffs = diffs[:n_pair]
            widths = widths[:n_pair]

            valid = (
                np.isfinite(diffs)
                & np.isfinite(widths)
                & (widths > 0)
            )

            if np.any(valid):

                total_width = float(
                    np.sum(widths[valid])
                )

                if total_width > 0:

                    weighted_segment_delta_i[i] = float(
                        np.sum(
                            diffs[valid]
                            * widths[valid]
                        )
                        / total_width
                    )

        # ----------------------------------------------------
        # NUMBER OF SEGMENTS
        # ----------------------------------------------------

        if n_key in event_fitting:

            value = np.asarray(
                event_fitting[n_key],
                dtype=float,
            ).ravel()

            if value.size and np.isfinite(value[0]):

                number_of_segments[i] = float(
                    value[0]
                )

    return {

        "Peak segment ΔI":
            peak_segment_delta_i,

        "Time-weighted segment ΔI":
            weighted_segment_delta_i,

        "Number of segments":
            number_of_segments,
    }

def _segment_similarity_percent(a: float, b: float) -> float:
    """Return blockade-amplitude similarity on a bounded 0..100% scale.

    Similarity = 100 * min(|a|, |b|) / max(|a|, |b|).
    Therefore 100% means identical magnitudes. The absolute values are used so
    the result is independent of the sign convention used for segment_mean_diffs.
    """

    a = abs(float(a))
    b = abs(float(b))

    high = max(a, b)
    low = min(a, b)

    if high <= 1e-15:
        return 100.0

    return 100.0 * low / high

def reclassify_segment_counts_by_similarity(
    event_fitting: dict[str, np.ndarray],
    n_events: int,
    similarity_threshold_percent: float,
) -> np.ndarray:
    """Recalculate an effective segment count after merging similar neighbours.

    Adjacent fitted segments are compared in their original temporal order using
    the magnitude of SEGMENT_INFO_i_segment_mean_diffs. Two neighbouring segment
    groups are merged when their amplitude similarity is greater than or equal to
    ``similarity_threshold_percent``. If segment widths are available, merged
    amplitudes are duration-weighted; otherwise equal weights are used.

    This is a post-processing sensitivity analysis only. It does not change the
    NanoSense fit, GMM cluster assignment, or exported source event data.
    """

    threshold = float(np.clip(similarity_threshold_percent, 0.0, 100.0))
    reclassified = np.full(n_events, np.nan, dtype=float)

    for i in range(n_events):
        diff_key = f"SEGMENT_INFO_{i}_segment_mean_diffs"
        width_key = f"SEGMENT_INFO_{i}_segment_widths_time"
        n_key = f"SEGMENT_INFO_{i}_number_of_segments"

        # If the fitting file explicitly says this is a one-segment event, no
        # amplitude-comparison step is required.
        stored_n = np.array([], dtype=float)
        if n_key in event_fitting:
            stored_n = np.asarray(event_fitting[n_key], dtype=float).ravel()
            if stored_n.size and np.isfinite(stored_n[0]) and int(round(stored_n[0])) <= 1:
                reclassified[i] = 1.0
                continue

        if diff_key not in event_fitting:
            continue

        diffs = np.asarray(event_fitting[diff_key], dtype=float).ravel()
        if diffs.size == 0:
            continue

        # Preserve the temporal order. For a multi-segment event, require all
        # segment amplitudes to be finite so that we do not accidentally bridge
        # across an unknown segment.
        if not np.all(np.isfinite(diffs)):
            continue

        amplitudes = np.abs(diffs)

        if width_key in event_fitting:
            widths = np.asarray(event_fitting[width_key], dtype=float).ravel()
        else:
            widths = np.ones_like(amplitudes, dtype=float)

        if len(widths) != len(amplitudes) or not np.all(np.isfinite(widths)):
            widths = np.ones_like(amplitudes, dtype=float)
        else:
            widths = np.where(widths > 0, widths, 1.0)

        # Sequentially merge only adjacent segments. After a merge, compare the
        # next segment against the duration-weighted amplitude of the current group.
        effective_count = 1
        current_amp = float(amplitudes[0])
        current_weight = float(widths[0])

        for next_amp, next_weight in zip(amplitudes[1:], widths[1:]):
            similarity = _segment_similarity_percent(current_amp, float(next_amp))

            if similarity >= threshold:
                total_weight = current_weight + float(next_weight)
                current_amp = (
                    current_amp * current_weight
                    + float(next_amp) * float(next_weight)
                ) / total_weight
                current_weight = total_weight
            else:
                effective_count += 1
                current_amp = float(next_amp)
                current_weight = float(next_weight)

        reclassified[i] = float(effective_count)

    return reclassified

def select_2d_gmm_components_bic(
    dwell_ms: np.ndarray,
    delta_i: np.ndarray,
    max_components: int = 5,
):
    """
    Compare 1..max_components full-covariance 2D GMMs using BIC.

    Every candidate model uses exactly the same preprocessing as the main
    2D GMM: [log10(dwell time), delta-I] followed by StandardScaler.

    Lower BIC is better. The returned best_k is the number of Gaussian
    components with the minimum BIC. This is a statistical model-selection
    diagnostic; it does not by itself prove that the same number of physical
    nanopore mechanisms exists.
    """

    dwell_ms = np.asarray(dwell_ms, dtype=float)
    delta_i = np.asarray(delta_i, dtype=float)

    valid = (
        np.isfinite(dwell_ms)
        & (dwell_ms > 0)
        & np.isfinite(delta_i)
    )

    valid_idx = np.flatnonzero(valid)

    if len(valid_idx) < 20:
        raise ValueError(
            "At least 20 events with finite dwell time and ΔI are required "
            "for BIC model selection."
        )

    features = np.column_stack(
        [
            np.log10(dwell_ms[valid_idx]),
            delta_i[valid_idx],
        ]
    )

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    max_components = int(max(1, max_components))
    max_components = min(
        max_components,
        max(1, len(valid_idx) // 5),
    )

    rows = []
    models = {}

    for k in range(1, max_components + 1):

        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=0,
            n_init=10,
            reg_covar=1e-6,
        )

        model.fit(features_scaled)

        bic_value = float(
            model.bic(features_scaled)
        )

        rows.append(
            {
                "Components (K)": k,
                "BIC": bic_value,
            }
        )

        models[k] = model

    table = pd.DataFrame(rows)

    best_row = int(
        table["BIC"].idxmin()
    )

    best_k = int(
        table.loc[
            best_row,
            "Components (K)",
        ]
    )

    min_bic = float(
        table.loc[
            best_row,
            "BIC",
        ]
    )

    table["ΔBIC from best"] = (
        table["BIC"]
        - min_bic
    )

    return {
        "table": table,
        "best_k": best_k,
        "best_model": models[best_k],
        "models": models,
        "scaler": scaler,
        "valid": valid,
        "valid_idx": valid_idx,
    }

def build_bic_selected_population_result(
    bic_result: dict,
    dwell_ms: np.ndarray,
    delta_i: np.ndarray,
    selected_k: int | None = None,
):
    """Turn a BIC-tested GMM into ordered event populations.

    By default this uses the minimum-BIC model. If selected_k is supplied, the
    corresponding already-fitted candidate model is used instead. Components are
    ordered by the median ORIGINAL dwell time of their assigned events so that
    Cluster 1 is the shortest-dwell statistical population and Cluster K is the
    longest-dwell statistical population. The labels remain generic clusters;
    they are not assumed to represent specific physical nanopore mechanisms.
    """

    dwell_ms = np.asarray(dwell_ms, dtype=float)
    delta_i = np.asarray(delta_i, dtype=float)

    if selected_k is None:
        selected_k = int(bic_result["best_k"])
    else:
        selected_k = int(selected_k)

    models = bic_result.get("models", {})
    if selected_k not in models:
        raise ValueError(
            f"K = {selected_k} was not included in the BIC candidate models."
        )

    model = models[selected_k]
    scaler = bic_result["scaler"]
    valid = np.asarray(bic_result["valid"], dtype=bool)
    valid_idx = np.asarray(bic_result["valid_idx"], dtype=int)

    features = np.column_stack(
        [
            np.log10(dwell_ms[valid_idx]),
            delta_i[valid_idx],
        ]
    )

    features_scaled = scaler.transform(features)
    raw_labels = model.predict(features_scaled)
    raw_probabilities = model.predict_proba(features_scaled)

    n_components = int(model.n_components)

    component_median_dwell = np.full(
        n_components,
        np.inf,
        dtype=float,
    )

    for component in range(n_components):
        component_values = dwell_ms[
            valid_idx[raw_labels == component]
        ]
        if len(component_values):
            component_median_dwell[component] = float(
                np.median(component_values)
            )

    component_order = np.argsort(component_median_dwell)

    raw_to_ordered = np.empty(n_components, dtype=int)
    raw_to_ordered[component_order] = np.arange(n_components)

    ordered_valid_labels = raw_to_ordered[raw_labels]

    labels = np.full(
        len(dwell_ms),
        -1,
        dtype=int,
    )
    labels[valid_idx] = ordered_valid_labels

    probabilities = np.full(
        (len(dwell_ms), n_components),
        np.nan,
        dtype=float,
    )
    probabilities[valid_idx, :] = raw_probabilities[
        :,
        component_order,
    ]

    centres_original = scaler.inverse_transform(
        model.means_
    )[component_order]

    centre_dwell_ms = 10 ** centres_original[:, 0]
    centre_delta_i = centres_original[:, 1]

    median_dwell_ms = component_median_dwell[component_order]

    cluster_indices = [
        np.flatnonzero(labels == cluster)
        for cluster in range(n_components)
    ]

    counts = np.asarray(
        [len(idx) for idx in cluster_indices],
        dtype=int,
    )

    # Descriptive statistics calculated directly from the ORIGINAL events
    # assigned to each selected-K cluster. These are different from the GMM
    # component centres, which are model parameters in log-dwell + ΔI space.
    mean_dwell_ms = np.asarray(
        [
            float(np.mean(dwell_ms[idx])) if len(idx) else np.nan
            for idx in cluster_indices
        ],
        dtype=float,
    )

    mean_delta_i = np.asarray(
        [
            float(np.mean(delta_i[idx])) if len(idx) else np.nan
            for idx in cluster_indices
        ],
        dtype=float,
    )

    assignment_probability = np.max(
        probabilities[valid_idx, :],
        axis=1,
    )

    return {
        "model": model,
        "scaler": scaler,
        "valid": valid,
        "valid_idx": valid_idx,
        "n_components": n_components,
        "component_order": component_order,
        "raw_to_ordered": raw_to_ordered,
        "labels": labels,
        "probabilities": probabilities,
        "cluster_indices": cluster_indices,
        "counts": counts,
        "median_dwell_ms": median_dwell_ms,
        "mean_dwell_ms": mean_dwell_ms,
        "mean_delta_i": mean_delta_i,
        "centre_dwell_ms": centre_dwell_ms,
        "centre_delta_i": centre_delta_i,
        "assignment_probability": assignment_probability,
    }



def read_sources(dataset_bytes, event_data_bytes=None, fitting_bytes=None,
                 dwell_unit="s"):
    """Load the required dataset and validate any supplied companion files."""
    dataset = load_npz_bytes(dataset_bytes)
    if "X" not in dataset:
        raise ValueError("The dataset NPZ must contain an X array.")
    X = np.asarray(dataset["X"], dtype=float)
    if X.ndim != 2 or X.shape[1] < 5 or len(X) == 0:
        raise ValueError("X must be a non-empty 2D array with at least five columns.")
    factor = {"s": 1000.0, "ms": 1.0, "us": 0.001}[dwell_unit]
    dwell_ms = X[:, 4].copy() * factor
    delta_i = X[:, 0].copy()
    n = len(X)

    event_data = load_npz_bytes(event_data_bytes) if event_data_bytes else None
    fitting = load_npz_bytes(fitting_bytes) if fitting_bytes else None
    original_ids = np.arange(n, dtype=int)
    notes = ["Dataset loaded; X[:,0] is ΔI and X[:,4] is dwell time."]
    if event_data is not None:
        if "events" not in event_data:
            raise ValueError("event_data must contain the events array.")
        events = np.asarray(event_data["events"], dtype=object)
        if len(events) != n:
            raise ValueError("event_data and dataset event counts do not match.")
        original_ids = np.asarray(
            [int(ev.get("event_id", i)) for i, ev in enumerate(events)], dtype=int
        )
        if len(np.unique(original_ids)) != n:
            raise ValueError("event_data contains duplicate original event IDs.")
        if all(isinstance(ev, dict) and "start_time" in ev and "end_time" in ev
               for ev in events):
            widths = np.asarray(
                [float(ev["end_time"]) - float(ev["start_time"]) for ev in events]
            )
            if "sampling_rate" in event_data:
                rate = float(np.asarray(event_data["sampling_rate"]).item())
                atol = max(1e-12, 0.51 / rate) if rate > 0 else 1e-9
            else:
                atol = 1e-9
            source_dwell_s = dwell_ms / 1000.0
            valid_widths = np.isfinite(source_dwell_s) & (source_dwell_s > 0)
            if np.any(valid_widths) and not np.allclose(
                source_dwell_s[valid_widths], widths[valid_widths],
                rtol=1e-7, atol=atol
            ):
                raise ValueError(
                    "Dwell times do not match event_data start/end times. "
                    "Check that the files and dwell-time units are correct."
                )
            notes.append("event_data event order and dwell times match the dataset.")
        else:
            raise ValueError("event_data events must contain start_time and end_time.")

    if fitting is not None:
        ids = sorted({int(m.group(2)) for key in fitting
                      if (m := EVENT_KEY_RE.match(key)) is not None})
        if ids and ids != list(range(n)):
            raise ValueError(
                "event_fitting event indices are not the expected contiguous 0..N-1."
            )
        analysis_keys = [f"EVENT_ANALYSIS_{i}" for i in range(n)]
        if any(key in fitting for key in analysis_keys):
            if not all(key in fitting for key in analysis_keys):
                raise ValueError("event_fitting has incomplete EVENT_ANALYSIS entries.")
            for i, key in enumerate(analysis_keys):
                if not np.allclose(
                    np.asarray(fitting[key], dtype=float), X[i],
                    equal_nan=True, rtol=1e-7, atol=1e-12
                ):
                    raise ValueError(
                        f"event_fitting EVENT_ANALYSIS_{i} does not match dataset row {i}."
                    )
            notes.append("event_fitting analysis rows match the dataset.")
        else:
            raise ValueError(
                "event_fitting needs EVENT_ANALYSIS_i entries to verify synchronization."
            )

    valid = np.isfinite(dwell_ms) & (dwell_ms > 0) & np.isfinite(delta_i)
    if valid.sum() < 20:
        notes.append("Fewer than 20 valid events; GMM fitting is unavailable.")
    return {
        "dataset": dataset, "event_data": event_data, "fitting": fitting,
        "X": X, "dwell_ms": dwell_ms, "delta_i": delta_i,
        "original_ids": original_ids, "valid": valid, "notes": notes,
        "dwell_unit": dwell_unit
    }


def descriptive_table(result, dwell_ms, delta_i):
    """Statistics of hard-assigned original events, not transformed GMM centres."""
    rows = []
    n_valid = int(np.sum(result["valid"]))
    for k, indices in enumerate(result["cluster_indices"], start=1):
        d = np.asarray(dwell_ms)[indices]
        a = np.asarray(delta_i)[indices]
        rows.append({
            "Cluster": f"Cluster {k}", "Events": len(indices),
            "% of valid events": 100 * len(indices) / n_valid,
            "Median dwell (ms)": float(np.median(d)) if len(d) else np.nan,
            "Mean dwell (ms)": float(np.mean(d)) if len(d) else np.nan,
            "Mean ΔI (nA)": float(np.mean(a)) if len(a) else np.nan,
        })
    return pd.DataFrame(rows)


def event_assignment_table(data, result):
    n = len(data["dwell_ms"])
    labels = result["labels"]
    probs = result["probabilities"]
    confidence = np.full(n, np.nan)
    if len(result["valid_idx"]):
        confidence[result["valid_idx"]] = np.max(
            probs[result["valid_idx"]], axis=1
        )
    table = pd.DataFrame({
        "original_row_index": np.arange(n),
        "original_event_id": data["original_ids"],
        "dwell_time_ms": data["dwell_ms"],
        "delta_i": data["delta_i"],
        "cluster": np.where(labels >= 0, labels + 1, -1),
        "assignment_probability": confidence,
    })
    for k in range(result["n_components"]):
        table[f"p_cluster_{k+1}"] = probs[:, k]
    return table


def topology_statistics(result, original_counts, effective_counts):
    """Percentages are calculated only among events with usable segment counts."""
    rows = []
    for k, indices in enumerate(result["cluster_indices"], start=1):
        original = np.asarray(original_counts)[indices]
        effective = np.asarray(effective_counts)[indices]
        good = np.isfinite(effective) & (effective >= 1)
        e = np.rint(effective[good]).astype(int)
        o = original[np.isfinite(original) & (original >= 1)]
        n = len(e)
        rows.append({
            "Cluster": f"Cluster {k}",
            "GMM events": len(indices),
            "Usable topology events": n,
            "Missing topology": len(indices) - n,
            "Mean original segments": float(np.mean(o)) if len(o) else np.nan,
            "Mean reclassified segments": float(np.mean(e)) if n else np.nan,
            "% linear": 100 * np.mean(e == 1) if n else np.nan,
            "% folded": 100 * np.mean(e == 2) if n else np.nan,
            "% complex": 100 * np.mean(e >= 3) if n else np.nan,
        })
    return pd.DataFrame(rows)


def _npz_bytes(arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def optional_filtered_files(indices, data):
    """Reindex each provided source, preserving metadata and event ordering."""
    indices = np.asarray(indices, dtype=int)
    indices = np.sort(indices)
    dataset = data["dataset"]
    out = {"dataset": _npz_bytes({
        **{key: value for key, value in dataset.items() if key != "X"},
        "X": np.asarray(dataset["X"])[indices].copy()
    })}
    if data["event_data"] is not None:
        source = data["event_data"]
        events = []
        for new_id, old_id in enumerate(indices):
            ev = copy.deepcopy(source["events"][old_id])
            ev["event_id"] = int(new_id)
            events.append(ev)
        out["event_data"] = _npz_bytes({
            **{key: value for key, value in source.items() if key != "events"},
            "events": np.asarray(events, dtype=object)
        })
    if data["fitting"] is not None:
        source = data["fitting"]
        remap = {int(old): int(new) for new, old in enumerate(indices)}
        filtered = {}
        for key, value in source.items():
            match = EVENT_KEY_RE.match(key)
            if match is None:
                filtered[key] = value
            else:
                prefix, old_id, suffix = match.groups()
                if int(old_id) in remap:
                    filtered[f"{prefix}_{remap[int(old_id)]}{suffix}"] = value
        out["event_fitting"] = _npz_bytes(filtered)
    return out


def export_bundle(data, bic_result, result, topology_table=None,
                  effective_counts=None, threshold=None, source_name="dataset"):
    """Create analysis tables and only the NPZ source types actually supplied."""
    if any(len(idx) == 0 for idx in result["cluster_indices"]):
        raise ValueError("At least one component has no hard-assigned events. Choose another K before exporting.")
    assignment = event_assignment_table(data, result)
    summary = descriptive_table(result, data["dwell_ms"], data["delta_i"])
    if effective_counts is not None:
        assignment["reclassified_segments"] = effective_counts
    metadata = {
        "application": "Nanopore Studio", "version": "2.0",
        "source_name": source_name, "dwell_input_unit": data["dwell_unit"],
        "features": ["log10(dwell_ms)", "delta_i"],
        "preprocessing": "StandardScaler",
        "covariance_type": "full", "random_state": 0,
        "n_init": 10, "reg_covar": 1e-6,
        "K_BIC": int(bic_result["best_k"]),
        "K_selected": int(result["n_components"]),
        "K_tested": list(map(int, bic_result["table"]["Components (K)"])),
        "K_selected_BIC": float(bic_result["table"].loc[
            bic_result["table"]["Components (K)"] == result["n_components"], "BIC"
        ].iloc[0]),
        "valid_events": int(result["valid"].sum()),
        "excluded_events": int((~result["valid"]).sum()),
        "topology_similarity_threshold_percent": threshold,
        "topology_rule": (
            "Adjacent amplitude similarity = 100*min(abs(a),abs(b))/max(abs(a),abs(b)). "
            "Greedy duration-weighted merging. 1=linear-like, 2=folded-like, >=3=complex."
            if threshold is not None else None
        ),
        "cluster_order": "ascending median original dwell time",
        "note": "Gaussian components are statistical descriptions, not proven physical mechanisms."
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("analysis/cluster_summary.csv", summary.to_csv(index=False))
        zf.writestr("analysis/event_assignments.csv", assignment.to_csv(index=False))
        zf.writestr("analysis/bic_model_selection.csv",
                    bic_result["table"].to_csv(index=False))
        zf.writestr("analysis/metadata.json", json.dumps(metadata, indent=2))
        if topology_table is not None:
            zf.writestr("analysis/topology_summary.csv",
                        topology_table.to_csv(index=False))
        excluded = assignment[assignment["cluster"] < 0].copy()
        if len(excluded):
            excluded["exclusion_reason"] = np.where(
                ~np.isfinite(excluded["dwell_time_ms"]), "non-finite dwell",
                np.where(excluded["dwell_time_ms"] <= 0, "non-positive dwell",
                         "non-finite ΔI")
            )
            zf.writestr("analysis/excluded_events.csv",
                        excluded.to_csv(index=False))
        for k, indices in enumerate(result["cluster_indices"], start=1):
            if not len(indices):
                continue
            folder = f"GMM_CLUSTER_{k}"
            mapping = assignment.iloc[indices].copy()
            mapping.insert(0, "new_event_id", np.arange(len(indices)))
            zf.writestr(f"{folder}/event_id_mapping.csv", mapping.to_csv(index=False))
            source_files = optional_filtered_files(indices, data)
            for kind, content in source_files.items():
                zf.writestr(f"{folder}/{source_name}_{folder}.{kind}.npz", content)
    return buffer.getvalue()
