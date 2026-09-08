"""BIC-guided nanopore analysis. Source measurements are never modified."""
from __future__ import annotations
import copy
import io
import json
import re
import zipfile
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from splitter_core import load_npz_bytes, EVENT_KEY_RE, build_filtered_files
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


def _to_bytes(arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def load_sources(dataset_bytes, event_bytes=None, fitting_bytes=None, dwell_unit="s"):
    dataset = load_npz_bytes(dataset_bytes)
    if "X" not in dataset:
        raise ValueError("dataset.npz does not contain X.")
    X = np.asarray(dataset["X"], dtype=float)
    if X.ndim != 2 or X.shape[1] < 5 or len(X) == 0:
        raise ValueError("X must be a non-empty 2D array with at least five columns.")
    factor = {"s": 1000., "ms": 1., "us": .001}[dwell_unit]
    dwell = X[:, 4] * factor
    n = len(X)
    event_data = load_npz_bytes(event_bytes) if event_bytes is not None else None
    fitting = load_npz_bytes(fitting_bytes) if fitting_bytes is not None else None
    ids = np.arange(n, dtype=int)
    if event_data is not None:
        if "events" not in event_data:
            raise ValueError("event_data is missing the events array.")
        events = np.asarray(event_data["events"], dtype=object)
        if len(events) != n:
            raise ValueError("event_data and dataset have different event counts.")
        if any(not isinstance(e, dict) or "start_time" not in e or
               "end_time" not in e for e in events):
            raise ValueError("event_data must contain start_time and end_time.")
        ids = np.array([int(e.get("event_id", i)) for i, e in enumerate(events)])
        if len(np.unique(ids)) != n:
            raise ValueError("Duplicate original event IDs.")
        widths = np.array([float(e["end_time"] - e["start_time"]) for e in events])
        rate = float(np.asarray(event_data.get("sampling_rate", 0)).item())
        atol = max(1e-12, .51 / rate) if rate > 0 else 1e-9
        valid_widths = np.isfinite(dwell) & (dwell > 0)
        if not np.allclose(dwell[valid_widths] / 1000, widths[valid_widths],
                           rtol=1e-7, atol=atol):
            raise ValueError("Dwell times do not match event_data. Check file pairing and units.")
    if fitting is not None:
        analysis = [f"EVENT_ANALYSIS_{i}" for i in range(n)]
        if not all(key in fitting for key in analysis):
            raise ValueError("event_fitting is missing EVENT_ANALYSIS rows.")
        for i, key in enumerate(analysis):
            if not np.allclose(np.asarray(fitting[key], dtype=float), X[i],
                               equal_nan=True, rtol=1e-7, atol=1e-12):
                raise ValueError(f"event_fitting row {i} does not match dataset.")
        present_ids = {int(m.group(2)) for key in fitting
                       if (m := EVENT_KEY_RE.match(key)) is not None}
        if present_ids != set(range(n)):
            raise ValueError("event_fitting event indices do not match dataset.")
    return dict(dataset=dataset, event_data=event_data, fitting=fitting,
                X=X, dwell_ms=dwell, original_ids=ids, dwell_unit=dwell_unit)


def get_feature(data, source="Dataset ΔI", column=0):
    if source == "Dataset ΔI":
        if column == 4:
            raise ValueError("The dwell-time column cannot be used as ΔI.")
        return np.asarray(data["X"][:, column], dtype=float), (
            "ΔI (nA)" if column == 0 else f"X[:, {column}]"
        )
    if data["fitting"] is None:
        raise ValueError("event_fitting is required for segment-derived ΔI.")
    metrics = derive_event_metrics(data["fitting"], len(data["X"]))
    key = ("Peak segment ΔI" if source == "Peak segment ΔI"
           else "Time-weighted segment ΔI")
    return np.asarray(metrics[key], dtype=float), f"{key} (nA)"


def describe(result, dwell_ms, delta_i):
    rows = []
    n = int(result["valid"].sum())
    for k, idx in enumerate(result["cluster_indices"], start=1):
        d, a = np.asarray(dwell_ms)[idx], np.asarray(delta_i)[idx]
        rows.append({
            "Cluster": f"Cluster {k}", "Events": len(idx),
            "% of valid events": 100 * len(idx) / n,
            "Median dwell (ms)": float(np.median(d)) if len(d) else np.nan,
            "Mean dwell (ms)": float(np.mean(d)) if len(d) else np.nan,
            "Mean ΔI (nA)": float(np.mean(a)) if len(a) else np.nan
        })
    return pd.DataFrame(rows)


def assignments(data, result, delta_i):
    n = len(data["X"])
    labels = result["labels"]
    probs = result["probabilities"]
    confidence = np.full(n, np.nan)
    idx = result["valid_idx"]
    confidence[idx] = np.max(probs[idx], axis=1)
    table = pd.DataFrame({
        "original_row_index": np.arange(n),
        "original_event_id": data["original_ids"],
        "dwell_time_ms": data["dwell_ms"],
        "delta_i": delta_i,
        "cluster": np.where(labels >= 0, labels + 1, -1),
        "assignment_probability": confidence,
    })
    for k in range(result["n_components"]):
        table[f"p_cluster_{k+1}"] = probs[:, k]
    return table


def filtered_sources(indices, data):
    indices = np.sort(np.asarray(indices, dtype=int))
    if data["event_data"] is not None and data["fitting"] is not None:
        files, _ = build_filtered_files(
            indices.copy(), "GMM", data["event_data"], data["dataset"], data["fitting"]
        )
        return files
    dataset = data["dataset"]
    files = {"dataset": _to_bytes({
        **{key: value for key, value in dataset.items() if key != "X"},
        "X": np.asarray(dataset["X"])[indices].copy()
    })}
    if data["event_data"] is not None:
        original = data["event_data"]
        events = []
        for new_id, old_row in enumerate(indices):
            event = copy.deepcopy(original["events"][old_row])
            event["event_id"] = int(new_id)
            events.append(event)
        files["event_data"] = _to_bytes({
            **{key: value for key, value in original.items() if key != "events"},
            "events": np.asarray(events, dtype=object)
        })
    if data["fitting"] is not None:
        remap = {int(old): int(new) for new, old in enumerate(indices)}
        filtered = {}
        for key, value in data["fitting"].items():
            match = EVENT_KEY_RE.match(key)
            if match is None:
                filtered[key] = value
            else:
                prefix, old_id, suffix = match.groups()
                if int(old_id) in remap:
                    filtered[f"{prefix}_{remap[int(old_id)]}{suffix}"] = value
        files["event_fitting"] = _to_bytes(filtered)
    return files


def export_results(data, bic, result, delta_i, feature_name, stem="dataset"):
    table = assignments(data, result, delta_i)
    summary = describe(result, data["dwell_ms"], delta_i)
    metadata = {
        "application": "Nanopore Analysis", "version": "2.1",
        "feature": feature_name, "dwell_input_unit": data["dwell_unit"],
        "features": ["log10(dwell_ms)", feature_name],
        "preprocessing": "StandardScaler", "covariance_type": "full",
        "random_state": 0, "n_init": 10, "reg_covar": 1e-6,
        "K_BIC": int(bic["best_k"]), "K_selected": int(result["n_components"]),
        "K_tested": [int(k) for k in bic["table"]["Components (K)"]],
        "cluster_order": "ascending median original dwell time",
        "valid_events": int(result["valid"].sum()),
        "excluded_events": int((~result["valid"]).sum()),
        "note": "Statistical components are not established physical mechanisms."
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("analysis/cluster_summary.csv", summary.to_csv(index=False))
        z.writestr("analysis/event_assignments.csv", table.to_csv(index=False))
        z.writestr("analysis/bic_model_selection.csv", bic["table"].to_csv(index=False))
        z.writestr("analysis/metadata.json", json.dumps(metadata, indent=2))
        excluded = table[table["cluster"] < 0].copy()
        if len(excluded):
            d, a = excluded["dwell_time_ms"], excluded["delta_i"]
            excluded["exclusion_reason"] = np.select(
                [~np.isfinite(d), d <= 0, ~np.isfinite(a)],
                ["non-finite dwell", "non-positive dwell", "non-finite ΔI"],
                default="invalid 2D feature"
            )
            z.writestr("analysis/excluded_events.csv", excluded.to_csv(index=False))
        for k, idx in enumerate(result["cluster_indices"], start=1):
            if not len(idx):
                continue
            folder = f"GMM_CLUSTER_{k}"
            mapping = table.iloc[idx].copy()
            mapping.insert(0, "new_event_id", np.arange(len(idx)))
            z.writestr(f"{folder}/event_id_mapping.csv", mapping.to_csv(index=False))
            for kind, content in filtered_sources(idx, data).items():
                z.writestr(f"{folder}/{stem}_{folder}.{kind}.npz", content)
    return buf.getvalue()
