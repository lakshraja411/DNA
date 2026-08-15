"""Streamlit interface for the Nanopore Event Population Splitter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

from splitter_core import (
    auto_cutoff_gmm,
    build_filtered_files,
    clean_stem,
    load_npz_bytes,
    make_zip,
    validate_inputs,
)


APP_VERSION = "1.4.0"


# ============================================================
# DATA LOADING / DERIVED METRICS
# ============================================================


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


@st.cache_resource(show_spinner=False)
def load_three_files(
    event_data_bytes: bytes,
    dataset_bytes: bytes,
    event_fitting_bytes: bytes,
):
    """Cache NPZ parsing and derived metrics."""

    event_data = load_npz_bytes(
        event_data_bytes
    )

    dataset = load_npz_bytes(
        dataset_bytes
    )

    event_fitting = load_npz_bytes(
        event_fitting_bytes
    )

    n_events = len(
        np.asarray(dataset["X"])
    )

    derived_metrics = derive_event_metrics(
        event_fitting,
        n_events,
    )

    return (
        event_data,
        dataset,
        event_fitting,
        derived_metrics,
    )



# ============================================================
# GMM MODEL-SELECTION HELPERS
# ============================================================


def _bic_table(models: dict[int, GaussianMixture], features: np.ndarray):
    """Return a tidy BIC table for fitted GMM models."""

    rows = []

    for k in sorted(models):

        bic_value = float(
            models[k].bic(
                features
            )
        )

        rows.append(
            {
                "Components": int(k),
                "BIC": bic_value,
            }
        )

    frame = pd.DataFrame(
        rows
    )

    if len(frame):

        best_bic = float(
            frame["BIC"].min()
        )

        frame["ΔBIC"] = (
            frame["BIC"]
            - best_bic
        )

    return frame


def fit_1d_gmm_models(
    dwell_ms: np.ndarray,
    max_components: int = 3,
):
    """
    Fit 1..max_components GMMs to log10(dwell time).

    Returns the finite-event indices, transformed feature matrix,
    fitted models, BIC table, and best component count.
    """

    dwell_ms = np.asarray(
        dwell_ms,
        dtype=float,
    )

    valid = (
        np.isfinite(
            dwell_ms
        )
        &
        (dwell_ms > 0)
    )

    event_indices = np.flatnonzero(
        valid
    )

    log_dwell = np.log10(
        dwell_ms[
            valid
        ]
    ).reshape(
        -1,
        1,
    )

    if len(log_dwell) < 10:

        raise ValueError(
            "At least 10 finite positive dwell times are required "
            "for GMM model comparison."
        )

    max_k = min(
        int(max_components),
        max(
            1,
            len(log_dwell) // 5,
        ),
    )

    models = {}

    for k in range(
        1,
        max_k + 1,
    ):

        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=0,
            n_init=10,
            reg_covar=1e-6,
        )

        model.fit(
            log_dwell
        )

        models[k] = model

    bic_frame = _bic_table(
        models,
        log_dwell,
    )

    best_k = int(
        bic_frame.loc[
            bic_frame["BIC"].idxmin(),
            "Components",
        ]
    )

    return {
        "event_indices": event_indices,
        "features": log_dwell,
        "models": models,
        "bic": bic_frame,
        "best_k": best_k,
    }


def fit_2d_gmm_models(
    dwell_ms: np.ndarray,
    y_values: np.ndarray,
    max_components: int = 3,
):
    """
    Fit 1..max_components GMMs to:
        [log10(dwell time), Y]

    Features are standardized before fitting so the arbitrary units or
    magnitude of ΔI cannot dominate the clustering.
    """

    dwell_ms = np.asarray(
        dwell_ms,
        dtype=float,
    )

    y_values = np.asarray(
        y_values,
        dtype=float,
    )

    if len(dwell_ms) != len(y_values):

        raise ValueError(
            "Dwell time and Y metric must contain the same number of events."
        )

    valid = (
        np.isfinite(
            dwell_ms
        )
        &
        (dwell_ms > 0)
        &
        np.isfinite(
            y_values
        )
    )

    event_indices = np.flatnonzero(
        valid
    )

    raw_features = np.column_stack(
        [
            np.log10(
                dwell_ms[
                    valid
                ]
            ),
            y_values[
                valid
            ],
        ]
    )

    if len(raw_features) < 10:

        raise ValueError(
            "At least 10 finite events are required for 2D GMM comparison."
        )

    scaler = StandardScaler()

    features = scaler.fit_transform(
        raw_features
    )

    max_k = min(
        int(max_components),
        max(
            1,
            len(features) // 5,
        ),
    )

    models = {}

    for k in range(
        1,
        max_k + 1,
    ):

        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=0,
            n_init=10,
            reg_covar=1e-6,
        )

        model.fit(
            features
        )

        models[k] = model

    bic_frame = _bic_table(
        models,
        features,
    )

    best_k = int(
        bic_frame.loc[
            bic_frame["BIC"].idxmin(),
            "Components",
        ]
    )

    return {
        "event_indices": event_indices,
        "raw_features": raw_features,
        "features": features,
        "scaler": scaler,
        "models": models,
        "bic": bic_frame,
        "best_k": best_k,
    }


def ordered_binary_labels_1d(
    model: GaussianMixture,
    features: np.ndarray,
):
    """
    Return binary labels ordered by mean log dwell:
        0 = short-like
        1 = long-like
    """

    if model.n_components != 2:

        raise ValueError(
            "Binary labels require a two-component model."
        )

    raw_labels = model.predict(
        features
    )

    order = np.argsort(
        model.means_[
            :,
            0
        ]
    )

    mapping = {
        int(order[0]): 0,
        int(order[1]): 1,
    }

    labels = np.asarray(
        [
            mapping[
                int(label)
            ]
            for label in raw_labels
        ],
        dtype=int,
    )

    return labels


def ordered_binary_labels_2d(
    model: GaussianMixture,
    features: np.ndarray,
    scaler: StandardScaler,
):
    """
    Return binary 2D-GMM labels ordered by mean dwell:
        0 = short-like
        1 = long-like
    """

    if model.n_components != 2:

        raise ValueError(
            "Binary labels require a two-component model."
        )

    raw_labels = model.predict(
        features
    )

    component_means_raw = scaler.inverse_transform(
        model.means_
    )

    order = np.argsort(
        component_means_raw[
            :,
            0
        ]
    )

    mapping = {
        int(order[0]): 0,
        int(order[1]): 1,
    }

    labels = np.asarray(
        [
            mapping[
                int(label)
            ]
            for label in raw_labels
        ],
        dtype=int,
    )

    return labels, component_means_raw, order


def gmm_agreement_table(
    dwell_ms: np.ndarray,
    one_d_result: dict,
    two_d_result: dict,
):
    """
    Compare 1D and 2D two-component assignments for the events that are
    valid in both analyses.
    """

    if (
        2 not in one_d_result["models"]
        or 2 not in two_d_result["models"]
    ):

        return None

    one_ids = np.asarray(
        one_d_result[
            "event_indices"
        ],
        dtype=int,
    )

    two_ids = np.asarray(
        two_d_result[
            "event_indices"
        ],
        dtype=int,
    )

    common_ids = np.intersect1d(
        one_ids,
        two_ids,
    )

    if len(common_ids) == 0:

        return None

    one_labels_all = ordered_binary_labels_1d(
        one_d_result[
            "models"
        ][2],
        one_d_result[
            "features"
        ],
    )

    two_labels_all, _, _ = ordered_binary_labels_2d(
        two_d_result[
            "models"
        ][2],
        two_d_result[
            "features"
        ],
        two_d_result[
            "scaler"
        ],
    )

    one_lookup = {
        int(event_id): int(label)
        for event_id, label in zip(
            one_ids,
            one_labels_all,
        )
    }

    two_lookup = {
        int(event_id): int(label)
        for event_id, label in zip(
            two_ids,
            two_labels_all,
        )
    }

    one_common = np.asarray(
        [
            one_lookup[
                int(event_id)
            ]
            for event_id in common_ids
        ],
        dtype=int,
    )

    two_common = np.asarray(
        [
            two_lookup[
                int(event_id)
            ]
            for event_id in common_ids
        ],
        dtype=int,
    )

    agreement = float(
        np.mean(
            one_common
            ==
            two_common
        )
    )

    counts = np.zeros(
        (
            2,
            2,
        ),
        dtype=int,
    )

    for one_label, two_label in zip(
        one_common,
        two_common,
    ):

        counts[
            one_label,
            two_label,
        ] += 1

    table = pd.DataFrame(
        counts,
        index=[
            "1D short-like",
            "1D long-like",
        ],
        columns=[
            "2D short-like",
            "2D long-like",
        ],
    )

    changed_ids = common_ids[
        one_common
        !=
        two_common
    ]

    return {
        "agreement": agreement,
        "common_ids": common_ids,
        "changed_ids": changed_ids,
        "table": table,
    }


def plot_2d_gmm_boundary(
    dwell_ms: np.ndarray,
    y_values: np.ndarray,
    two_d_result: dict,
    y_label: str,
    density_bandwidth_scale: float = 0.9,
):
    """
    Draw a publication-style 2D KDE density cloud and overlay the
    two-component GMM equal-posterior boundary.

    This is visualisation only; it does not change export classification.
    """

    if 2 not in two_d_result["models"]:

        return None

    event_ids = np.asarray(
        two_d_result[
            "event_indices"
        ],
        dtype=int,
    )

    x = np.asarray(
        dwell_ms[
            event_ids
        ],
        dtype=float,
    )

    y = np.asarray(
        y_values[
            event_ids
        ],
        dtype=float,
    )

    # Robust display range.
    x_lo, x_hi = np.percentile(
        x,
        [
            0.2,
            99.8,
        ],
    )

    y_lo, y_hi = np.percentile(
        y,
        [
            0.2,
            99.8,
        ],
    )

    keep = (
        (x >= x_lo)
        &
        (x <= x_hi)
        &
        (y >= y_lo)
        &
        (y <= y_hi)
    )

    x_display = x[
        keep
    ]

    y_display = y[
        keep
    ]

    if len(x_display) < 5:

        return None

    # Smooth KDE cloud in the same visual style used elsewhere in the app.
    x_grid, y_grid, density = true_2d_kde_density(
        x_display,
        y_display,
        log_x=False,
        grid_size=220,
        bandwidth_scale=density_bandwidth_scale,
        x_percentiles=(
            0.0,
            100.0,
        ),
        y_percentiles=(
            0.0,
            100.0,
        ),
    )

    if density is None:

        return None

    fig, ax = plt.subplots(
        figsize=(
            9,
            5.5,
        )
    )

    fig.patch.set_facecolor(
        "black"
    )

    ax.set_facecolor(
        "black"
    )

    threshold = (
        0.0015
        * float(
            np.max(
                density
            )
        )
    )

    density_masked = np.ma.masked_where(
        density <= threshold,
        density,
    )

    ax.pcolormesh(
        x_grid,
        y_grid,
        density_masked,
        shading="auto",
        cmap="magma",
    )

    # --------------------------------------------------------
    # GMM POSTERIOR BOUNDARY
    # --------------------------------------------------------

    model = two_d_result[
        "models"
    ][2]

    scaler = two_d_result[
        "scaler"
    ]

    log_x_grid = np.linspace(
        np.log10(
            max(
                float(x_lo),
                1e-12,
            )
        ),
        np.log10(
            float(x_hi)
        ),
        260,
    )

    y_boundary_grid = np.linspace(
        float(y_lo),
        float(y_hi),
        260,
    )

    LG, YG = np.meshgrid(
        log_x_grid,
        y_boundary_grid,
    )

    raw_grid = np.column_stack(
        [
            LG.ravel(),
            YG.ravel(),
        ]
    )

    scaled_grid = scaler.transform(
        raw_grid
    )

    posterior = model.predict_proba(
        scaled_grid
    )

    component_means_raw = scaler.inverse_transform(
        model.means_
    )

    order = np.argsort(
        component_means_raw[
            :,
            0
        ]
    )

    p_short = posterior[
        :,
        order[0]
    ].reshape(
        LG.shape
    )

    dwell_grid_ms = (
        10
        **
        log_x_grid
    )

    # 0.5 posterior = equal-probability decision boundary.
    ax.contour(
        dwell_grid_ms,
        y_boundary_grid,
        p_short,
        levels=[
            0.5
        ],
        colors=[
            "white"
        ],
        linewidths=[
            2.0
        ],
    )

    # Component centres.
    centre_dwell = (
        10
        **
        component_means_raw[
            :,
            0
        ]
    )

    centre_y = component_means_raw[
        :,
        1
    ]

    for component_id in order:

        ax.scatter(
            [
                centre_dwell[
                    component_id
                ]
            ],
            [
                centre_y[
                    component_id
                ]
            ],
            marker="x",
            s=95,
            linewidths=2.2,
            color="white",
        )

    ax.set_xlim(
        float(x_lo),
        float(x_hi),
    )

    ax.set_ylim(
        float(y_lo),
        float(y_hi),
    )

    ax.set_xlabel(
        "Dwell time (ms)",
        color="white",
    )

    ax.set_ylabel(
        y_label,
        color="white",
    )

    ax.set_title(
        "2D GMM: KDE cloud + equal-posterior boundary",
        color="white",
    )

    ax.tick_params(
        colors="white"
    )

    for spine in ax.spines.values():

        spine.set_color(
            "white"
        )

    fig.tight_layout()

    return fig



# ============================================================
# KDE FUNCTIONS
# ============================================================


def silverman_bandwidth(
    values: np.ndarray,
) -> float:
    """Return robust automatic KDE bandwidth."""

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        return 0.1

    std = float(
        np.std(
            values,
            ddof=1,
        )
    )

    q25, q75 = np.percentile(
        values,
        [25, 75],
    )

    if q75 > q25:

        iqr_sigma = float(
            (q75 - q25)
            / 1.349
        )

    else:

        iqr_sigma = np.nan

    candidates = [

        value

        for value in (
            std,
            iqr_sigma,
        )

        if (
            np.isfinite(value)
            and value > 0
        )
    ]

    if candidates:

        sigma = min(
            candidates
        )

    else:

        sigma = max(
            abs(
                float(
                    np.mean(values)
                )
            ),
            1.0,
        )

    bandwidth = (
        0.9
        * sigma
        * len(values) ** (-1 / 5)
    )

    return max(
        float(bandwidth),
        1e-6,
    )


def kde_curve(
    values: np.ndarray,
    log_space: bool,
    n_points: int = 600,
):
    """Calculate 1D KDE curve."""

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    # --------------------------------------------------------
    # LOG DWELL KDE
    # --------------------------------------------------------

    if log_space:

        values = values[
            values > 0
        ]

        transformed = np.log10(
            values
        )

    else:

        transformed = values

    if len(transformed) < 2:

        return (
            None,
            None,
        )

    lo = float(
        np.min(transformed)
    )

    hi = float(
        np.max(transformed)
    )

    if np.isclose(
        lo,
        hi,
    ):

        return (
            None,
            None,
        )

    padding = (
        0.03
        * (hi - lo)
    )

    grid = np.linspace(
        lo - padding,
        hi + padding,
        n_points,
    )

    bandwidth = silverman_bandwidth(
        transformed
    )

    kde = KernelDensity(
        kernel="gaussian",
        bandwidth=bandwidth,
    )

    kde.fit(
        transformed.reshape(
            -1,
            1,
        )
    )

    density = np.exp(
        kde.score_samples(
            grid.reshape(
                -1,
                1,
            )
        )
    )

    if log_space:

        x = 10 ** grid

    else:

        x = grid

    return (
        x,
        density,
    )



# ============================================================
# HISTOGRAM / 2D DENSITY HELPERS
# ============================================================


def estimate_dwell_resolution_ms(dwell_ms: np.ndarray) -> float:
    """
    Estimate the discrete dwell-time spacing from the data.

    Short nanopore events often occur at discrete sample intervals. Using a
    histogram bin width close to this spacing avoids artificial empty gaps
    caused by choosing far too many bins.
    """

    values = np.asarray(dwell_ms, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]

    if len(values) < 2:
        return 0.005

    unique_values = np.unique(np.round(values, 6))

    if len(unique_values) < 2:
        return 0.005

    diffs = np.diff(unique_values)
    diffs = diffs[diffs > 1e-6]

    if len(diffs) == 0:
        return 0.005

    # A low percentile is robust to occasional missing dwell values while
    # still recovering the underlying sampling increment.
    resolution = float(np.percentile(diffs, 10))

    return float(np.clip(resolution, 0.001, 0.05))


def linear_histogram_edges(values: np.ndarray, bin_width_ms: float) -> np.ndarray:
    """Create linear bin edges aligned to the selected bin width."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.array([0.0, bin_width_ms], dtype=float)

    bin_width_ms = max(float(bin_width_ms), 1e-9)

    low = float(np.min(values))
    high = float(np.max(values))

    # Centre quantized dwell values inside their bins. This prevents the
    # alternating-bar appearance that can happen when bin edges sit directly
    # on the discrete sample times.
    start = np.floor(low / bin_width_ms) * bin_width_ms - 0.5 * bin_width_ms
    stop = np.ceil(high / bin_width_ms) * bin_width_ms + 1.5 * bin_width_ms

    edges = np.arange(start, stop, bin_width_ms)

    if len(edges) < 2:
        edges = np.array([low - 0.5 * bin_width_ms, high + 0.5 * bin_width_ms])

    return edges


@st.cache_data(show_spinner=False)
def true_2d_kde_density(
    x: np.ndarray,
    y: np.ndarray,
    log_x: bool = False,
    grid_size: int = 220,
    bandwidth_scale: float = 1.0,
    x_percentiles: tuple[float, float] = (0.2, 99.8),
    y_percentiles: tuple[float, float] = (0.2, 99.8),
):
    """
    Genuine smooth 2D Gaussian KDE for nanopore event-density plots.

    This version uses scipy.stats.gaussian_kde because it gives the classic
    continuous KDE cloud appearance desired for ΔI-vs-dwell plots.

    Parameters
    ----------
    x
        Dwell time in ms.
    y
        Selected event metric (e.g. ΔI in pA).
    log_x
        If True, KDE is fitted in log10(dwell time). For figures like the
        desired example, leave this False so dwell time remains linear.
    grid_size
        Number of evaluation points per axis.
    bandwidth_scale
        Multiplier applied to Scott's KDE bandwidth.
        < 1 gives sharper structure; > 1 gives smoother structure.
    x_percentiles, y_percentiles
        Robust display limits used to prevent a few extreme outliers from
        stretching the density field.

    Returns
    -------
    x_grid, y_grid, density
        1D display coordinates and 2D KDE density array.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    if log_x:
        valid &= x > 0

    x = x[valid]
    y = y[valid]

    if len(x) < 5:
        return None, None, None

    if log_x:
        x_work = np.log10(x)
    else:
        x_work = x.copy()

    # Robust display window.
    x_low, x_high = np.percentile(
        x_work,
        x_percentiles,
    )

    y_low, y_high = np.percentile(
        y,
        y_percentiles,
    )

    keep = (
        (x_work >= x_low)
        & (x_work <= x_high)
        & (y >= y_low)
        & (y <= y_high)
    )

    x_work = x_work[keep]
    y = y[keep]

    if len(x_work) < 5:
        return None, None, None

    if x_low >= x_high or y_low >= y_high:
        return None, None, None

    # Fit a true 2D Gaussian KDE.
    values = np.vstack(
        [
            x_work,
            y,
        ]
    )

    try:
        kde = gaussian_kde(
            values,
            bw_method="scott",
        )

        # Scale Scott's rule to give user-controlled smoothing.
        kde.set_bandwidth(
            bw_method=kde.factor * float(bandwidth_scale)
        )

    except Exception:
        return None, None, None

    # Evaluation grid.
    x_grid_work = np.linspace(
        x_low,
        x_high,
        int(grid_size),
    )

    y_grid = np.linspace(
        y_low,
        y_high,
        int(grid_size),
    )

    GX, GY = np.meshgrid(
        x_grid_work,
        y_grid,
    )

    positions = np.vstack(
        [
            GX.ravel(),
            GY.ravel(),
        ]
    )

    density = kde(
        positions
    ).reshape(
        GX.shape
    )

    # Convert x back to real dwell time for display.
    if log_x:
        x_grid = 10 ** x_grid_work
    else:
        x_grid = x_grid_work

    return (
        x_grid,
        y_grid,
        density,
    )



# ============================================================
# AXIS / HOVER HELPERS
# ============================================================


def axis_limit_controls(
    key_prefix: str,
    x_default: tuple[float, float],
    y_default: tuple[float, float],
    *,
    log_x: bool = False,
):
    """
    Optional manual plot limits.

    These controls affect only what is displayed.
    They do not alter the dwell-time cutoff, population assignment,
    or exported event files.
    """

    x_range = None
    y_range = None

    with st.expander(
        "Set axis limits / zoom",
        expanded=False,
    ):

        st.caption(
            "These limits change only the visible plot range. "
            "They do not remove events or change the SHORT/LONG split."
        )

        set_x = st.checkbox(
            "Set custom X-axis limits",
            value=False,
            key=f"{key_prefix}_custom_x",
        )

        if set_x:

            x1, x2 = st.columns(2)

            x_span = abs(
                float(x_default[1])
                - float(x_default[0])
            )

            x_step = max(
                x_span / 200.0,
                1e-6,
            )

            with x1:

                x_min = st.number_input(
                    "X minimum",
                    value=float(x_default[0]),
                    step=float(x_step),
                    format="%.6f",
                    key=f"{key_prefix}_xmin",
                )

            with x2:

                x_max = st.number_input(
                    "X maximum",
                    value=float(x_default[1]),
                    step=float(x_step),
                    format="%.6f",
                    key=f"{key_prefix}_xmax",
                )

            if log_x and x_min <= 0:

                st.warning(
                    "For a logarithmic X-axis, X minimum must be > 0."
                )

            elif x_min >= x_max:

                st.warning(
                    "X minimum must be smaller than X maximum."
                )

            else:

                x_range = (
                    float(x_min),
                    float(x_max),
                )

        set_y = st.checkbox(
            "Set custom Y-axis limits",
            value=False,
            key=f"{key_prefix}_custom_y",
        )

        if set_y:

            y1, y2 = st.columns(2)

            y_span = abs(
                float(y_default[1])
                - float(y_default[0])
            )

            y_step = max(
                y_span / 200.0,
                1e-6,
            )

            with y1:

                y_min = st.number_input(
                    "Y minimum",
                    value=float(y_default[0]),
                    step=float(y_step),
                    format="%.6f",
                    key=f"{key_prefix}_ymin",
                )

            with y2:

                y_max = st.number_input(
                    "Y maximum",
                    value=float(y_default[1]),
                    step=float(y_step),
                    format="%.6f",
                    key=f"{key_prefix}_ymax",
                )

            if y_min >= y_max:

                st.warning(
                    "Y minimum must be smaller than Y maximum."
                )

            else:

                y_range = (
                    float(y_min),
                    float(y_max),
                )

    return x_range, y_range


def apply_axis_limits(
    ax,
    x_range,
    y_range,
):
    """Apply optional manual limits to a Matplotlib axis."""

    if x_range is not None:
        ax.set_xlim(
            x_range[0],
            x_range[1],
        )

    if y_range is not None:
        ax.set_ylim(
            y_range[0],
            y_range[1],
        )


def event_hover_inspector(
    x: np.ndarray,
    y: np.ndarray,
    event_ids: np.ndarray,
    y_name: str,
    *,
    log_x: bool,
    x_range=None,
    y_range=None,
):
    """
    Optional interactive scatter used only for inspecting individual events.

    The publication-style density plot stays Matplotlib/pcolormesh exactly as
    before. This separate inspector provides event-level hover information.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=5,
                opacity=0.55,
            ),
            customdata=np.asarray(
                event_ids,
                dtype=int,
            ).reshape(-1, 1),
            hovertemplate=(
                "Event ID: %{customdata[0]:.0f}"
                "<br>Dwell time: %{x:.5f} ms"
                f"<br>{y_name}: "
                "%{y:.6g}"
                "<extra></extra>"
            ),
            name="Events",
        )
    )

    fig.update_layout(
        height=430,
        margin=dict(
            l=60,
            r=20,
            t=35,
            b=60,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
        dragmode="zoom",
        showlegend=False,
    )

    fig.update_xaxes(
        title="Dwell time (ms)",
        type="log" if log_x else "linear",
        showgrid=False,
        showline=True,
        linecolor="black",
        ticks="outside",
    )

    fig.update_yaxes(
        title=y_name,
        showgrid=False,
        showline=True,
        linecolor="black",
        ticks="outside",
    )

    if x_range is not None:

        if log_x:

            fig.update_xaxes(
                range=[
                    np.log10(
                        x_range[0]
                    ),
                    np.log10(
                        x_range[1]
                    ),
                ]
            )

        else:

            fig.update_xaxes(
                range=list(
                    x_range
                )
            )

    if y_range is not None:

        fig.update_yaxes(
            range=list(
                y_range
            )
        )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
        },
    )


# ============================================================
# MAIN APP
# ============================================================


def main() -> None:

    st.set_page_config(

        page_title=
        "Nanopore Event Population Splitter",

        page_icon="🧬",

        layout="wide",
    )

    st.title(
        "🧬 Nanopore Event Population Splitter"
    )

    st.caption(
        "Split short- and long-dwell event populations "
        "while keeping event_data, dataset, and event_fitting "
        "files synchronized. Includes dwell-time and "
        "2D event-density plots."
    )

    # ========================================================
    # 1. LOAD FILES
    # ========================================================

    with st.sidebar:

        st.header(
            "1 · Load matching files"
        )

        f_event_data = st.file_uploader(

            "event_data (.npz)",

            type=["npz"],

            key="event_data",
        )

        f_dataset = st.file_uploader(

            "dataset (.npz)",

            type=["npz"],

            key="dataset",
        )

        f_fitting = st.file_uploader(

            "event_fitting (.npz)",

            type=["npz"],

            key="event_fitting",
        )

        st.divider()

        st.caption(
            f"Version {APP_VERSION}"
        )

    if not all(
        [
            f_event_data,
            f_dataset,
            f_fitting,
        ]
    ):

        st.info(
            "Upload the three matching NPZ files "
            "from one dataset to begin."
        )

        st.stop()

    # ========================================================
    # LOAD AND VALIDATE
    # ========================================================

    with st.spinner(
        "Loading and checking event synchronization..."
    ):

        (
            event_data,
            dataset,
            event_fitting,
            derived_metrics,
        ) = load_three_files(

            f_event_data.getvalue(),

            f_dataset.getvalue(),

            f_fitting.getvalue(),
        )

        try:

            dwell_s, _, notes = validate_inputs(

                event_data,
                dataset,
                event_fitting,
            )

        except Exception as exc:

            st.error(
                f"Validation failed: {exc}"
            )

            st.stop()

    dwell_ms = (
        np.asarray(
            dwell_s,
            dtype=float,
        )
        * 1000.0
    )

    n_events = len(
        dwell_ms
    )

    X = np.asarray(
        dataset["X"],
        dtype=float,
    )

    dwell_resolution_ms = estimate_dwell_resolution_ms(
        dwell_ms
    )

    st.success(
        f"Files are synchronized correctly: "
        f"**{n_events:,} events**."
    )

    st.caption(
        "Checks: "
        + "; ".join(notes)
        + "."
    )

    # ========================================================
    # AUTOMATIC GMM SPLIT
    # ========================================================

    try:

        suggested_cutoff, gmm = (
            auto_cutoff_gmm(
                dwell_ms
            )
        )

    except Exception as exc:

        suggested_cutoff = float(
            np.median(
                dwell_ms
            )
        )

        gmm = None

        st.warning(

            "Automatic two-population suggestion "
            "was unavailable "

            f"({exc}). "

            "The median is being used only "
            "as an initial value."
        )


    # ========================================================
    # 1D GMM MODEL SELECTION
    # ========================================================

    try:

        gmm_1d_comparison = fit_1d_gmm_models(
            dwell_ms,
            max_components=3,
        )

    except Exception as exc:

        gmm_1d_comparison = None

        st.warning(
            "1D GMM model comparison was unavailable: "
            f"{exc}"
        )

    # ========================================================
    # COLUMNS
    # ========================================================

    plot_col, control_col = st.columns(
        [2, 1],
        gap="large",
    )

    # ========================================================
    # 2. CUT-OFF CONTROL
    # ========================================================

    with control_col:

        st.subheader(
            "2 · Choose the dwell-time boundary"
        )

        st.write(
            "**SHORT:** dwell ≤ cutoff  \n"
            "**LONG:** dwell > cutoff"
        )

        st.caption(

            "The GMM value is an automatic starting point only. "

            "Inspect the distribution and choose the valley "
            "that best separates the populations "
            "in that dataset."
        )

        cutoff_ms = st.number_input(

            "Cutoff (ms)",

            min_value=
            float(
                dwell_ms.min()
            ),

            max_value=
            float(
                dwell_ms.max()
            ),

            value=
            float(
                suggested_cutoff
            ),

            step=0.005,

            format="%.5f",
        )

        # ----------------------------------------------------
        # GMM INFO
        # ----------------------------------------------------

        if gmm is not None:

            st.caption(

                f"GMM suggestion: "
                f"**{suggested_cutoff:.4f} ms**  \n"

                f"Approx. centres: "
                f"{gmm['short_geometric_mean_ms']:.4f} ms "
                f"and "
                f"{gmm['long_geometric_mean_ms']:.4f} ms"
            )


        # ----------------------------------------------------
        # 1D MODEL-SELECTION SUMMARY
        # ----------------------------------------------------

        if gmm_1d_comparison is not None:

            best_1d_k = int(
                gmm_1d_comparison[
                    "best_k"
                ]
            )

            bic_summary_frame = (
                gmm_1d_comparison[
                    "bic"
                ]
                .copy()
            )

            bic_text = " · ".join(
                [
                    (
                        f"K={int(row.Components)}: "
                        f"{row.BIC:.1f}"
                    )
                    for row in bic_summary_frame.itertuples(
                        index=False
                    )
                ]
            )

            st.caption(
                "1D log-dwell BIC — "
                + bic_text
                + ". Lower is better."
            )

            if best_1d_k == 2:

                st.success(
                    "BIC currently favours **2 dwell-time components**."
                )

            else:

                st.warning(
                    f"BIC currently favours **{best_1d_k} component"
                    f"{'' if best_1d_k == 1 else 's'}**, not 2. "
                    "The two-component cutoff is still shown for exploration, "
                    "but should not automatically be treated as evidence for "
                    "two physical populations."
                )

        # ----------------------------------------------------
        # SPLIT EVENT INDICES
        # ----------------------------------------------------

        short_idx = np.flatnonzero(
            dwell_ms <= cutoff_ms
        )

        long_idx = np.flatnonzero(
            dwell_ms > cutoff_ms
        )

        metric_a, metric_b = st.columns(
            2
        )

        metric_a.metric(

            "Short events",

            f"{len(short_idx):,}",

            f"{100 * len(short_idx) / n_events:.1f}%",
        )

        metric_b.metric(

            "Long events",

            f"{len(long_idx):,}",

            f"{100 * len(long_idx) / n_events:.1f}%",
        )

        if (
            len(short_idx)
            and len(long_idx)
        ):

            st.write(

                f"Short median: "
                f"**{np.median(dwell_ms[short_idx]):.4f} ms**  \n"

                f"Long median: "
                f"**{np.median(dwell_ms[long_idx]):.4f} ms**"
            )

        else:

            st.warning(
                "Move the cutoff so that both populations "
                "contain events."
            )

        st.caption(
            f"Estimated dwell-time resolution: "
            f"**{dwell_resolution_ms:.4f} ms**"
        )

    # ========================================================
    # PLOTS
    # ========================================================

    with plot_col:

        st.subheader(
            "Event population plots"
        )

        plot_type = st.radio(

            "Plot type",

            [
                "Dwell histogram",
                "Dwell density (KDE)",
                "2D event density",
                "GMM model comparison",
            ],

            horizontal=True,
        )

        # ====================================================
        # A. HISTOGRAM
        # ====================================================

        if plot_type == "Dwell histogram":

            hist_view = st.radio(
                "Population view",
                [
                    "All",
                    "Short",
                    "Long",
                    "Short + Long overlay",
                ],
                horizontal=True,
                key="hist_population_view",
            )

            log_axis = st.toggle(
                "Logarithmic dwell-time axis",
                value=True,
                key="hist_log_axis",
            )

            if hist_view == "Short":
                hist_values = dwell_ms[short_idx]
            elif hist_view == "Long":
                hist_values = dwell_ms[long_idx]
            else:
                hist_values = dwell_ms

            fig, ax = plt.subplots(
                figsize=(9, 4.8)
            )

            if log_axis:

                positive_values = hist_values[
                    np.isfinite(hist_values)
                    & (hist_values > 0)
                ]

                if len(positive_values) == 0:
                    st.warning(
                        "No positive dwell times are available "
                        "for a logarithmic histogram."
                    )
                    st.stop()

                n_log_bins = st.slider(
                    "Number of log-spaced bins",
                    min_value=25,
                    max_value=150,
                    value=70,
                    step=5,
                    key="hist_log_bins",
                )

                bins = np.logspace(
                    np.log10(np.min(positive_values)),
                    np.log10(np.max(positive_values)),
                    n_log_bins,
                )

                if hist_view == "Short + Long overlay":

                    ax.hist(
                        dwell_ms[short_idx],
                        bins=bins,
                        alpha=0.55,
                        label=f"Short ({len(short_idx):,})",
                    )

                    ax.hist(
                        dwell_ms[long_idx],
                        bins=bins,
                        alpha=0.55,
                        label=f"Long ({len(long_idx):,})",
                    )

                else:

                    ax.hist(
                        hist_values,
                        bins=bins,
                        alpha=0.85,
                        label=f"{hist_view} ({len(hist_values):,})",
                    )

                ax.set_xscale(
                    "log"
                )

            else:

                bin_control = st.radio(
                    "Histogram bin control",
                    [
                        "Number of bins",
                        "Bin width (ms)",
                    ],
                    index=0,
                    horizontal=True,
                    key="hist_bin_control",
                    help=(
                        "Choose the total number of histogram bins directly, "
                        "or specify an exact dwell-time bin width in milliseconds."
                    ),
                )

                if bin_control == "Number of bins":

                    n_bins = st.slider(
                        "Number of bins",
                        min_value=10,
                        max_value=200,
                        value=50,
                        step=5,
                        key="hist_n_bins",
                    )

                    finite_hist_values = np.asarray(
                        hist_values,
                        dtype=float,
                    )

                    finite_hist_values = finite_hist_values[
                        np.isfinite(
                            finite_hist_values
                        )
                    ]

                    if len(finite_hist_values) == 0:

                        st.warning(
                            "No finite dwell times are available "
                            "for this histogram."
                        )

                        st.stop()

                    bins = np.histogram_bin_edges(
                        finite_hist_values,
                        bins=int(n_bins),
                    )

                else:

                    default_width = float(
                        max(
                            dwell_resolution_ms,
                            0.001,
                        )
                    )

                    bin_width_ms = st.number_input(
                        "Histogram bin width (ms)",
                        min_value=0.001,
                        value=default_width,
                        step=default_width,
                        format="%.5f",
                        key="hist_bin_width",
                        help=(
                            "This sets the physical dwell-time width "
                            "of every histogram bin."
                        ),
                    )

                    bins = linear_histogram_edges(
                        hist_values,
                        bin_width_ms,
                    )

                if hist_view == "Short + Long overlay":

                    ax.hist(
                        dwell_ms[short_idx],
                        bins=bins,
                        alpha=0.55,
                        label=f"Short ({len(short_idx):,})",
                    )

                    ax.hist(
                        dwell_ms[long_idx],
                        bins=bins,
                        alpha=0.55,
                        label=f"Long ({len(long_idx):,})",
                    )

                else:

                    ax.hist(
                        hist_values,
                        bins=bins,
                        alpha=0.85,
                        label=f"{hist_view} ({len(hist_values):,})",
                    )

            ax.axvline(
                cutoff_ms,
                linestyle="--",
                linewidth=2,
                label=f"Cutoff = {cutoff_ms:.4f} ms",
            )

            ax.set_xlabel(
                "Dwell time (ms)"
            )

            ax.set_ylabel(
                "Count"
            )

            ax.set_title(
                f"{hist_view} dwell-time distribution"
            )

            ax.legend()

            fig.tight_layout()

            hist_x_default = ax.get_xlim()
            hist_y_default = ax.get_ylim()

            hist_x_range, hist_y_range = axis_limit_controls(
                "hist",
                hist_x_default,
                hist_y_default,
                log_x=log_axis,
            )

            apply_axis_limits(
                ax,
                hist_x_range,
                hist_y_range,
            )

            st.pyplot(
                fig,
                clear_figure=True,
            )

            if not log_axis:
                st.caption(
                    "For linear histograms you can now choose either the "
                    "**number of bins** directly or an exact **bin width (ms)**. "
                    "The same bin edges are used for Short/Long overlays so the "
                    "two populations remain directly comparable."
                )

        # ====================================================
        # B. KDE DWELL DENSITY
        # ====================================================

        elif plot_type == "Dwell density (KDE)":

            log_axis = st.toggle(

                "Fit density in log10(dwell time)",

                value=True,

                key="kde_log_axis",

                help=(
                    "Recommended for nanopore dwell times "
                    "because the distribution is usually "
                    "strongly right-skewed."
                ),
            )

            density_view = st.radio(

                "Show",

                [
                    "Short + Long",
                    "All events",
                ],

                horizontal=True,
            )

            fig, ax = plt.subplots(
                figsize=(9, 4.8)
            )

            # ------------------------------------------------
            # ALL EVENTS
            # ------------------------------------------------

            if density_view == "All events":

                x_all, d_all = kde_curve(
                    dwell_ms,
                    log_axis,
                )

                if x_all is not None:

                    ax.plot(

                        x_all,
                        d_all,

                        linewidth=2,

                        label=
                        "All events",
                    )

            # ------------------------------------------------
            # SHORT AND LONG SEPARATELY
            # ------------------------------------------------

            else:

                if len(short_idx) >= 2:

                    (
                        x_short,
                        d_short,
                    ) = kde_curve(

                        dwell_ms[
                            short_idx
                        ],

                        log_axis,
                    )

                    if x_short is not None:

                        ax.plot(

                            x_short,
                            d_short,

                            linewidth=2,

                            label=
                            f"Short "
                            f"({len(short_idx):,})",
                        )

                if len(long_idx) >= 2:

                    (
                        x_long,
                        d_long,
                    ) = kde_curve(

                        dwell_ms[
                            long_idx
                        ],

                        log_axis,
                    )

                    if x_long is not None:

                        ax.plot(

                            x_long,
                            d_long,

                            linewidth=2,

                            label=
                            f"Long "
                            f"({len(long_idx):,})",
                        )

            # ------------------------------------------------
            # CUT-OFF LINE
            # ------------------------------------------------

            ax.axvline(

                cutoff_ms,

                linestyle="--",

                linewidth=2,

                label=
                f"Cutoff = "
                f"{cutoff_ms:.4f} ms",
            )

            if log_axis:

                ax.set_xscale(
                    "log"
                )

                ax.set_ylabel(
                    "KDE density in log10(dwell time)"
                )

            else:

                ax.set_ylabel(
                    "KDE density"
                )

            ax.set_xlabel(
                "Dwell time (ms)"
            )

            ax.legend()

            fig.tight_layout()

            kde_x_default = ax.get_xlim()
            kde_y_default = ax.get_ylim()

            kde_x_range, kde_y_range = axis_limit_controls(
                "kde",
                kde_x_default,
                kde_y_default,
                log_x=log_axis,
            )

            apply_axis_limits(
                ax,
                kde_x_range,
                kde_y_range,
            )

            st.pyplot(
                fig,
                clear_figure=True,
            )

            st.caption(

                "The KDE is a smoothed view of the dwell-time "
                "distribution. Use it together with the histogram "
                "and 2D density plot when deciding the split."
            )

        # ====================================================
        # C. 2D EVENT DENSITY
        # ====================================================

        elif plot_type == "2D event density":

            st.caption(
                "Smooth 2D Gaussian KDE of dwell time versus the selected "
                "event metric. Linear dwell time is the default for the classic "
                "glowing ΔI–Δt density-map appearance."
            )

            # ------------------------------------------------
            # AVAILABLE Y METRICS
            # ------------------------------------------------

            # ------------------------------------------------
            # Y-AXIS SOURCE
            # ------------------------------------------------
            #
            # The event_fitting file gives segment-derived ΔI values.
            # The dataset.npz file also contains the reduced event feature
            # matrix X.  For this dataset, X[:,0] is used as the default
            # dataset ΔI feature.  The column remains user-selectable so the
            # app is robust to other NanoSense dataset layouts.
            # ------------------------------------------------

            y_source = st.selectbox(
                "Y-axis source",
                [
                    "ΔI from dataset.npz",
                    "Peak segment ΔI from event_fitting",
                    "Time-weighted segment ΔI from event_fitting",
                    "Number of segments from event_fitting",
                    "Other raw dataset column",
                ],
                index=0,
                help=(
                    "Use dataset.npz if you want the same reduced-event ΔI "
                    "feature stored in the NanoSense dataset. "
                    "event_fitting options calculate ΔI from fitted segments."
                ),
            )

            if y_source == "ΔI from dataset.npz":

                dataset_delta_i_col = st.selectbox(
                    "dataset.npz ΔI column",
                    options=list(range(X.shape[1])),
                    index=0,
                    format_func=lambda j: f"X[:, {j}]",
                    help=(
                        "Default is X[:,0] for the uploaded dataset. "
                        "This is adjustable because dataset feature layouts "
                        "can vary between analysis/software versions."
                    ),
                )

                y_name = (
                    f"ΔI from dataset.npz "
                    f"(X[:, {dataset_delta_i_col}])"
                )

                y_values = np.asarray(
                    X[:, dataset_delta_i_col],
                    dtype=float,
                )

            elif y_source == "Peak segment ΔI from event_fitting":

                y_name = "Peak segment ΔI (from event_fitting)"

                y_values = np.asarray(
                    derived_metrics[
                        "Peak segment ΔI"
                    ],
                    dtype=float,
                )

            elif y_source == "Time-weighted segment ΔI from event_fitting":

                y_name = "Time-weighted segment ΔI (from event_fitting)"

                y_values = np.asarray(
                    derived_metrics[
                        "Time-weighted segment ΔI"
                    ],
                    dtype=float,
                )

            elif y_source == "Number of segments from event_fitting":

                y_name = "Number of segments (from event_fitting)"

                y_values = np.asarray(
                    derived_metrics[
                        "Number of segments"
                    ],
                    dtype=float,
                )

            else:

                raw_columns = [
                    j
                    for j in range(X.shape[1])
                    if j != 4
                ]

                raw_col = st.selectbox(
                    "Raw dataset column",
                    options=raw_columns,
                    format_func=lambda j: f"X[:, {j}]",
                )

                y_name = f"Raw dataset X[:, {raw_col}]"

                y_values = np.asarray(
                    X[:, raw_col],
                    dtype=float,
                )

            # ------------------------------------------------
            # ΔI DISPLAY UNITS
            # ------------------------------------------------
            #
            # For this NanoSense dataset, ΔI-like values are stored on an
            # nA-scale.  The user can display them either in nA or pA.
            # Conversion changes only the plotted units.
            # ------------------------------------------------

            if (
                y_source == "ΔI from dataset.npz"
                or y_source == "Peak segment ΔI from event_fitting"
                or y_source == "Time-weighted segment ΔI from event_fitting"
            ):

                delta_i_unit = st.radio(
                    "ΔI display unit",
                    [
                        "pA",
                        "nA",
                    ],
                    index=0,
                    horizontal=True,
                    key="density_delta_i_unit",
                    help=(
                        "The underlying ΔI values are treated as nA-scale. "
                        "Choosing pA multiplies the plotted values by 1000."
                    ),
                )

                if delta_i_unit == "pA":

                    y_values = (
                        np.asarray(
                            y_values,
                            dtype=float,
                        )
                        * 1000.0
                    )

                    if "dataset.npz" in y_name:

                        y_name = (
                            "ΔI from dataset.npz "
                            f"(X[:, {dataset_delta_i_col}]) (pA)"
                        )

                    elif "Peak segment" in y_name:

                        y_name = (
                            "Peak segment ΔI "
                            "(from event_fitting) (pA)"
                        )

                    else:

                        y_name = (
                            "Time-weighted segment ΔI "
                            "(from event_fitting) (pA)"
                        )

                else:

                    if "dataset.npz" in y_name:

                        y_name = (
                            "ΔI from dataset.npz "
                            f"(X[:, {dataset_delta_i_col}]) (nA)"
                        )

                    elif "Peak segment" in y_name:

                        y_name = (
                            "Peak segment ΔI "
                            "(from event_fitting) (nA)"
                        )

                    else:

                        y_name = (
                            "Time-weighted segment ΔI "
                            "(from event_fitting) (nA)"
                        )

            controls_a, controls_b = st.columns(
                2
            )

            # ------------------------------------------------
            # LEFT CONTROL
            # ------------------------------------------------

            with controls_a:

                population_view = st.radio(
                    "Population",
                    [
                        "All",
                        "Short",
                        "Long",
                    ],
                    horizontal=True,
                )

                st.caption(
                    "Population selection controls the KDE itself: "
                    "**Short** fits only short events, **Long** fits only long "
                    "events, and **All** fits all events. Axis limits change only "
                    "the displayed view."
                )

            # ------------------------------------------------
            # RIGHT CONTROL
            # ------------------------------------------------

            with controls_b:

                log_axis = st.toggle(
                    "Logarithmic dwell-time axis",
                    value=False,
                    key="density2d_log_axis",
                    help=(
                        "Leave this off for the classic ΔI-vs-Δt density view. "
                        "Turn it on only when you want to inspect long dwell-time tails."
                    ),
                )

                grid_size = st.slider(
                    "Density resolution",
                    min_value=120,
                    max_value=320,
                    value=220,
                    step=20,
                )

                kde_bandwidth_scale = st.slider(
                    "KDE smoothing",
                    min_value=0.40,
                    max_value=2.00,
                    value=0.90,
                    step=0.05,
                    help=(
                        "Multiplier on Scott's automatic 2D KDE bandwidth. "
                        "Lower values make the density cloud sharper; higher "
                        "values make it smoother. Around 0.8–1.1 usually gives "
                        "the classic smooth nanopore density appearance."
                    ),
                )

                density_threshold_percent = st.slider(
                    "Background cutoff (% of peak density)",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.15,
                    step=0.05,
                    help=(
                        "Low-density regions below this fraction of the "
                        "peak are hidden to give the glowing density-map look."
                    ),
                )

            # ------------------------------------------------
            # POPULATION SELECTION
            # ------------------------------------------------

            if population_view == "Short":

                selected = short_idx

            elif population_view == "Long":

                selected = long_idx

            else:

                selected = np.arange(
                    n_events
                )

            x_plot = dwell_ms[
                selected
            ]

            y_plot = y_values[
                selected
            ]

            # Keep original event IDs for optional hover inspection.
            event_ids_plot = np.asarray(
                selected,
                dtype=int,
            )

            # ------------------------------------------------
            # REMOVE NaNs
            # ------------------------------------------------

            valid = (
                np.isfinite(
                    x_plot
                )
                &
                np.isfinite(
                    y_plot
                )
            )

            if log_axis:

                valid &= (
                    x_plot > 0
                )

            x_plot = x_plot[
                valid
            ]

            y_plot = y_plot[
                valid
            ]

            event_ids_plot = event_ids_plot[
                valid
            ]

            if len(x_plot) == 0:

                st.warning(
                    "No finite points are available "
                    "for this plot."
                )

            else:

                # --------------------------------------------
                # OPTIONAL OUTLIER DISPLAY FILTER
                # --------------------------------------------

                (
                    p_low,
                    p_high,
                ) = st.slider(
                    "Displayed Y percentile range",
                    min_value=0.0,
                    max_value=100.0,
                    value=(
                        0.0,
                        100.0,
                    ),
                    step=0.5,
                    help=(
                        "Useful if a few extreme outliers squash the "
                        "main density cloud. Keep 0–100% to display everything."
                    ),
                )

                if (
                    p_low > 0.0
                    or p_high < 100.0
                ):

                    (
                        y_low,
                        y_high,
                    ) = np.percentile(
                        y_plot,
                        [
                            p_low,
                            p_high,
                        ],
                    )

                    keep = (
                        (y_plot >= y_low)
                        &
                        (y_plot <= y_high)
                    )

                    x_plot = x_plot[
                        keep
                    ]

                    y_plot = y_plot[
                        keep
                    ]

                    event_ids_plot = event_ids_plot[
                        keep
                    ]

                # --------------------------------------------
                # AXIS CONTROLS
                # --------------------------------------------
                #
                # IMPORTANT:
                # Population selection determines which events are used
                # to fit the KDE:
                #
                #   All   -> all selected events
                #   Short -> dwell <= cutoff
                #   Long  -> dwell > cutoff
                #
                # The axis limits below affect only what is displayed.
                # They never change which events are used to fit the KDE.
                # --------------------------------------------

                density_x_default = (
                    float(np.min(x_plot)),
                    float(np.max(x_plot)),
                )

                density_y_default = (
                    float(np.min(y_plot)),
                    float(np.max(y_plot)),
                )

                density_x_range, density_y_range = axis_limit_controls(
                    "density2d",
                    density_x_default,
                    density_y_default,
                    log_x=log_axis,
                )

                # --------------------------------------------
                # TRUE 2D KDE OF THE SELECTED POPULATION
                # --------------------------------------------

                x_grid, y_grid, density = true_2d_kde_density(
                    x_plot,
                    y_plot,
                    log_x=log_axis,
                    grid_size=grid_size,
                    bandwidth_scale=kde_bandwidth_scale,
                    x_percentiles=(0.2, 99.8),
                    y_percentiles=(0.2, 99.8),
                )

                if density is None:

                    st.warning(
                        "Not enough valid points are available "
                        "to calculate the density map."
                    )

                else:

                    fig, ax = plt.subplots(
                        figsize=(9, 5.4)
                    )

                    # Dark background to match the requested visual style.
                    fig.patch.set_facecolor(
                        "black"
                    )
                    ax.set_facecolor(
                        "black"
                    )

                    threshold = (
                        density_threshold_percent
                        / 100.0
                        * float(
                            np.max(density)
                        )
                    )

                    density_masked = np.ma.masked_where(
                        density <= threshold,
                        density,
                    )

                    mesh = ax.pcolormesh(
                        x_grid,
                        y_grid,
                        density_masked,
                        shading="auto",
                        cmap="magma",
                    )

                    if log_axis:

                        ax.set_xscale(
                            "log"
                        )

                    # ----------------------------------------
                    # CUT-OFF
                    # ----------------------------------------

                    ax.axvline(
                        cutoff_ms,
                        color="white",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.9,
                        label=f"Cutoff = {cutoff_ms:.4f} ms",
                    )

                    # ----------------------------------------
                    # COLOUR BAR
                    # ----------------------------------------

                    cbar = fig.colorbar(
                        mesh,
                        ax=ax,
                    )

                    cbar.set_label(
                        "KDE density",
                        color="white",
                    )

                    cbar.ax.tick_params(
                        colors="white"
                    )

                    # ----------------------------------------
                    # LABELS / DARK THEME
                    # ----------------------------------------

                    ax.set_xlabel(
                        "Dwell time (ms)",
                        color="white",
                    )

                    ax.set_ylabel(
                        y_name,
                        color="white",
                    )

                    ax.set_title(
                        f"{population_view} events",
                        color="white",
                    )

                    ax.tick_params(
                        colors="white"
                    )

                    for spine in ax.spines.values():
                        spine.set_color(
                            "white"
                        )

                    legend = ax.legend(
                        facecolor="black",
                        edgecolor="white",
                        framealpha=0.65,
                    )

                    for text_item in legend.get_texts():
                        text_item.set_color(
                            "white"
                        )

                    fig.tight_layout()

                    apply_axis_limits(
                        ax,
                        density_x_range,
                        density_y_range,
                    )

                    st.pyplot(
                        fig,
                        clear_figure=True,
                    )

                    st.caption(
                        f"The KDE was fitted using **{len(x_plot):,}** events "
                        f"from the **{population_view}** population. "
                        "The dashed line is the dwell-time cutoff used for "
                        "SHORT/LONG export. Axis limits and KDE smoothing affect "
                        "only the visualization."
                    )

                    inspect_events = st.toggle(
                        "Inspect individual events interactively",
                        value=False,
                        key="density_event_inspector",
                        help=(
                            "Keeps the smooth KDE plot above unchanged. "
                            "Turn this on only when you want to hover over "
                            "individual events and read their coordinates."
                        ),
                    )

                    if inspect_events:

                        st.caption(
                            "Hover over a point to see its original event ID, "
                            "dwell time, and Y-axis value. Drag or scroll to zoom."
                        )

                        event_hover_inspector(
                            x_plot,
                            y_plot,
                            event_ids_plot,
                            y_name,
                            log_x=log_axis,
                            x_range=density_x_range,
                            y_range=density_y_range,
                        )


        # ====================================================
        # D. GMM MODEL COMPARISON
        # ====================================================

        elif plot_type == "GMM model comparison":

            st.caption(
                "Compare whether the event distribution is better described "
                "by 1, 2, or 3 Gaussian components. The 1D analysis uses "
                "log10(dwell time). The 2D analysis uses "
                "[log10(dwell time), ΔI] after standardizing both features."
            )

            st.info(
                "BIC values should be compared **within the same feature "
                "space** (1D with 1D, or 2D with 2D). Do not compare the raw "
                "1D BIC number directly with the raw 2D BIC number."
            )

            # ------------------------------------------------
            # CHOOSE ΔI FOR THE 2D GMM
            # ------------------------------------------------

            gmm_y_source = st.selectbox(
                "ΔI source for the 2D GMM",
                [
                    "dataset.npz X[:, 0]",
                    "Peak segment ΔI from event_fitting",
                    "Time-weighted segment ΔI from event_fitting",
                    "Other dataset column",
                ],
                index=0,
                key="gmm_compare_y_source",
                help=(
                    "The 2D model is standardized before fitting, so choosing "
                    "nA versus pA does not alter the clustering."
                ),
            )

            if gmm_y_source == "dataset.npz X[:, 0]":

                gmm_y_values = np.asarray(
                    X[
                        :,
                        0
                    ],
                    dtype=float,
                )

                gmm_y_label = (
                    "ΔI from dataset.npz (X[:, 0])"
                )

            elif gmm_y_source == "Peak segment ΔI from event_fitting":

                gmm_y_values = np.asarray(
                    derived_metrics[
                        "Peak segment ΔI"
                    ],
                    dtype=float,
                )

                gmm_y_label = (
                    "Peak segment ΔI"
                )

            elif gmm_y_source == "Time-weighted segment ΔI from event_fitting":

                gmm_y_values = np.asarray(
                    derived_metrics[
                        "Time-weighted segment ΔI"
                    ],
                    dtype=float,
                )

                gmm_y_label = (
                    "Time-weighted segment ΔI"
                )

            else:

                gmm_raw_columns = [
                    j
                    for j in range(
                        X.shape[
                            1
                        ]
                    )
                    if j != 4
                ]

                gmm_raw_col = st.selectbox(
                    "Dataset column for 2D GMM",
                    options=gmm_raw_columns,
                    index=0,
                    format_func=lambda j: f"X[:, {j}]",
                    key="gmm_compare_raw_col",
                )

                gmm_y_values = np.asarray(
                    X[
                        :,
                        gmm_raw_col
                    ],
                    dtype=float,
                )

                gmm_y_label = (
                    f"dataset.npz X[:, {gmm_raw_col}]"
                )

            # ------------------------------------------------
            # FIT 2D MODELS
            # ------------------------------------------------

            try:

                gmm_2d_comparison = fit_2d_gmm_models(
                    dwell_ms,
                    gmm_y_values,
                    max_components=3,
                )

            except Exception as exc:

                gmm_2d_comparison = None

                st.error(
                    "2D GMM comparison failed: "
                    f"{exc}"
                )

            if (
                gmm_1d_comparison is not None
                and
                gmm_2d_comparison is not None
            ):

                # --------------------------------------------
                # BIC TABLES
                # --------------------------------------------

                one_col, two_col = st.columns(
                    2
                )

                with one_col:

                    st.markdown(
                        "#### 1D · log₁₀(dwell time)"
                    )

                    one_bic_display = (
                        gmm_1d_comparison[
                            "bic"
                        ]
                        .copy()
                    )

                    one_bic_display[
                        "BIC"
                    ] = one_bic_display[
                        "BIC"
                    ].round(
                        1
                    )

                    one_bic_display[
                        "ΔBIC"
                    ] = one_bic_display[
                        "ΔBIC"
                    ].round(
                        1
                    )

                    st.dataframe(
                        one_bic_display,
                        hide_index=True,
                        use_container_width=True,
                    )

                    one_best = int(
                        gmm_1d_comparison[
                            "best_k"
                        ]
                    )

                    st.metric(
                        "Best 1D model",
                        f"{one_best} component"
                        f"{'' if one_best == 1 else 's'}",
                    )

                with two_col:

                    st.markdown(
                        "#### 2D · log₁₀(dwell) + ΔI"
                    )

                    two_bic_display = (
                        gmm_2d_comparison[
                            "bic"
                        ]
                        .copy()
                    )

                    two_bic_display[
                        "BIC"
                    ] = two_bic_display[
                        "BIC"
                    ].round(
                        1
                    )

                    two_bic_display[
                        "ΔBIC"
                    ] = two_bic_display[
                        "ΔBIC"
                    ].round(
                        1
                    )

                    st.dataframe(
                        two_bic_display,
                        hide_index=True,
                        use_container_width=True,
                    )

                    two_best = int(
                        gmm_2d_comparison[
                            "best_k"
                        ]
                    )

                    st.metric(
                        "Best 2D model",
                        f"{two_best} component"
                        f"{'' if two_best == 1 else 's'}",
                    )

                # --------------------------------------------
                # INTERPRETATION
                # --------------------------------------------

                if (
                    one_best == 2
                    and
                    two_best == 2
                ):

                    st.success(
                        "Both feature spaces currently favour a "
                        "**2-component model**."
                    )

                elif (
                    one_best == two_best
                ):

                    st.info(
                        f"Both analyses favour **{one_best} component"
                        f"{'' if one_best == 1 else 's'}**."
                    )

                else:

                    st.warning(
                        "The preferred number of components changes when ΔI "
                        "is added. That means ΔI is revealing structure that "
                        "is not captured by dwell time alone, or vice versa."
                    )

                # --------------------------------------------
                # 1D vs 2D TWO-COMPONENT ASSIGNMENT AGREEMENT
                # --------------------------------------------

                agreement = gmm_agreement_table(
                    dwell_ms,
                    gmm_1d_comparison,
                    gmm_2d_comparison,
                )

                if agreement is not None:

                    st.markdown(
                        "#### If we force 2 components in both models"
                    )

                    agreement_percent = (
                        100.0
                        * agreement[
                            "agreement"
                        ]
                    )

                    agree_col, changed_col = st.columns(
                        2
                    )

                    agree_col.metric(
                        "Same event assignment",
                        f"{agreement_percent:.1f}%",
                    )

                    changed_col.metric(
                        "Events that switch group",
                        f"{len(agreement['changed_ids']):,}",
                    )

                    st.dataframe(
                        agreement[
                            "table"
                        ],
                        use_container_width=True,
                    )

                    st.caption(
                        "Rows are the 1D log-dwell GMM assignments; columns "
                        "are the 2D log-dwell + ΔI assignments. Components are "
                        "ordered by mean dwell time and labelled short-like / "
                        "long-like. This comparison does not alter the current "
                        "SHORT/LONG export."
                    )

                # --------------------------------------------
                # 2D GMM BOUNDARY PLOT
                # --------------------------------------------

                st.markdown(
                    "#### 2D GMM decision boundary"
                )

                gmm_boundary_fig = plot_2d_gmm_boundary(
                    dwell_ms,
                    gmm_y_values,
                    gmm_2d_comparison,
                    gmm_y_label,
                    density_bandwidth_scale=0.9,
                )

                if gmm_boundary_fig is None:

                    st.warning(
                        "The 2D boundary plot could not be generated."
                    )

                else:

                    st.pyplot(
                        gmm_boundary_fig,
                        clear_figure=True,
                    )

                    st.caption(
                        "The glowing background is a 2D KDE visualization. "
                        "The **white curve** is where the two 2D-GMM posterior "
                        "probabilities are equal (50/50). The white × marks are "
                        "the component centres. Unlike the 1D GMM, this boundary "
                        "does not have to be a single dwell-time cutoff."
                    )

                # --------------------------------------------
                # COMPONENT CENTRES
                # --------------------------------------------

                if 2 in gmm_2d_comparison[
                    "models"
                ]:

                    (
                        _labels_2d,
                        means_2d_raw,
                        order_2d,
                    ) = ordered_binary_labels_2d(
                        gmm_2d_comparison[
                            "models"
                        ][2],
                        gmm_2d_comparison[
                            "features"
                        ],
                        gmm_2d_comparison[
                            "scaler"
                        ],
                    )

                    means_rows = []

                    for label_name, component_id in zip(
                        [
                            "Short-like",
                            "Long-like",
                        ],
                        order_2d,
                    ):

                        means_rows.append(
                            {
                                "2D component": label_name,
                                "Geometric mean dwell (ms)": (
                                    10
                                    **
                                    means_2d_raw[
                                        component_id,
                                        0
                                    ]
                                ),
                                gmm_y_label: means_2d_raw[
                                    component_id,
                                    1
                                ],
                            }
                        )

                    means_frame = pd.DataFrame(
                        means_rows
                    )

                    means_frame[
                        "Geometric mean dwell (ms)"
                    ] = means_frame[
                        "Geometric mean dwell (ms)"
                    ].round(
                        4
                    )

                    st.dataframe(
                        means_frame,
                        hide_index=True,
                        use_container_width=True,
                    )

                st.info(
                    "For now, **export is still based on the manually chosen "
                    "1D dwell-time cutoff**. This comparison section is here "
                    "to tell us whether that simple split is actually justified "
                    "before we decide whether to offer 2D-GMM export."
                )


    # ========================================================
    # 3. EXPORT
    # ========================================================

    st.subheader(
        "3 · Export synchronized populations"
    )

    st.write(

        "Each exported population contains its own "
        "**event_data**, **dataset**, and **event_fitting** "
        "NPZ file. IDs are re-numbered consistently from "
        "0 to N−1, and a CSV records the original IDs "
        "for traceability."
    )

    default_stem = clean_stem(
        f_dataset.name
    )

    stem = st.text_input(

        "Output dataset name",

        value=
        default_stem,

        help=
        "This becomes the prefix of the six filtered NPZ files.",
    ).strip()

    stem = (
        stem
        or default_stem
    )

    if (
        not len(short_idx)
        or not len(long_idx)
    ):

        st.warning(

            "Both populations must contain at least one event "
            "before export."
        )

        st.stop()

    # ========================================================
    # BUILD FILTERED FILES
    # ========================================================

    if st.button(
        "Build filtered files",
        type="primary",
    ):

        progress = st.progress(

            0,

            text=
            "Building SHORT population...",
        )

        (
            short_files,
            short_map,
        ) = build_filtered_files(

            short_idx,

            "SHORT",

            event_data,

            dataset,

            event_fitting,
        )

        progress.progress(

            45,

            text=
            "Building LONG population...",
        )

        (
            long_files,
            long_map,
        ) = build_filtered_files(

            long_idx,

            "LONG",

            event_data,

            dataset,

            event_fitting,
        )

        progress.progress(

            85,

            text=
            "Packing ZIP...",
        )

        zip_bytes = make_zip(

            stem,

            cutoff_ms,

            short_files,

            long_files,

            short_map,

            long_map,
        )

        progress.progress(

            100,

            text="Done",
        )

        st.session_state[
            "result_zip"
        ] = zip_bytes

        st.session_state[
            "result_name"
        ] = (

            f"{stem}_"
            f"dwell_split_"
            f"{cutoff_ms:.4f}ms.zip"
        )

        st.session_state[
            "result_summary"
        ] = (

            len(short_idx),

            len(long_idx),

            cutoff_ms,
        )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    if (
        "result_zip"
        in st.session_state
    ):

        (
            n_short,
            n_long,
            used_cutoff,
        ) = st.session_state[
            "result_summary"
        ]

        st.success(

            f"Ready: "
            f"**{n_short:,} short** + "
            f"**{n_long:,} long** events at "
            f"**{used_cutoff:.4f} ms**."
        )

        st.download_button(

            "Download filtered populations (.zip)",

            data=
            st.session_state[
                "result_zip"
            ],

            file_name=
            st.session_state[
                "result_name"
            ],

            mime=
            "application/zip",
        )


if __name__ == "__main__":
    main()
