"""Streamlit interface for the Nanopore Event Population Splitter."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
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


APP_VERSION = "1.4.3-bic-model-selection"


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


# ============================================================
# 2D GMM HELPERS
# ============================================================


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
        "scaler": scaler,
        "valid": valid,
        "valid_idx": valid_idx,
    }


def fit_2d_gmm(
    dwell_ms: np.ndarray,
    delta_i: np.ndarray,
):
    """
    Fit a two-component Gaussian mixture to
    [log10(dwell time), delta-I].

    Both features are standardized before fitting so that the numerical
    scale of delta-I cannot dominate the dwell-time feature.

    The two fitted components are labelled Short-like and Long-like by
    their median dwell time. This is a 2D population assignment: there is
    no single vertical dwell-time cutoff.
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
            "for the 2D GMM."
        )

    features = np.column_stack(
        [
            np.log10(dwell_ms[valid_idx]),
            delta_i[valid_idx],
        ]
    )

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=0,
        n_init=20,
        reg_covar=1e-6,
    )

    raw_labels = model.fit_predict(features_scaled)
    raw_probabilities = model.predict_proba(features_scaled)

    component_median_dwell = []

    for component in range(2):
        component_values = dwell_ms[
            valid_idx[raw_labels == component]
        ]

        if len(component_values) == 0:
            component_median_dwell.append(np.inf)
        else:
            component_median_dwell.append(
                float(np.median(component_values))
            )

    short_component = int(
        np.argmin(component_median_dwell)
    )
    long_component = int(
        np.argmax(component_median_dwell)
    )

    labels = np.full(
        len(dwell_ms),
        -1,
        dtype=int,
    )

    labels[
        valid_idx[raw_labels == short_component]
    ] = 0

    labels[
        valid_idx[raw_labels == long_component]
    ] = 1

    p_short = np.full(
        len(dwell_ms),
        np.nan,
        dtype=float,
    )

    p_long = np.full(
        len(dwell_ms),
        np.nan,
        dtype=float,
    )

    p_short[valid_idx] = raw_probabilities[
        :,
        short_component,
    ]

    p_long[valid_idx] = raw_probabilities[
        :,
        long_component,
    ]

    # Convert the GMM centres back into the original feature units.
    centres_original = scaler.inverse_transform(
        model.means_
    )

    short_centre = centres_original[
        short_component
    ]

    long_centre = centres_original[
        long_component
    ]

    result = {
        "model": model,
        "scaler": scaler,
        "short_component": short_component,
        "long_component": long_component,
        "labels": labels,
        "p_short": p_short,
        "p_long": p_long,
        "valid": valid,
        "valid_idx": valid_idx,
        "short_idx": np.flatnonzero(labels == 0),
        "long_idx": np.flatnonzero(labels == 1),
        "short_centre_dwell_ms": float(
            10 ** short_centre[0]
        ),
        "long_centre_dwell_ms": float(
            10 ** long_centre[0]
        ),
        "short_centre_delta_i": float(
            short_centre[1]
        ),
        "long_centre_delta_i": float(
            long_centre[1]
        ),
    }

    return result


def make_2d_gmm_zip(
    stem: str,
    delta_i_name: str,
    short_files: dict,
    long_files: dict,
    short_map: pd.DataFrame,
    long_map: pd.DataFrame,
    p_short: np.ndarray,
    p_long: np.ndarray,
    excluded_idx: np.ndarray | None = None,
    dwell_ms: np.ndarray | None = None,
    delta_i: np.ndarray | None = None,
) -> bytes:
    """
    Package synchronized 2D-GMM populations into one ZIP archive.

    Events without valid 2D features are never silently discarded. If the
    user explicitly chooses to continue, they are written to
    excluded_events.csv with their original row indices and the reason they
    could not be classified.
    """

    short_map = short_map.copy()
    long_map = long_map.copy()

    if len(short_map):
        source_rows = short_map[
            "original_row_index"
        ].to_numpy(dtype=int)

        short_map[
            "gmm_p_short_like"
        ] = p_short[source_rows]

        short_map[
            "gmm_p_long_like"
        ] = p_long[source_rows]

    if len(long_map):
        source_rows = long_map[
            "original_row_index"
        ].to_numpy(dtype=int)

        long_map[
            "gmm_p_short_like"
        ] = p_short[source_rows]

        long_map[
            "gmm_p_long_like"
        ] = p_long[source_rows]

    if excluded_idx is None:
        excluded_idx = np.array([], dtype=int)
    else:
        excluded_idx = np.asarray(
            excluded_idx,
            dtype=int,
        )

    excluded_table = pd.DataFrame()

    if len(excluded_idx):

        if dwell_ms is None or delta_i is None:
            raise ValueError(
                "dwell_ms and delta_i are required when excluded events are recorded."
            )

        dwell_ms = np.asarray(
            dwell_ms,
            dtype=float,
        )

        delta_i = np.asarray(
            delta_i,
            dtype=float,
        )

        exclusion_reasons = []

        for idx in excluded_idx:

            reasons = []

            if not np.isfinite(dwell_ms[idx]):
                reasons.append(
                    "non-finite dwell time"
                )

            elif dwell_ms[idx] <= 0:
                reasons.append(
                    "non-positive dwell time"
                )

            if not np.isfinite(delta_i[idx]):
                reasons.append(
                    "non-finite ΔI"
                )

            exclusion_reasons.append(
                "; ".join(reasons)
                if reasons
                else "invalid 2D feature"
            )

        excluded_table = pd.DataFrame(
            {
                "original_row_index": excluded_idx,
                "dwell_time_ms": dwell_ms[excluded_idx],
                "delta_i": delta_i[excluded_idx],
                "exclusion_reason": exclusion_reasons,
            }
        )

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:

        for population, files in (
            ("SHORT_2D", short_files),
            ("LONG_2D", long_files),
        ):

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

        mapping = pd.concat(
            [
                short_map,
                long_map,
            ],
            ignore_index=True,
        )

        zf.writestr(
            "event_id_mapping.csv",
            mapping.to_csv(index=False).encode("utf-8"),
        )

        if len(excluded_table):

            zf.writestr(
                "excluded_events.csv",
                excluded_table.to_csv(index=False).encode("utf-8"),
            )

        excluded_note = (
            f"Excluded/unclassified events: {len(excluded_idx)}\n"
        )

        if len(excluded_idx):

            excluded_note += (
                "These events were not assigned to either 2D population because "
                "one or more required 2D features were invalid. "
                "They are listed in excluded_events.csv.\n"
            )

        zf.writestr(
            "split_info.txt",
            (
                "Nanopore 2D GMM population split\n"
                "Features: log10(dwell time) + ΔI\n"
                f"ΔI source: {delta_i_name}\n"
                "Both features were standardized before fitting.\n"
                "Two-component full-covariance Gaussian mixture model.\n"
                "Components were named Short-like and Long-like according "
                "to their median dwell time.\n"
                "There is no single dwell-time cutoff for this 2D split.\n"
                f"Short-like events: {len(short_map)}\n"
                f"Long-like events: {len(long_map)}\n"
                f"{excluded_note}\n"
                "Each exported population contains synchronized event_data, "
                "dataset, and event_fitting NPZ files.\n"
                "Event IDs are re-numbered 0..N-1 within each population.\n"
                "event_id_mapping.csv preserves original event IDs, row "
                "indices, and GMM posterior assignment probabilities.\n"
            ).encode("utf-8"),
        )

    return buffer.getvalue()


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
        "files synchronized. Includes dwell-time, 2D KDE, and "
        "optional dwell + ΔI GMM population analysis."
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
    # 2B. OPTIONAL 2D GMM SPLIT
    # ========================================================

    st.divider()

    st.subheader(
        "2B · Compare with a 2D GMM: dwell time + ΔI"
    )

    st.caption(
        "This section analyses populations in log10(dwell time) + ΔI space. "
        "The main Short-like/Long-like split still uses two components for "
        "direct comparison and export, while the optional BIC check can ask "
        "how many Gaussian components the data statistically prefer. "
        "The original dwell-only Short/Long split above is left unchanged."
    )

    gmm2d_control_col, gmm2d_result_col = st.columns(
        [1.05, 1.45],
        gap="large",
    )

    with gmm2d_control_col:

        gmm2d_source = st.selectbox(
            "ΔI used for the 2D GMM",
            [
                "ΔI from dataset.npz",
                "Peak segment ΔI from event_fitting",
                "Time-weighted segment ΔI from event_fitting",
            ],
            index=0,
            help=(
                "Choose the event-level blockade feature used together with "
                "log10(dwell time). The GMM standardizes both features before fitting."
            ),
        )

        if gmm2d_source == "ΔI from dataset.npz":

            gmm2d_dataset_columns = [
                j
                for j in range(X.shape[1])
                if j != 4
            ]

            gmm2d_dataset_col = st.selectbox(
                "dataset.npz ΔI column for 2D GMM",
                options=gmm2d_dataset_columns,
                index=(
                    gmm2d_dataset_columns.index(0)
                    if 0 in gmm2d_dataset_columns
                    else 0
                ),
                format_func=lambda j: "ΔI" if j == 0 else f"X[:, {j}]",
                help=(
                    "X[:,0] is the event blockade height ΔI for your current dataset layout. "
                    "Do not choose the dwell-time column X[:,4]."
                ),
            )

            gmm2d_delta_i = np.asarray(
                X[:, gmm2d_dataset_col],
                dtype=float,
            )

            gmm2d_delta_i_name = (
                "ΔI (nA)"
                if gmm2d_dataset_col == 0
                else f"dataset.npz X[:, {gmm2d_dataset_col}]"
            )

        elif gmm2d_source == "Peak segment ΔI from event_fitting":

            gmm2d_delta_i = np.asarray(
                derived_metrics[
                    "Peak segment ΔI"
                ],
                dtype=float,
            )

            gmm2d_delta_i_name = (
                "Peak segment ΔI from event_fitting"
            )

        else:

            gmm2d_delta_i = np.asarray(
                derived_metrics[
                    "Time-weighted segment ΔI"
                ],
                dtype=float,
            )

            gmm2d_delta_i_name = (
                "Time-weighted segment ΔI from event_fitting"
            )

        st.caption(
            "The GMM uses the stored ΔI scale directly. Because both features "
            "are standardized, changing nA to pA would not change the assignments."
        )

        st.divider()

        run_bic_selection = st.checkbox(
            "Check how many 2D Gaussian components BIC prefers",
            value=False,
            key="run_2d_bic_selection_v143",
            help=(
                "Fits K = 1 up to the selected maximum using the same "
                "log10(dwell) + ΔI preprocessing. Lower BIC is better. "
                "This is exploratory model selection and does not prove the "
                "same number of physical nanopore mechanisms."
            ),
        )

        bic_max_components = st.slider(
            "Maximum number of components to test",
            min_value=2,
            max_value=6,
            value=5,
            step=1,
            key="bic_max_components_v143",
            disabled=not run_bic_selection,
        )

        st.caption(
            "BIC adds a penalty for extra model complexity, so it does not "
            "automatically reward adding more and more Gaussian components."
        )

    bic2d_result = None

    if run_bic_selection:

        try:

            with st.spinner(
                "Comparing 2D GMMs with BIC..."
            ):

                bic2d_result = select_2d_gmm_components_bic(
                    dwell_ms,
                    gmm2d_delta_i,
                    max_components=bic_max_components,
                )

        except Exception as exc:

            with gmm2d_result_col:

                st.warning(
                    f"BIC model selection could not be completed: {exc}"
                )

    gmm2d_result = None

    try:

        gmm2d_result = fit_2d_gmm(
            dwell_ms,
            gmm2d_delta_i,
        )

    except Exception as exc:

        with gmm2d_result_col:
            st.warning(
                f"2D GMM could not be fitted: {exc}"
            )

    if bic2d_result is not None:

        with gmm2d_result_col:

            best_k = int(
                bic2d_result["best_k"]
            )

            st.markdown(
                "#### BIC component-number check"
            )

            st.metric(
                "BIC-preferred number of Gaussian components",
                f"K = {best_k}",
            )

            if best_k == 1:

                st.info(
                    "BIC prefers one Gaussian component in this 2D feature space. "
                    "A forced two-component split may therefore be over-splitting "
                    "a single statistical population."
                )

            elif best_k == 2:

                st.success(
                    "BIC prefers two Gaussian components. This supports using the "
                    "current two-component 2D GMM as the simplest preferred model."
                )

            else:

                st.warning(
                    f"BIC prefers **{best_k} Gaussian components**. The current "
                    "Short-like/Long-like analysis below still intentionally fits "
                    "two components for a directly comparable two-population split. "
                    "The BIC result says that two Gaussians may be an oversimplified "
                    "statistical description of this dataset."
                )

            with st.expander(
                "Show BIC values and plot",
                expanded=True,
            ):

                bic_table = bic2d_result[
                    "table"
                ].copy()

                st.dataframe(
                    bic_table.style.format(
                        {
                            "BIC": "{:.1f}",
                            "ΔBIC from best": "{:.1f}",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

                fig_bic, ax_bic = plt.subplots(
                    figsize=(6.6, 4.0)
                )

                ax_bic.plot(
                    bic_table["Components (K)"],
                    bic_table["BIC"],
                    marker="o",
                    linewidth=1.8,
                )

                ax_bic.axvline(
                    best_k,
                    linestyle="--",
                    linewidth=1.4,
                    label=f"Minimum BIC: K = {best_k}",
                )

                ax_bic.set_xlabel(
                    "Number of Gaussian components (K)"
                )

                ax_bic.set_ylabel(
                    "BIC"
                )

                ax_bic.set_title(
                    "2D GMM model selection by BIC"
                )

                ax_bic.set_xticks(
                    bic_table["Components (K)"]
                )

                ax_bic.legend()

                fig_bic.tight_layout()

                st.pyplot(
                    fig_bic,
                    clear_figure=True,
                )

                st.caption(
                    "Lower BIC is better. ΔBIC is measured relative to the best "
                    "candidate model. The comparison uses the same standardized "
                    "[log10(dwell time), ΔI] feature space as the 2D GMM."
                )

            st.caption(
                "Important: BIC chooses the number of Gaussian components that "
                "best balances fit and model complexity. It does not by itself "
                "tell us how many physical DNA-translocation mechanisms exist."
            )

            st.divider()

    if gmm2d_result is not None:

        gmm2d_short_idx = gmm2d_result[
            "short_idx"
        ]

        gmm2d_long_idx = gmm2d_result[
            "long_idx"
        ]

        gmm2d_valid = gmm2d_result[
            "valid"
        ]

        gmm2d_unclassified = int(
            np.sum(~gmm2d_valid)
        )

        with gmm2d_result_col:

            metric_2d_a, metric_2d_b, metric_2d_c = st.columns(3)

            metric_2d_a.metric(
                "2D Short-like",
                f"{len(gmm2d_short_idx):,}",
            )

            metric_2d_a.caption(
                f"{100 * len(gmm2d_short_idx) / n_events:.1f}% of all events"
            )

            metric_2d_b.metric(
                "2D Long-like",
                f"{len(gmm2d_long_idx):,}",
            )

            metric_2d_b.caption(
                f"{100 * len(gmm2d_long_idx) / n_events:.1f}% of all events"
            )

            metric_2d_c.metric(
                "No valid ΔI",
                f"{gmm2d_unclassified:,}",
            )

            if gmm2d_unclassified:
                metric_2d_c.caption(
                    "Not assigned by the 2D model"
                )

            compare_mask = gmm2d_valid

            split_1d_short = (
                dwell_ms <= cutoff_ms
            )

            split_2d_short = (
                gmm2d_result["labels"] == 0
            )

            agreement = float(
                np.mean(
                    split_1d_short[compare_mask]
                    == split_2d_short[compare_mask]
                )
            )

            assignment_probability = np.maximum(
                gmm2d_result["p_short"][compare_mask],
                gmm2d_result["p_long"][compare_mask],
            )

            median_assignment_probability = float(
                np.median(assignment_probability)
            )

            uncertain_fraction = float(
                np.mean(
                    assignment_probability < 0.70
                )
            )

            st.write(
                f"**Agreement with the current 1D dwell split:** "
                f"{100 * agreement:.1f}%  \n"
                f"**Median posterior assignment probability:** "
                f"{100 * median_assignment_probability:.1f}%  \n"
                f"**Events with assignment probability < 70%:** "
                f"{100 * uncertain_fraction:.1f}%"
            )

            st.caption(
                "Approximate 2D GMM centres — "
                f"Short-like: {gmm2d_result['short_centre_dwell_ms']:.4f} ms, "
                f"ΔI = {gmm2d_result['short_centre_delta_i']:.4g}; "
                f"Long-like: {gmm2d_result['long_centre_dwell_ms']:.4f} ms, "
                f"ΔI = {gmm2d_result['long_centre_delta_i']:.4g}."
            )

            st.info(
                "For the 2D split, Short-like and Long-like are cluster names. "
                "There is no single dwell-time cutoff because ΔI also contributes "
                "to each event's assignment."
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
                "2D GMM classification",
                "2D event density",
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
        # C. 2D GMM CLASSIFICATION - MATPLOTLIB
        # ====================================================

        elif plot_type == "2D GMM classification":

            if gmm2d_result is None:

                st.warning(
                    "The 2D GMM is not available for this dataset / ΔI source."
                )

            else:

                st.caption(
                    "Matplotlib view of the 2D GMM assignments. The dashed contour "
                    "is the 50/50 posterior boundary between the two fitted components."
                )

                gmm_plot_log_x = st.toggle(
                    "Logarithmic dwell-time axis",
                    value=False,
                    key="gmm2d_plot_log_x",
                )

                valid_idx = gmm2d_result[
                    "valid_idx"
                ]

                short_plot_idx = gmm2d_result[
                    "short_idx"
                ]

                long_plot_idx = gmm2d_result[
                    "long_idx"
                ]

                x_valid = dwell_ms[
                    valid_idx
                ]

                y_valid = gmm2d_delta_i[
                    valid_idx
                ]

                x_low, x_high = np.percentile(
                    x_valid,
                    [0.2, 99.8],
                )

                y_low, y_high = np.percentile(
                    y_valid,
                    [0.2, 99.8],
                )

                if gmm_plot_log_x:
                    gx = np.geomspace(
                        max(float(x_low), np.finfo(float).tiny),
                        float(x_high),
                        260,
                    )
                else:
                    gx = np.linspace(
                        float(x_low),
                        float(x_high),
                        260,
                    )

                gy = np.linspace(
                    float(y_low),
                    float(y_high),
                    260,
                )

                GX, GY = np.meshgrid(
                    gx,
                    gy,
                )

                grid_features = np.column_stack(
                    [
                        np.log10(GX.ravel()),
                        GY.ravel(),
                    ]
                )

                grid_scaled = gmm2d_result[
                    "scaler"
                ].transform(
                    grid_features
                )

                grid_p_short = gmm2d_result[
                    "model"
                ].predict_proba(
                    grid_scaled
                )[
                    :,
                    gmm2d_result["short_component"],
                ].reshape(
                    GX.shape
                )

                fig, ax = plt.subplots(
                    figsize=(9, 5.4)
                )

                ax.scatter(
                    dwell_ms[short_plot_idx],
                    gmm2d_delta_i[short_plot_idx],
                    s=13,
                    alpha=0.45,
                    label=f"2D Short-like ({len(short_plot_idx):,})",
                )

                ax.scatter(
                    dwell_ms[long_plot_idx],
                    gmm2d_delta_i[long_plot_idx],
                    s=13,
                    alpha=0.45,
                    label=f"2D Long-like ({len(long_plot_idx):,})",
                )

                ax.contour(
                    GX,
                    GY,
                    grid_p_short,
                    levels=[0.5],
                    linewidths=2.0,
                    linestyles="--",
                )

                if gmm_plot_log_x:
                    ax.set_xscale(
                        "log"
                    )

                ax.set_xlabel(
                    "Dwell time (ms)"
                )

                ax.set_ylabel(
                    gmm2d_delta_i_name
                )

                ax.set_title(
                    "2D GMM classification: dwell time + ΔI"
                )

                ax.legend()

                fig.tight_layout()

                gmm2d_x_default = ax.get_xlim()
                gmm2d_y_default = ax.get_ylim()

                gmm2d_x_range, gmm2d_y_range = axis_limit_controls(
                    "gmm2d_classification",
                    gmm2d_x_default,
                    gmm2d_y_default,
                    log_x=gmm_plot_log_x,
                )

                apply_axis_limits(
                    ax,
                    gmm2d_x_range,
                    gmm2d_y_range,
                )

                st.pyplot(
                    fig,
                    clear_figure=True,
                )

                st.caption(
                    "The point colours show the 2D GMM assignment. The dashed "
                    "boundary marks equal posterior probability. Unlike the 1D "
                    "split, this boundary can curve because both dwell time and ΔI "
                    "are used."
                )

        # ====================================================
        # D. 2D EVENT DENSITY
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
                    format_func=lambda j: "ΔI" if j == 0 else f"X[:, {j}]",
                    help=(
                        "X[:,0] is the event blockade height ΔI for the uploaded dataset. "
                        "This is adjustable because dataset feature layouts "
                        "can vary between analysis/software versions."
                    ),
                )

                y_name = (
                    "ΔI"
                    if dataset_delta_i_col == 0
                    else (
                        f"ΔI from dataset.npz "
                        f"(X[:, {dataset_delta_i_col}])"
                    )
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

                    if dataset_delta_i_col == 0 and y_source == "ΔI from dataset.npz":

                        y_name = "ΔI (pA)"

                    elif "dataset.npz" in y_name:

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

                    if dataset_delta_i_col == 0 and y_source == "ΔI from dataset.npz":

                        y_name = "ΔI (nA)"

                    elif "dataset.npz" in y_name:

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


    # ========================================================
    # 3. EXPORT
    # ========================================================

    st.subheader(
        "3 · Export synchronized populations"
    )

    st.write(
        "Choose whether the exported populations are defined by the original "
        "1D dwell-time cutoff or by the optional 2D GMM using dwell time + ΔI. "
        "Each population contains synchronized event_data, dataset, and "
        "event_fitting files."
    )

    export_method = st.radio(
        "Population definition used for export",
        [
            "1D dwell-time split",
            "2D GMM: dwell time + ΔI",
        ],
        horizontal=True,
        key="export_population_definition",
    )

    default_stem = clean_stem(
        f_dataset.name
    )

    stem = st.text_input(
        "Output dataset name",
        value=default_stem,
        help="This becomes the prefix of the filtered NPZ files.",
    ).strip()

    stem = (
        stem
        or default_stem
    )

    export_ready = True
    export_excluded_idx = np.array([], dtype=int)

    if export_method == "1D dwell-time split":

        export_short_idx = short_idx
        export_long_idx = long_idx

        if (
            not len(export_short_idx)
            or not len(export_long_idx)
        ):

            st.warning(
                "Both 1D populations must contain at least one event before export."
            )

            export_ready = False

    else:

        if gmm2d_result is None:

            st.warning(
                "The 2D GMM is not available, so the 2D populations cannot be exported."
            )

            export_ready = False

            export_short_idx = np.array([], dtype=int)
            export_long_idx = np.array([], dtype=int)

        else:

            export_short_idx = gmm2d_result[
                "short_idx"
            ]

            export_long_idx = gmm2d_result[
                "long_idx"
            ]

            export_excluded_idx = np.flatnonzero(
                ~gmm2d_result["valid"]
            )

            n_unclassified_2d = int(
                len(export_excluded_idx)
            )

            if n_unclassified_2d:

                excluded_fraction = (
                    100.0
                    * n_unclassified_2d
                    / n_events
                )

                st.warning(
                    f"{n_unclassified_2d:,} of {n_events:,} events "
                    f"({excluded_fraction:.3f}%) cannot be classified by the "
                    "2D GMM because dwell time and/or ΔI is invalid."
                )

                st.caption(
                    "These events will never be silently assigned to Short-like "
                    "or Long-like. If you continue, they will be omitted from the "
                    "two population NPZ files and recorded separately in "
                    "excluded_events.csv with the reason for exclusion."
                )

                exclude_unclassified_2d = st.checkbox(
                    "Exclude unclassifiable events from the 2D split and continue",
                    value=False,
                    key="allow_2d_unclassified_exclusion_v142",
                    help=(
                        "Only events with valid positive dwell time and finite ΔI "
                        "can be classified by the 2D GMM. Excluded events are "
                        "listed explicitly in the exported ZIP."
                    ),
                )

                if not exclude_unclassified_2d:

                    export_ready = False

            if (
                not len(export_short_idx)
                or not len(export_long_idx)
            ):

                st.warning(
                    "Both 2D populations must contain at least one event before export."
                )

                export_ready = False

    # Use version-specific session keys so stale state from an older app
    # cannot trigger a KeyError after deployment.
    result_zip_key = "result_zip_v142mpl"
    result_name_key = "result_name_v142mpl"
    result_message_key = "result_message_v142mpl"

    if st.button(
        "Build filtered files",
        type="primary",
        disabled=not export_ready,
    ):

        progress = st.progress(
            0,
            text="Building first population...",
        )

        if export_method == "1D dwell-time split":

            short_label = "SHORT"
            long_label = "LONG"

        else:

            short_label = "SHORT_2D"
            long_label = "LONG_2D"

        short_files, short_map = build_filtered_files(
            export_short_idx,
            short_label,
            event_data,
            dataset,
            event_fitting,
        )

        progress.progress(
            45,
            text="Building second population...",
        )

        long_files, long_map = build_filtered_files(
            export_long_idx,
            long_label,
            event_data,
            dataset,
            event_fitting,
        )

        progress.progress(
            85,
            text="Packing ZIP...",
        )

        if export_method == "1D dwell-time split":

            zip_bytes = make_zip(
                stem,
                cutoff_ms,
                short_files,
                long_files,
                short_map,
                long_map,
            )

            result_name = (
                f"{stem}_dwell_split_{cutoff_ms:.4f}ms.zip"
            )

            result_message = (
                f"Ready: **{len(export_short_idx):,} short** + "
                f"**{len(export_long_idx):,} long** events using the "
                f"**{cutoff_ms:.4f} ms** dwell-time boundary."
            )

        else:

            zip_bytes = make_2d_gmm_zip(
                stem,
                gmm2d_delta_i_name,
                short_files,
                long_files,
                short_map,
                long_map,
                gmm2d_result["p_short"],
                gmm2d_result["p_long"],
                excluded_idx=export_excluded_idx,
                dwell_ms=dwell_ms,
                delta_i=gmm2d_delta_i,
            )

            result_name = (
                f"{stem}_2D_GMM_dwell_deltaI_split.zip"
            )

            result_message = (
                f"Ready: **{len(export_short_idx):,} 2D Short-like** + "
                f"**{len(export_long_idx):,} 2D Long-like** events using "
                f"**log10(dwell time) + {gmm2d_delta_i_name}**."
            )

            if len(export_excluded_idx):

                result_message += (
                    f" **{len(export_excluded_idx):,} unclassifiable event(s)** "
                    "were excluded from the two populations and recorded in "
                    "`excluded_events.csv`."
                )

        progress.progress(
            100,
            text="Done",
        )

        st.session_state[
            result_zip_key
        ] = zip_bytes

        st.session_state[
            result_name_key
        ] = result_name

        st.session_state[
            result_message_key
        ] = result_message

    # Only display a result when all state entries exist. This avoids the
    # stale-session KeyError seen after replacing an older Streamlit app.
    if all(
        key in st.session_state
        for key in (
            result_zip_key,
            result_name_key,
            result_message_key,
        )
    ):

        st.success(
            st.session_state[
                result_message_key
            ]
        )

        st.download_button(
            "Download filtered populations (.zip)",
            data=st.session_state[
                result_zip_key
            ],
            file_name=st.session_state[
                result_name_key
            ],
            mime="application/zip",
        )


if __name__ == "__main__":
    main()
