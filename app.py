"""Streamlit interface for the Nanopore Event Population Splitter."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter

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


# ============================================================
# 2D GMM: DWELL TIME + DELTA I
# ============================================================


def fit_2d_gmm(
    dwell_ms: np.ndarray,
    delta_i: np.ndarray,
):
    """
    Fit a 2-component GMM to [log10(dwell time), Delta I].

    Both features are standardized before fitting so that the numerical
    scale of Delta I cannot dominate the dwell-time feature.

    Components are ordered afterwards by their median dwell time:
    component 0 -> shorter-dwell ("short-like")
    component 1 -> longer-dwell ("long-like")
    """

    dwell_ms = np.asarray(
        dwell_ms,
        dtype=float,
    )

    delta_i = np.asarray(
        delta_i,
        dtype=float,
    )

    if len(dwell_ms) != len(delta_i):
        raise ValueError(
            "Dwell-time and Delta-I arrays must have the same length."
        )

    valid = (
        np.isfinite(dwell_ms)
        & (dwell_ms > 0)
        & np.isfinite(delta_i)
    )

    valid_idx = np.flatnonzero(
        valid
    )

    if len(valid_idx) < 10:
        raise ValueError(
            "At least 10 events with finite dwell time and Delta I "
            "are required for the 2D GMM."
        )

    features = np.column_stack([
        np.log10(
            dwell_ms[valid]
        ),
        delta_i[valid],
    ])

    scaler = StandardScaler()

    features_scaled = scaler.fit_transform(
        features
    )

    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=0,
        n_init=20,
        reg_covar=1e-6,
    )

    raw_labels = model.fit_predict(
        features_scaled
    )

    raw_probabilities = model.predict_proba(
        features_scaled
    )

    # Order the two components by observed median dwell time.
    raw_median_dwell = np.array([
        np.median(
            dwell_ms[valid][raw_labels == component]
        )
        for component in range(2)
    ])

    component_order = np.argsort(
        raw_median_dwell
    )

    short_component = int(
        component_order[0]
    )

    long_component = int(
        component_order[1]
    )

    labels = np.full(
        len(dwell_ms),
        -1,
        dtype=int,
    )

    labels[valid_idx[
        raw_labels == short_component
    ]] = 0

    labels[valid_idx[
        raw_labels == long_component
    ]] = 1

    probability_short = np.full(
        len(dwell_ms),
        np.nan,
        dtype=float,
    )

    probability_long = np.full(
        len(dwell_ms),
        np.nan,
        dtype=float,
    )

    probability_short[valid_idx] = (
        raw_probabilities[
            :,
            short_component,
        ]
    )

    probability_long[valid_idx] = (
        raw_probabilities[
            :,
            long_component,
        ]
    )

    short_idx = np.flatnonzero(
        labels == 0
    )

    long_idx = np.flatnonzero(
        labels == 1
    )

    invalid_idx = np.flatnonzero(
        labels < 0
    )

    confidence = np.full(
        len(dwell_ms),
        np.nan,
        dtype=float,
    )

    confidence[valid_idx] = np.max(
        raw_probabilities,
        axis=1,
    )

    details = {
        "model":
            model,

        "scaler":
            scaler,

        "valid_idx":
            valid_idx,

        "invalid_idx":
            invalid_idx,

        "short_idx":
            short_idx,

        "long_idx":
            long_idx,

        "probability_short":
            probability_short,

        "probability_long":
            probability_long,

        "confidence":
            confidence,

        "short_median_dwell_ms":
            float(
                np.median(
                    dwell_ms[short_idx]
                )
            ),

        "long_median_dwell_ms":
            float(
                np.median(
                    dwell_ms[long_idx]
                )
            ),

        "short_median_delta_i":
            float(
                np.median(
                    delta_i[short_idx]
                )
            ),

        "long_median_delta_i":
            float(
                np.median(
                    delta_i[long_idx]
                )
            ),

        "bic":
            float(
                model.bic(
                    features_scaled
                )
            ),

        "aic":
            float(
                model.aic(
                    features_scaled
                )
            ),
    }

    return details


def make_zip_2d(
    stem: str,
    delta_i_metric: str,
    short_files: dict[str, bytes],
    long_files: dict[str, bytes],
    short_map: pd.DataFrame,
    long_map: pd.DataFrame,
    confidence: np.ndarray,
) -> bytes:
    """
    Package the 2D-GMM populations without pretending there is one
    dwell-time cutoff.
    """

    short_map = short_map.copy()
    long_map = long_map.copy()

    if "original_row_index" in short_map:
        short_map["2D_GMM_assignment_probability"] = [
            confidence[int(i)]
            for i in short_map["original_row_index"]
        ]

    if "original_row_index" in long_map:
        long_map["2D_GMM_assignment_probability"] = [
            confidence[int(i)]
            for i in long_map["original_row_index"]
        ]

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
            mapping.to_csv(
                index=False
            ).encode(
                "utf-8"
            ),
        )

        zf.writestr(
            "split_info.txt",
            (
                "Nanopore 2D-GMM population split\n"
                "Features: log10(dwell time) + Delta I\n"
                f"Delta-I metric: {delta_i_metric}\n"
                "Both features were standardized before GMM fitting.\n"
                "Two full-covariance Gaussian components were fitted.\n"
                "SHORT_2D and LONG_2D are ordered by median dwell time.\n"
                "There is no single dwell-time cutoff for this 2D split.\n"
                f"Short-like events: {len(short_map)}\n"
                f"Long-like events: {len(long_map)}\n\n"
                "Each population contains synchronized event_data, dataset, "
                "and event_fitting NPZ files.\n"
                "Event IDs are re-numbered 0..N-1 within each population.\n"
                "event_id_mapping.csv preserves original event IDs and includes "
                "the 2D-GMM assignment probability.\n"
            ).encode(
                "utf-8"
            ),
        )

    return buffer.getvalue()


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


def smooth_2d_hist_density(
    x: np.ndarray,
    y: np.ndarray,
    log_x: bool = True,
    grid_size: int = 180,
    smoothing_sigma: float = 2.0,
    x_percentiles: tuple[float, float] = (0.5, 99.5),
):
    """
    Build a fast smooth 2D density field.

    A 2D histogram is calculated first and then Gaussian-smoothed. This gives
    the continuous glowing density-map appearance while remaining fast enough
    for thousands of nanopore events in Streamlit.
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

    x_low, x_high = np.percentile(x_work, x_percentiles)
    y_low, y_high = float(np.min(y)), float(np.max(y))

    if not np.isfinite(x_low) or not np.isfinite(x_high) or x_low >= x_high:
        return None, None, None

    if not np.isfinite(y_low) or not np.isfinite(y_high) or y_low >= y_high:
        pad = 0.5 if y_low == y_high else 0.0
        y_low -= pad
        y_high += pad

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

    density, x_edges, y_edges = np.histogram2d(
        x_work,
        y,
        bins=int(grid_size),
        range=[[x_low, x_high], [y_low, y_high]],
    )

    density = gaussian_filter(
        density.T,
        sigma=float(smoothing_sigma),
        mode="nearest",
    )

    if log_x:
        x_edges = 10 ** x_edges

    return x_edges, y_edges, density



# ============================================================
# INTERACTIVE PLOT HELPERS
# ============================================================


def _finite_range(values: np.ndarray, positive_only: bool = False) -> tuple[float, float]:
    """Return a safe finite min/max range for Streamlit axis controls."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if positive_only:
        values = values[values > 0]

    if len(values) == 0:
        return (0.001, 1.0) if positive_only else (0.0, 1.0)

    low = float(np.min(values))
    high = float(np.max(values))

    if np.isclose(low, high):
        pad = max(abs(low) * 0.05, 1e-6)
        low -= pad
        high += pad

        if positive_only:
            low = max(low, np.finfo(float).tiny)

    return low, high


def axis_range_controls(
    prefix: str,
    x_values: np.ndarray,
    y_values: np.ndarray | None = None,
    log_x: bool = False,
):
    """
    Streamlit controls for optional manual axis limits.

    Plotly itself still provides drag zoom, pan, autoscale and reset. These
    controls are useful when an exact numerical viewing range is required.
    """

    x_low, x_high = _finite_range(x_values, positive_only=log_x)

    x_range = None
    y_range = None

    with st.expander("Axis range / zoom controls", expanded=False):
        st.caption(
            "You can also zoom directly on the graph: drag a box to zoom, "
            "scroll to zoom, pan, or use the Plotly toolbar to reset/autoscale."
        )

        set_x = st.checkbox(
            "Set exact X-axis range",
            value=False,
            key=f"{prefix}_set_x_range",
        )

        if set_x:
            col_x1, col_x2 = st.columns(2)

            with col_x1:
                x_min = st.number_input(
                    "X minimum",
                    value=float(x_low),
                    format="%.6g",
                    key=f"{prefix}_x_min",
                )

            with col_x2:
                x_max = st.number_input(
                    "X maximum",
                    value=float(x_high),
                    format="%.6g",
                    key=f"{prefix}_x_max",
                )

            if log_x and (x_min <= 0 or x_max <= 0):
                st.warning("For a logarithmic X-axis, both limits must be > 0.")
            elif x_min >= x_max:
                st.warning("X minimum must be smaller than X maximum.")
            else:
                x_range = (float(x_min), float(x_max))

        if y_values is not None:
            y_low, y_high = _finite_range(y_values, positive_only=False)

            set_y = st.checkbox(
                "Set exact Y-axis range",
                value=False,
                key=f"{prefix}_set_y_range",
            )

            if set_y:
                col_y1, col_y2 = st.columns(2)

                with col_y1:
                    y_min = st.number_input(
                        "Y minimum",
                        value=float(y_low),
                        format="%.6g",
                        key=f"{prefix}_y_min",
                    )

                with col_y2:
                    y_max = st.number_input(
                        "Y maximum",
                        value=float(y_high),
                        format="%.6g",
                        key=f"{prefix}_y_max",
                    )

                if y_min >= y_max:
                    st.warning("Y minimum must be smaller than Y maximum.")
                else:
                    y_range = (float(y_min), float(y_max))

    return x_range, y_range


def apply_axis_ranges(
    fig: go.Figure,
    x_range: tuple[float, float] | None,
    y_range: tuple[float, float] | None,
    log_x: bool,
) -> None:
    """Apply manual Plotly axis limits when requested."""

    if x_range is not None:
        if log_x:
            fig.update_xaxes(
                range=[
                    np.log10(x_range[0]),
                    np.log10(x_range[1]),
                ]
            )
        else:
            fig.update_xaxes(range=list(x_range))

    if y_range is not None:
        fig.update_yaxes(range=list(y_range))


def plotly_config() -> dict:
    """Common Plotly interaction settings."""

    return {
        "scrollZoom": True,
        "displaylogo": False,
        "responsive": True,
        "modeBarButtonsToAdd": ["drawline", "eraseshape"],
    }


def histogram_trace(
    values: np.ndarray,
    bins: np.ndarray,
    name: str,
    log_x: bool,
    opacity: float = 0.85,
) -> go.Bar:
    """Build an interactive histogram trace from explicit bin edges."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if log_x:
        values = values[values > 0]

    counts, edges = np.histogram(values, bins=bins)

    if log_x:
        centres = np.sqrt(edges[:-1] * edges[1:])
    else:
        centres = 0.5 * (edges[:-1] + edges[1:])

    widths = np.diff(edges)

    return go.Bar(
        x=centres,
        y=counts,
        width=widths,
        name=name,
        opacity=opacity,
        customdata=np.column_stack([edges[:-1], edges[1:]]),
        hovertemplate=(
            "Dwell bin: %{customdata[0]:.5g}–%{customdata[1]:.5g} ms"
            "<br>Count: %{y:,}<extra>%{fullData.name}</extra>"
        ),
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
    # PLOTS
    # ========================================================

    with plot_col:

        st.subheader(
            "Event population plots"
        )

        st.caption(
            "All plots are interactive: hover for values, drag to zoom, "
            "scroll to zoom, pan, and use the toolbar to reset/autoscale."
        )

        plot_type = st.radio(
            "Plot type",
            [
                "Dwell histogram",
                "Dwell density (KDE)",
                "2D event density",
            ],
            horizontal=True,
        )

        # ====================================================
        # A. INTERACTIVE HISTOGRAM
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

            else:
                default_width = float(max(dwell_resolution_ms, 0.001))

                bin_width_ms = st.number_input(
                    "Histogram bin width (ms)",
                    min_value=0.001,
                    value=default_width,
                    step=default_width,
                    format="%.5f",
                    key="hist_bin_width",
                    help=(
                        "For short events, use a bin width close to the "
                        "dwell-time sampling resolution. This avoids artificial "
                        "gaps caused by using too many narrow bins."
                    ),
                )

                bins = linear_histogram_edges(hist_values, bin_width_ms)

            fig = go.Figure()

            if hist_view == "Short + Long overlay":
                fig.add_trace(
                    histogram_trace(
                        dwell_ms[short_idx],
                        bins,
                        f"Short ({len(short_idx):,})",
                        log_axis,
                        opacity=0.60,
                    )
                )
                fig.add_trace(
                    histogram_trace(
                        dwell_ms[long_idx],
                        bins,
                        f"Long ({len(long_idx):,})",
                        log_axis,
                        opacity=0.60,
                    )
                )
                y_for_range = np.concatenate([
                    np.histogram(dwell_ms[short_idx], bins=bins)[0],
                    np.histogram(dwell_ms[long_idx], bins=bins)[0],
                ])
            else:
                fig.add_trace(
                    histogram_trace(
                        hist_values,
                        bins,
                        f"{hist_view} ({len(hist_values):,})",
                        log_axis,
                    )
                )
                y_for_range = np.histogram(hist_values, bins=bins)[0]

            fig.add_vline(
                x=cutoff_ms,
                line_dash="dash",
                line_width=2,
                annotation_text=f"cutoff {cutoff_ms:.4f} ms",
                annotation_position="top",
            )

            fig.update_layout(
                title=f"{hist_view} dwell-time distribution",
                xaxis_title="Dwell time (ms)",
                yaxis_title="Count",
                barmode="overlay",
                hovermode="x unified",
                height=500,
                margin=dict(l=40, r=20, t=60, b=45),
            )

            fig.update_xaxes(
                type="log" if log_axis else "linear",
                showgrid=True,
            )

            x_range, y_range = axis_range_controls(
                "hist",
                hist_values,
                y_for_range,
                log_x=log_axis,
            )

            apply_axis_ranges(fig, x_range, y_range, log_axis)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config=plotly_config(),
                key="interactive_histogram",
            )

            if not log_axis:
                st.caption(
                    "The linear histogram uses a physical bin width rather "
                    "than a fixed number of bins. This is especially important "
                    "for very short events, whose dwell times are quantized by "
                    "the acquisition sampling interval."
                )

        # ====================================================
        # B. INTERACTIVE KDE DWELL DENSITY
        # ====================================================

        elif plot_type == "Dwell density (KDE)":

            log_axis = st.toggle(
                "Fit density in log10(dwell time)",
                value=True,
                key="kde_log_axis",
                help=(
                    "Recommended for nanopore dwell times because the "
                    "distribution is usually strongly right-skewed."
                ),
            )

            density_view = st.radio(
                "Show",
                [
                    "Short + Long",
                    "All events",
                ],
                horizontal=True,
                key="kde_population_view",
            )

            fig = go.Figure()
            all_kde_y = []
            all_kde_x = []

            if density_view == "All events":
                x_all, d_all = kde_curve(dwell_ms, log_axis)

                if x_all is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=x_all,
                            y=d_all,
                            mode="lines",
                            name="All events",
                            line=dict(width=3),
                            hovertemplate=(
                                "Dwell: %{x:.5g} ms"
                                "<br>Density: %{y:.5g}<extra>All events</extra>"
                            ),
                        )
                    )
                    all_kde_x.append(x_all)
                    all_kde_y.append(d_all)

            else:
                if len(short_idx) >= 2:
                    x_short, d_short = kde_curve(dwell_ms[short_idx], log_axis)

                    if x_short is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=x_short,
                                y=d_short,
                                mode="lines",
                                name=f"Short ({len(short_idx):,})",
                                line=dict(width=3),
                                hovertemplate=(
                                    "Dwell: %{x:.5g} ms"
                                    "<br>Density: %{y:.5g}<extra>Short</extra>"
                                ),
                            )
                        )
                        all_kde_x.append(x_short)
                        all_kde_y.append(d_short)

                if len(long_idx) >= 2:
                    x_long, d_long = kde_curve(dwell_ms[long_idx], log_axis)

                    if x_long is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=x_long,
                                y=d_long,
                                mode="lines",
                                name=f"Long ({len(long_idx):,})",
                                line=dict(width=3),
                                hovertemplate=(
                                    "Dwell: %{x:.5g} ms"
                                    "<br>Density: %{y:.5g}<extra>Long</extra>"
                                ),
                            )
                        )
                        all_kde_x.append(x_long)
                        all_kde_y.append(d_long)

            fig.add_vline(
                x=cutoff_ms,
                line_dash="dash",
                line_width=2,
                annotation_text=f"cutoff {cutoff_ms:.4f} ms",
                annotation_position="top",
            )

            fig.update_layout(
                title="Dwell-time KDE",
                xaxis_title="Dwell time (ms)",
                yaxis_title=(
                    "KDE density in log10(dwell time)"
                    if log_axis
                    else "KDE density"
                ),
                hovermode="x unified",
                height=500,
                margin=dict(l=40, r=20, t=60, b=45),
            )

            fig.update_xaxes(
                type="log" if log_axis else "linear",
                showgrid=True,
            )

            if all_kde_x:
                x_for_range = np.concatenate(all_kde_x)
                y_for_range = np.concatenate(all_kde_y)
            else:
                x_for_range = dwell_ms
                y_for_range = np.array([0.0, 1.0])

            x_range, y_range = axis_range_controls(
                "kde",
                x_for_range,
                y_for_range,
                log_x=log_axis,
            )

            apply_axis_ranges(fig, x_range, y_range, log_axis)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config=plotly_config(),
                key="interactive_kde",
            )

            st.caption(
                "The KDE is a smoothed view of the dwell-time distribution. "
                "Use it together with the histogram and 2D density plot when "
                "deciding the split."
            )

        # ====================================================
        # C. INTERACTIVE 2D EVENT DENSITY
        # ====================================================

        elif plot_type == "2D event density":

            st.caption(
                "Dwell time is on the x-axis. The density field is "
                "Gaussian-smoothed to give a continuous population map."
            )

            y_options = {
                "Peak segment ΔI (from event_fitting)":
                    derived_metrics["Peak segment ΔI"],
                "Time-weighted segment ΔI (from event_fitting)":
                    derived_metrics["Time-weighted segment ΔI"],
                "Number of segments (from event_fitting)":
                    derived_metrics["Number of segments"],
            }

            for j in range(X.shape[1]):
                if j == 4:
                    continue

                y_options[f"Raw dataset X[:, {j}]"] = X[:, j]

            controls_a, controls_b = st.columns(2)

            with controls_a:
                y_name = st.selectbox(
                    "Y-axis metric",
                    options=list(y_options.keys()),
                    index=0,
                )

                population_view = st.radio(
                    "Population",
                    ["All", "Short", "Long"],
                    horizontal=True,
                    key="density_population_view",
                )

                colourscale = st.selectbox(
                    "Density colour map",
                    ["Magma", "Inferno", "Viridis", "Plasma", "Turbo"],
                    index=0,
                    help="This changes only the visual appearance.",
                )

            with controls_b:
                log_axis = st.toggle(
                    "Logarithmic dwell-time axis",
                    value=True,
                    key="density2d_log_axis",
                )

                grid_size = st.slider(
                    "Density resolution",
                    min_value=80,
                    max_value=260,
                    value=180,
                    step=20,
                )

                smoothing_sigma = st.slider(
                    "Density smoothing",
                    min_value=0.5,
                    max_value=6.0,
                    value=2.0,
                    step=0.25,
                    help=(
                        "Higher values make the population cloud smoother. "
                        "This changes only the visualization, not the event split."
                    ),
                )

                density_threshold_percent = st.slider(
                    "Background cutoff (% of peak density)",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.5,
                    step=0.1,
                    help=(
                        "Low-density regions below this fraction of the peak "
                        "are hidden to give the glowing density-map look."
                    ),
                )

            y_values = np.asarray(y_options[y_name], dtype=float)

            if population_view == "Short":
                selected = short_idx
            elif population_view == "Long":
                selected = long_idx
            else:
                selected = np.arange(n_events)

            x_plot = dwell_ms[selected]
            y_plot = y_values[selected]

            valid = np.isfinite(x_plot) & np.isfinite(y_plot)

            if log_axis:
                valid &= x_plot > 0

            x_plot = x_plot[valid]
            y_plot = y_plot[valid]

            if len(x_plot) == 0:
                st.warning("No finite points are available for this plot.")

            else:
                p_low, p_high = st.slider(
                    "Displayed Y percentile range",
                    min_value=0.0,
                    max_value=100.0,
                    value=(0.0, 100.0),
                    step=0.5,
                    help=(
                        "Useful if a few extreme outliers squash the main "
                        "density cloud. Keep 0–100% to display everything."
                    ),
                )

                if p_low > 0.0 or p_high < 100.0:
                    y_low, y_high = np.percentile(y_plot, [p_low, p_high])
                    keep = (y_plot >= y_low) & (y_plot <= y_high)
                    x_plot = x_plot[keep]
                    y_plot = y_plot[keep]

                x_edges, y_edges, density = smooth_2d_hist_density(
                    x_plot,
                    y_plot,
                    log_x=log_axis,
                    grid_size=grid_size,
                    smoothing_sigma=smoothing_sigma,
                )

                if density is None:
                    st.warning(
                        "Not enough valid points are available to calculate "
                        "the density map."
                    )

                else:
                    threshold = (
                        density_threshold_percent
                        / 100.0
                        * float(np.max(density))
                    )

                    z = density.astype(float).copy()
                    z[z <= threshold] = np.nan

                    if log_axis:
                        x_centres = np.sqrt(x_edges[:-1] * x_edges[1:])
                    else:
                        x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])

                    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])

                    fig = go.Figure(
                        data=go.Heatmap(
                            x=x_centres,
                            y=y_centres,
                            z=z,
                            colorscale=colourscale,
                            zsmooth="best",
                            colorbar=dict(title="Smoothed density"),
                            hovertemplate=(
                                "Dwell: %{x:.5g} ms"
                                "<br>Y: %{y:.5g}"
                                "<br>Density: %{z:.5g}<extra></extra>"
                            ),
                        )
                    )

                    fig.add_vline(
                        x=cutoff_ms,
                        line_dash="dash",
                        line_width=2,
                        line_color="white",
                        annotation_text=f"cutoff {cutoff_ms:.4f} ms",
                        annotation_position="top",
                        annotation_font_color="white",
                    )

                    fig.update_layout(
                        title=f"{population_view} events",
                        xaxis_title="Dwell time (ms)",
                        yaxis_title=y_name,
                        template="plotly_dark",
                        paper_bgcolor="black",
                        plot_bgcolor="black",
                        height=570,
                        margin=dict(l=50, r=20, t=60, b=50),
                    )

                    fig.update_xaxes(
                        type="log" if log_axis else "linear",
                        showgrid=False,
                    )
                    fig.update_yaxes(showgrid=False)

                    x_range, y_range = axis_range_controls(
                        "density2d",
                        x_plot,
                        y_plot,
                        log_x=log_axis,
                    )

                    apply_axis_ranges(fig, x_range, y_range, log_axis)

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config=plotly_config(),
                        key="interactive_2d_density",
                    )

                    st.caption(
                        f"Showing **{len(x_plot):,}** finite events. The dashed "
                        "line is the same dwell-time cutoff used for SHORT/LONG "
                        "export. Smoothing and axis zoom affect only the plot, "
                        "not the underlying event data or population split."
                    )

    # ========================================================
    # 2B. OPTIONAL 2D GMM SPLIT: DWELL + DELTA I
    # ========================================================

    st.divider()

    st.subheader(
        "2B · 2D GMM split: dwell time + ΔI"
    )

    st.caption(
        "This is an additional population analysis. It does not use one "
        "vertical dwell-time cutoff. Each event is classified from its "
        "joint position in log10(dwell time) and ΔI."
    )

    gmm2d_metric_options = {
        "Peak segment ΔI":
            derived_metrics[
                "Peak segment ΔI"
            ],

        "Time-weighted segment ΔI":
            derived_metrics[
                "Time-weighted segment ΔI"
            ],
    }

    gmm2d_metric_name = st.selectbox(
        "ΔI metric used for the 2D GMM",
        options=list(
            gmm2d_metric_options.keys()
        ),
        index=1,
        help=(
            "Time-weighted segment ΔI is the default because it represents "
            "the whole event rather than only the largest segment. You can "
            "also test peak segment ΔI as a robustness comparison."
        ),
    )

    gmm2d_delta_i = np.asarray(
        gmm2d_metric_options[
            gmm2d_metric_name
        ],
        dtype=float,
    )

    try:

        gmm2d = fit_2d_gmm(
            dwell_ms,
            gmm2d_delta_i,
        )

        short_2d_idx = gmm2d[
            "short_idx"
        ]

        long_2d_idx = gmm2d[
            "long_idx"
        ]

        invalid_2d_idx = gmm2d[
            "invalid_idx"
        ]

        gmm2d_col1, gmm2d_col2, gmm2d_col3 = st.columns(
            3
        )

        gmm2d_col1.metric(
            "2D short-like",
            f"{len(short_2d_idx):,}",
        )

        gmm2d_col2.metric(
            "2D long-like",
            f"{len(long_2d_idx):,}",
        )

        gmm2d_col3.metric(
            "Unclassified",
            f"{len(invalid_2d_idx):,}",
        )

        st.write(
            f"**Short-like median dwell:** "
            f"{gmm2d['short_median_dwell_ms']:.4f} ms  \n"
            f"**Long-like median dwell:** "
            f"{gmm2d['long_median_dwell_ms']:.4f} ms  \n"
            f"**Short-like median ΔI:** "
            f"{gmm2d['short_median_delta_i']:.4g}  \n"
            f"**Long-like median ΔI:** "
            f"{gmm2d['long_median_delta_i']:.4g}"
        )

        valid_2d = np.zeros(
            n_events,
            dtype=bool,
        )

        valid_2d[
            gmm2d["valid_idx"]
        ] = True

        one_d_labels = np.where(
            dwell_ms <= cutoff_ms,
            0,
            1,
        )

        two_d_labels = np.full(
            n_events,
            -1,
            dtype=int,
        )

        two_d_labels[
            short_2d_idx
        ] = 0

        two_d_labels[
            long_2d_idx
        ] = 1

        agreement = float(
            np.mean(
                one_d_labels[
                    valid_2d
                ]
                ==
                two_d_labels[
                    valid_2d
                ]
            )
        )

        confidence_valid = gmm2d[
            "confidence"
        ][
            valid_2d
        ]

        median_confidence = float(
            np.median(
                confidence_valid
            )
        )

        low_confidence_fraction = float(
            np.mean(
                confidence_valid < 0.70
            )
        )

        st.caption(
            f"Agreement with the current 1D dwell split: "
            f"**{100 * agreement:.1f}%**. "
            f"Median 2D assignment probability: "
            f"**{100 * median_confidence:.1f}%**. "
            f"Events below 70% assignment probability: "
            f"**{100 * low_confidence_fraction:.1f}%**."
        )

        # ----------------------------------------------------
        # 2D GMM SCATTER
        # ----------------------------------------------------

        fig_2d_gmm = go.Figure()

        fig_2d_gmm.add_trace(
            go.Scattergl(
                x=dwell_ms[
                    short_2d_idx
                ],
                y=gmm2d_delta_i[
                    short_2d_idx
                ],
                mode="markers",
                name=(
                    f"Short-like "
                    f"({len(short_2d_idx):,})"
                ),
                marker=dict(
                    size=5,
                    opacity=0.55,
                ),
                customdata=gmm2d[
                    "probability_short"
                ][
                    short_2d_idx
                ],
                hovertemplate=(
                    "Dwell: %{x:.5g} ms"
                    "<br>ΔI: %{y:.5g}"
                    "<br>P(short-like): %{customdata:.3f}"
                    "<extra></extra>"
                ),
            )
        )

        fig_2d_gmm.add_trace(
            go.Scattergl(
                x=dwell_ms[
                    long_2d_idx
                ],
                y=gmm2d_delta_i[
                    long_2d_idx
                ],
                mode="markers",
                name=(
                    f"Long-like "
                    f"({len(long_2d_idx):,})"
                ),
                marker=dict(
                    size=5,
                    opacity=0.55,
                ),
                customdata=gmm2d[
                    "probability_long"
                ][
                    long_2d_idx
                ],
                hovertemplate=(
                    "Dwell: %{x:.5g} ms"
                    "<br>ΔI: %{y:.5g}"
                    "<br>P(long-like): %{customdata:.3f}"
                    "<extra></extra>"
                ),
            )
        )

        fig_2d_gmm.update_layout(
            title=(
                "2D GMM assignments: "
                "log10(dwell) + ΔI"
            ),
            xaxis_title=
                "Dwell time (ms)",
            yaxis_title=
                gmm2d_metric_name,
            height=600,
            margin=dict(
                l=50,
                r=20,
                t=60,
                b=50,
            ),
        )

        fig_2d_gmm.update_xaxes(
            type="log",
        )

        st.plotly_chart(
            fig_2d_gmm,
            use_container_width=True,
            config=plotly_config(),
            key="gmm_2d_assignment_plot",
        )

        st.caption(
            "The colours above are 2D-GMM assignments. Because ΔI is part "
            "of the clustering, differences in ΔI between these two groups "
            "should not later be treated as an independent finding. Use "
            "topology/segment count or other variables for independent "
            "population comparisons."
        )

        if len(
            invalid_2d_idx
        ):

            st.warning(
                f"{len(invalid_2d_idx):,} events have non-finite "
                f"{gmm2d_metric_name} and therefore cannot be assigned "
                "by the 2D GMM."
            )

    except Exception as exc:

        gmm2d = None

        short_2d_idx = np.array(
            [],
            dtype=int,
        )

        long_2d_idx = np.array(
            [],
            dtype=int,
        )

        invalid_2d_idx = np.arange(
            n_events,
            dtype=int,
        )

        st.warning(
            f"2D GMM could not be fitted: {exc}"
        )

    # ========================================================
    # 3. EXPORT
    # ========================================================

    st.subheader(
        "3 · Export synchronized populations"
    )

    st.write(
        "Choose whether to export the original **1D dwell-time split** "
        "or the new **2D dwell + ΔI GMM split**."
    )

    export_method = st.radio(
        "Population definition used for export",
        [
            "1D dwell-time split",
            "2D GMM: dwell + ΔI",
        ],
        horizontal=True,
    )

    default_stem = clean_stem(
        f_dataset.name
    )

    stem = st.text_input(
        "Output dataset name",
        value=default_stem,
        help=(
            "This becomes the prefix of the filtered NPZ files."
        ),
    ).strip()

    stem = (
        stem
        or default_stem
    )

    if export_method == "1D dwell-time split":

        export_short_idx = short_idx
        export_long_idx = long_idx

        export_short_label = "SHORT"
        export_long_label = "LONG"

        export_ready = (
            len(export_short_idx) > 0
            and len(export_long_idx) > 0
        )

        st.caption(
            f"1D export uses the dwell-time cutoff "
            f"**{cutoff_ms:.4f} ms**."
        )

    else:

        export_short_idx = short_2d_idx
        export_long_idx = long_2d_idx

        export_short_label = "SHORT_2D"
        export_long_label = "LONG_2D"

        export_ready = (
            gmm2d is not None
            and len(export_short_idx) > 0
            and len(export_long_idx) > 0
            and len(invalid_2d_idx) == 0
        )

        if gmm2d is None:

            st.warning(
                "The 2D GMM is unavailable, so a 2D export cannot be built."
            )

        elif len(
            invalid_2d_idx
        ):

            st.error(
                f"2D export is disabled because "
                f"{len(invalid_2d_idx):,} events do not have a finite "
                f"{gmm2d_metric_name}. "
                "I am not silently dropping those events."
            )

        else:

            st.caption(
                f"2D export uses **log10(dwell time) + "
                f"{gmm2d_metric_name}**. "
                "There is no single dwell-time cutoff."
            )

    if not export_ready:

        st.warning(
            "Both export populations must contain events and all required "
            "measurements must be valid before export."
        )

    # ========================================================
    # BUILD FILTERED FILES
    # ========================================================

    if st.button(
        "Build filtered files",
        type="primary",
        disabled=not export_ready,
    ):

        progress = st.progress(
            0,
            text=(
                f"Building "
                f"{export_short_label} population..."
            ),
        )

        (
            export_short_files,
            export_short_map,
        ) = build_filtered_files(
            export_short_idx,
            export_short_label,
            event_data,
            dataset,
            event_fitting,
        )

        progress.progress(
            45,
            text=(
                f"Building "
                f"{export_long_label} population..."
            ),
        )

        (
            export_long_files,
            export_long_map,
        ) = build_filtered_files(
            export_long_idx,
            export_long_label,
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
                export_short_files,
                export_long_files,
                export_short_map,
                export_long_map,
            )

            result_name = (
                f"{stem}_"
                f"dwell_split_"
                f"{cutoff_ms:.4f}ms.zip"
            )

            result_message = (
                f"Ready: **{len(export_short_idx):,} short** + "
                f"**{len(export_long_idx):,} long** events at "
                f"**{cutoff_ms:.4f} ms**."
            )

        else:

            zip_bytes = make_zip_2d(
                stem,
                gmm2d_metric_name,
                export_short_files,
                export_long_files,
                export_short_map,
                export_long_map,
                gmm2d["confidence"],
            )

            result_name = (
                f"{stem}_"
                f"2D_GMM_dwell_deltaI_split.zip"
            )

            result_message = (
                f"Ready: **{len(export_short_idx):,} 2D short-like** + "
                f"**{len(export_long_idx):,} 2D long-like** events using "
                f"**log10(dwell) + {gmm2d_metric_name}**."
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
        ] = result_name

        st.session_state[
            "result_message"
        ] = result_message

    # ========================================================
    # DOWNLOAD
    # ========================================================

    if (
        "result_zip"
        in st.session_state
    ):

        st.success(
            st.session_state[
                "result_message"
            ]
        )

        st.download_button(
            "Download filtered populations (.zip)",
            data=st.session_state[
                "result_zip"
            ],
            file_name=st.session_state[
                "result_name"
            ],
            mime="application/zip",
        )


if __name__ == "__main__":
    main()
    