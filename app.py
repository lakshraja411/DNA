"""Streamlit interface for the Nanopore Event Population Splitter."""

from __future__ import annotations

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.neighbors import KernelDensity
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


PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "responsive": True,
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
    ],
}


def finite_values(values: np.ndarray, positive_only: bool = False) -> np.ndarray:
    """Return finite numeric values, optionally keeping only positive values."""
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)

    if positive_only:
        mask &= values > 0

    return values[mask]


def manual_axis_controls(
    key_prefix: str,
    x_values: np.ndarray,
    y_values: np.ndarray | None = None,
    *,
    log_x: bool = False,
):
    """
    Optional manual X/Y axis limits.

    These limits change only the displayed plot. They never alter the
    short/long classification or the exported events.
    """

    x_clean = finite_values(x_values, positive_only=log_x)

    if len(x_clean) == 0:
        return None, None

    x_default_min = float(np.min(x_clean))
    x_default_max = float(np.max(x_clean))

    if np.isclose(x_default_min, x_default_max):
        pad = abs(x_default_min) * 0.05 or 1.0
        x_default_min -= pad
        x_default_max += pad

    x_range = None
    y_range = None

    with st.expander("Set axis limits / zoom", expanded=False):

        st.caption(
            "These controls only change the view. "
            "You can also drag to zoom, scroll to zoom, pan, or reset the axes "
            "directly on the graph."
        )

        set_x = st.checkbox(
            "Set custom X-axis limits",
            value=False,
            key=f"{key_prefix}_set_x",
        )

        if set_x:

            xa, xb = st.columns(2)

            x_step = max(
                abs(x_default_max - x_default_min) / 200.0,
                1e-6,
            )

            with xa:
                x_min = st.number_input(
                    "X minimum",
                    value=x_default_min,
                    step=x_step,
                    format="%.6f",
                    key=f"{key_prefix}_x_min",
                )

            with xb:
                x_max = st.number_input(
                    "X maximum",
                    value=x_default_max,
                    step=x_step,
                    format="%.6f",
                    key=f"{key_prefix}_x_max",
                )

            if log_x and x_min <= 0:
                st.warning(
                    "For a logarithmic X-axis, X minimum must be greater than zero."
                )

            elif x_min >= x_max:
                st.warning("X minimum must be smaller than X maximum.")

            else:
                x_range = (float(x_min), float(x_max))

        if y_values is not None:

            y_clean = finite_values(y_values)

            if len(y_clean):

                y_default_min = float(np.min(y_clean))
                y_default_max = float(np.max(y_clean))

                if np.isclose(y_default_min, y_default_max):
                    pad = abs(y_default_min) * 0.05 or 1.0
                    y_default_min -= pad
                    y_default_max += pad

                set_y = st.checkbox(
                    "Set custom Y-axis limits",
                    value=False,
                    key=f"{key_prefix}_set_y",
                )

                if set_y:

                    ya, yb = st.columns(2)

                    y_step = max(
                        abs(y_default_max - y_default_min) / 200.0,
                        1e-6,
                    )

                    with ya:
                        y_min = st.number_input(
                            "Y minimum",
                            value=y_default_min,
                            step=y_step,
                            format="%.6f",
                            key=f"{key_prefix}_y_min",
                        )

                    with yb:
                        y_max = st.number_input(
                            "Y maximum",
                            value=y_default_max,
                            step=y_step,
                            format="%.6f",
                            key=f"{key_prefix}_y_max",
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


def apply_axis_ranges(
    fig: go.Figure,
    x_range,
    y_range,
    *,
    log_x: bool = False,
):
    """Apply manual display ranges to a Plotly figure."""

    if x_range is not None:

        if log_x:
            fig.update_xaxes(
                range=[
                    np.log10(x_range[0]),
                    np.log10(x_range[1]),
                ]
            )
        else:
            fig.update_xaxes(
                range=list(x_range)
            )

    if y_range is not None:
        fig.update_yaxes(
            range=list(y_range)
        )


def style_matplotlib_like(
    fig: go.Figure,
    *,
    x_title: str,
    y_title: str,
    title: str | None = None,
    log_x: bool = False,
):
    """Make Plotly look close to the original clean Matplotlib figures."""

    fig.update_layout(
        template=None,
        height=470,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            color="black",
            size=13,
        ),
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
        ) if title else None,
        margin=dict(
            l=70,
            r=25,
            t=55 if title else 25,
            b=65,
        ),
        hovermode="closest",
        dragmode="zoom",
        legend=dict(
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
        ),
    )

    fig.update_xaxes(
        title=x_title,
        type="log" if log_x else "linear",
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="black",
        linewidth=1,
        ticks="outside",
        tickcolor="black",
        mirror=False,
    )

    fig.update_yaxes(
        title=y_title,
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="black",
        linewidth=1,
        ticks="outside",
        tickcolor="black",
        mirror=False,
    )


def style_dark_density(
    fig: go.Figure,
    *,
    x_title: str,
    y_title: str,
    title: str,
    log_x: bool = False,
):
    """Keep the smooth dark density-map appearance from the static version."""

    fig.update_layout(
        template=None,
        height=525,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(
            color="white",
            size=13,
        ),
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(color="white"),
        ),
        margin=dict(
            l=80,
            r=30,
            t=60,
            b=70,
        ),
        hovermode="closest",
        dragmode="zoom",
        legend=dict(
            bgcolor="rgba(0,0,0,0.55)",
            bordercolor="rgba(255,255,255,0.4)",
            borderwidth=1,
            font=dict(color="white"),
        ),
    )

    fig.update_xaxes(
        title=x_title,
        type="log" if log_x else "linear",
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="white",
        linewidth=1,
        tickcolor="white",
        tickfont=dict(color="white"),
        title_font=dict(color="white"),
        ticks="outside",
    )

    fig.update_yaxes(
        title=y_title,
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="white",
        linewidth=1,
        tickcolor="white",
        tickfont=dict(color="white"),
        title_font=dict(color="white"),
        ticks="outside",
    )


def add_cutoff_line(
    fig: go.Figure,
    cutoff_ms: float,
    y_min: float,
    y_max: float,
    *,
    dark: bool = False,
):
    """Add the cutoff as a normal trace so it remains visible in the legend."""

    line_color = "white" if dark else "black"

    fig.add_trace(
        go.Scatter(
            x=[cutoff_ms, cutoff_ms],
            y=[y_min, y_max],
            mode="lines",
            line=dict(
                color=line_color,
                dash="dash",
                width=2,
            ),
            name=f"Cutoff = {cutoff_ms:.4f} ms",
            hovertemplate=(
                f"Cutoff: {cutoff_ms:.5f} ms"
                "<extra></extra>"
            ),
        )
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

            # ------------------------------------------------
            # BIN EDGES
            # ------------------------------------------------

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
                    np.log10(
                        np.min(
                            positive_values
                        )
                    ),
                    np.log10(
                        np.max(
                            positive_values
                        )
                    ),
                    n_log_bins,
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
                        "For short events, use a bin width close to the "
                        "dwell-time sampling resolution. This avoids artificial "
                        "gaps caused by using too many narrow bins."
                    ),
                )

                bins = linear_histogram_edges(
                    hist_values,
                    bin_width_ms,
                )

            # ------------------------------------------------
            # BUILD ORIGINAL-STYLE INTERACTIVE HISTOGRAM
            # ------------------------------------------------

            fig = go.Figure()
            all_counts = []

            def add_hist_trace(values, name, opacity, color):

                values = finite_values(
                    values,
                    positive_only=log_axis,
                )

                counts, edges = np.histogram(
                    values,
                    bins=bins,
                )

                if log_axis:
                    centers = np.sqrt(
                        edges[:-1]
                        * edges[1:]
                    )
                else:
                    centers = (
                        edges[:-1]
                        + edges[1:]
                    ) / 2.0

                widths = (
                    edges[1:]
                    - edges[:-1]
                )

                custom = np.column_stack(
                    [
                        edges[:-1],
                        edges[1:],
                    ]
                )

                fig.add_trace(
                    go.Bar(
                        x=centers,
                        y=counts,
                        width=widths,
                        opacity=opacity,
                        name=name,
                        marker=dict(
                            color=color,
                            line=dict(
                                width=0,
                            ),
                        ),
                        customdata=custom,
                        hovertemplate=(
                            "Dwell bin: %{customdata[0]:.5f}–"
                            "%{customdata[1]:.5f} ms"
                            "<br>Count: %{y:,}"
                            "<extra>%{fullData.name}</extra>"
                        ),
                    )
                )

                all_counts.extend(
                    counts.tolist()
                )

            if hist_view == "Short + Long overlay":

                add_hist_trace(
                    dwell_ms[short_idx],
                    f"Short ({len(short_idx):,})",
                    0.55,
                    "#1f77b4",
                )

                add_hist_trace(
                    dwell_ms[long_idx],
                    f"Long ({len(long_idx):,})",
                    0.55,
                    "#ff7f0e",
                )

                fig.update_layout(
                    barmode="overlay"
                )

            else:

                label = (
                    f"{hist_view} "
                    f"({len(hist_values):,})"
                )

                add_hist_trace(
                    hist_values,
                    label,
                    0.85,
                    "#1f77b4",
                )

            max_count = (
                max(all_counts)
                if all_counts
                else 1
            )

            add_cutoff_line(
                fig,
                cutoff_ms,
                0,
                max_count * 1.05,
            )

            style_matplotlib_like(
                fig,
                x_title="Dwell time (ms)",
                y_title="Count",
                title=(
                    f"{hist_view} "
                    "dwell-time distribution"
                ),
                log_x=log_axis,
            )

            x_controls_values = finite_values(
                hist_values,
                positive_only=log_axis,
            )

            x_range, y_range = manual_axis_controls(
                "hist",
                x_controls_values,
                np.asarray(
                    all_counts,
                    dtype=float,
                ),
                log_x=log_axis,
            )

            apply_axis_ranges(
                fig,
                x_range,
                y_range,
                log_x=log_axis,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

            st.caption(
                "Hover over a bar to see the dwell-time interval and count. "
                "Drag or scroll on the plot to zoom. Axis limits only change "
                "the displayed view; they do not change the event split."
            )

            if not log_axis:

                st.caption(
                    "The linear histogram uses a physical bin width rather "
                    "than a fixed number of bins. This is especially important "
                    "for very short events, whose dwell times are quantized by "
                    "the acquisition sampling interval."
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

            fig = go.Figure()

            kde_x_values = []
            kde_y_values = []

            # ------------------------------------------------
            # ALL EVENTS
            # ------------------------------------------------

            if density_view == "All events":

                x_all, d_all = kde_curve(
                    dwell_ms,
                    log_axis,
                )

                if x_all is not None:

                    fig.add_trace(
                        go.Scatter(
                            x=x_all,
                            y=d_all,
                            mode="lines",
                            line=dict(
                                color="#1f77b4",
                                width=2,
                            ),
                            name="All events",
                            hovertemplate=(
                                "Dwell time: %{x:.5f} ms"
                                "<br>Density: %{y:.6g}"
                                "<extra>All events</extra>"
                            ),
                        )
                    )

                    kde_x_values.extend(
                        x_all.tolist()
                    )

                    kde_y_values.extend(
                        d_all.tolist()
                    )

            # ------------------------------------------------
            # SHORT + LONG
            # ------------------------------------------------

            else:

                if len(short_idx) >= 2:

                    x_short, d_short = kde_curve(
                        dwell_ms[
                            short_idx
                        ],
                        log_axis,
                    )

                    if x_short is not None:

                        fig.add_trace(
                            go.Scatter(
                                x=x_short,
                                y=d_short,
                                mode="lines",
                                line=dict(
                                    color="#1f77b4",
                                    width=2,
                                ),
                                name=(
                                    f"Short "
                                    f"({len(short_idx):,})"
                                ),
                                hovertemplate=(
                                    "Dwell time: %{x:.5f} ms"
                                    "<br>Density: %{y:.6g}"
                                    "<extra>Short</extra>"
                                ),
                            )
                        )

                        kde_x_values.extend(
                            x_short.tolist()
                        )

                        kde_y_values.extend(
                            d_short.tolist()
                        )

                if len(long_idx) >= 2:

                    x_long, d_long = kde_curve(
                        dwell_ms[
                            long_idx
                        ],
                        log_axis,
                    )

                    if x_long is not None:

                        fig.add_trace(
                            go.Scatter(
                                x=x_long,
                                y=d_long,
                                mode="lines",
                                line=dict(
                                    color="#ff7f0e",
                                    width=2,
                                ),
                                name=(
                                    f"Long "
                                    f"({len(long_idx):,})"
                                ),
                                hovertemplate=(
                                    "Dwell time: %{x:.5f} ms"
                                    "<br>Density: %{y:.6g}"
                                    "<extra>Long</extra>"
                                ),
                            )
                        )

                        kde_x_values.extend(
                            x_long.tolist()
                        )

                        kde_y_values.extend(
                            d_long.tolist()
                        )

            max_density = (
                max(kde_y_values)
                if kde_y_values
                else 1.0
            )

            add_cutoff_line(
                fig,
                cutoff_ms,
                0,
                max_density * 1.05,
            )

            style_matplotlib_like(
                fig,
                x_title="Dwell time (ms)",
                y_title=(
                    "KDE density in log10(dwell time)"
                    if log_axis
                    else "KDE density"
                ),
                log_x=log_axis,
            )

            x_range, y_range = manual_axis_controls(
                "kde",
                np.asarray(
                    kde_x_values,
                    dtype=float,
                ),
                np.asarray(
                    kde_y_values,
                    dtype=float,
                ),
                log_x=log_axis,
            )

            apply_axis_ranges(
                fig,
                x_range,
                y_range,
                log_x=log_axis,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

            st.caption(
                "Hover over the density curve to read the dwell time and KDE "
                "density. You can drag or scroll to zoom, or enter exact axis "
                "limits above."
            )

        # ====================================================
        # C. 2D EVENT DENSITY
        # ====================================================

        elif plot_type == "2D event density":

            st.caption(
                "Dwell time is on the x-axis. "
                "The density appearance is kept close to the smooth static "
                "version, while an invisible event layer provides exact hover "
                "readouts for the underlying events."
            )

            # ------------------------------------------------
            # AVAILABLE Y METRICS
            # ------------------------------------------------

            y_options = {
                "Peak segment ΔI (from event_fitting)":
                    derived_metrics[
                        "Peak segment ΔI"
                    ],

                "Time-weighted segment ΔI (from event_fitting)":
                    derived_metrics[
                        "Time-weighted segment ΔI"
                    ],

                "Number of segments (from event_fitting)":
                    derived_metrics[
                        "Number of segments"
                    ],
            }

            # ------------------------------------------------
            # ADD RAW DATASET COLUMNS
            # ------------------------------------------------

            for j in range(
                X.shape[1]
            ):

                # X[:,4] is dwell time and is already on the x-axis.
                if j == 4:
                    continue

                y_options[
                    f"Raw dataset X[:, {j}]"
                ] = X[:, j]

            controls_a, controls_b = st.columns(
                2
            )

            with controls_a:

                y_name = st.selectbox(
                    "Y-axis metric",
                    options=list(
                        y_options.keys()
                    ),
                    index=0,
                )

                population_view = st.radio(
                    "Population",
                    [
                        "All",
                        "Short",
                        "Long",
                    ],
                    horizontal=True,
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
                        "Low-density regions below this fraction of the "
                        "peak are hidden to preserve the clean density-map look."
                    ),
                )

            # ------------------------------------------------
            # DATA SELECTION
            # ------------------------------------------------

            y_values = np.asarray(
                y_options[
                    y_name
                ],
                dtype=float,
            )

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

            event_ids = np.asarray(
                selected,
                dtype=int,
            )

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

            event_ids = event_ids[
                valid
            ]

            if len(x_plot) == 0:

                st.warning(
                    "No finite points are available "
                    "for this plot."
                )

            else:

                # --------------------------------------------
                # OPTIONAL DISPLAY FILTER
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

                    y_low, y_high = np.percentile(
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

                    event_ids = event_ids[
                        keep
                    ]

                # --------------------------------------------
                # SMOOTH DENSITY FIELD
                # --------------------------------------------

                x_edges, y_edges, density = smooth_2d_hist_density(
                    x_plot,
                    y_plot,
                    log_x=log_axis,
                    grid_size=grid_size,
                    smoothing_sigma=smoothing_sigma,
                )

                if density is None:

                    st.warning(
                        "Not enough valid points are available "
                        "to calculate the density map."
                    )

                else:

                    threshold = (
                        density_threshold_percent
                        / 100.0
                        * float(
                            np.max(
                                density
                            )
                        )
                    )

                    density_display = density.copy()
                    density_display[
                        density_display
                        <= threshold
                    ] = np.nan

                    # Centres of the density cells.
                    if log_axis:

                        x_centers = np.sqrt(
                            x_edges[:-1]
                            * x_edges[1:]
                        )

                    else:

                        x_centers = (
                            x_edges[:-1]
                            + x_edges[1:]
                        ) / 2.0

                    y_centers = (
                        y_edges[:-1]
                        + y_edges[1:]
                    ) / 2.0

                    fig = go.Figure()

                    # ----------------------------------------
                    # VISUAL DENSITY LAYER
                    # ----------------------------------------

                    fig.add_trace(
                        go.Heatmap(
                            x=x_centers,
                            y=y_centers,
                            z=density_display,
                            colorscale="Magma",
                            showscale=True,
                            hoverinfo="skip",
                            colorbar=dict(
                                title=dict(
                                    text="Smoothed<br>event density",
                                    font=dict(
                                        color="white"
                                    ),
                                ),
                                tickfont=dict(
                                    color="white"
                                ),
                                outlinecolor="white",
                                outlinewidth=0.5,
                            ),
                        )
                    )

                    # ----------------------------------------
                    # INVISIBLE EVENT HOVER LAYER
                    # ----------------------------------------
                    #
                    # This layer is intentionally almost invisible, so the
                    # density plot keeps the same smooth appearance. Hovering
                    # over an actual event reports its coordinates.
                    # ----------------------------------------

                    hover_custom = np.column_stack(
                        [
                            event_ids,
                        ]
                    )

                    fig.add_trace(
                        go.Scattergl(
                            x=x_plot,
                            y=y_plot,
                            mode="markers",
                            marker=dict(
                                size=8,
                                color="rgba(255,255,255,0.01)",
                                line=dict(
                                    width=0,
                                ),
                            ),
                            customdata=hover_custom,
                            name="Events",
                            showlegend=False,
                            hovertemplate=(
                                "Event ID: %{customdata[0]:.0f}"
                                "<br>Dwell time: %{x:.5f} ms"
                                f"<br>{y_name}: "
                                "%{y:.6g}"
                                "<extra></extra>"
                            ),
                        )
                    )

                    add_cutoff_line(
                        fig,
                        cutoff_ms,
                        float(
                            np.min(
                                y_plot
                            )
                        ),
                        float(
                            np.max(
                                y_plot
                            )
                        ),
                        dark=True,
                    )

                    style_dark_density(
                        fig,
                        x_title="Dwell time (ms)",
                        y_title=y_name,
                        title=(
                            f"{population_view} events"
                        ),
                        log_x=log_axis,
                    )

                    x_range, y_range = manual_axis_controls(
                        "density2d",
                        x_plot,
                        y_plot,
                        log_x=log_axis,
                    )

                    apply_axis_ranges(
                        fig,
                        x_range,
                        y_range,
                        log_x=log_axis,
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )

                    st.caption(
                        f"Showing **{len(x_plot):,}** finite events. "
                        "Hover over an event to see its original event ID, "
                        "dwell time, and Y-axis value. Manual axis limits and "
                        "mouse zoom change only the view, not the split or export."
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
    