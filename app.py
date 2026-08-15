"""Streamlit interface for the Nanopore Event Population Splitter."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import streamlit as st
from sklearn.neighbors import KernelDensity

from splitter_core import (
    auto_cutoff_gmm,
    build_filtered_files,
    clean_stem,
    load_npz_bytes,
    make_zip,
    validate_inputs,
)


APP_VERSION = "1.1.0"


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

            log_axis = st.toggle(

                "Logarithmic dwell-time axis",

                value=True,

                key="hist_log_axis",
            )

            fig, ax = plt.subplots(
                figsize=(9, 4.8)
            )

            if log_axis:

                bins = np.logspace(

                    np.log10(
                        dwell_ms.min()
                    ),

                    np.log10(
                        dwell_ms.max()
                    ),

                    80,
                )

                ax.hist(
                    dwell_ms,
                    bins=bins,
                )

                ax.set_xscale(
                    "log"
                )

            else:

                ax.hist(
                    dwell_ms,
                    bins=100,
                )

            ax.axvline(

                cutoff_ms,

                linestyle="--",

                linewidth=2,

                label=
                f"Cutoff = "
                f"{cutoff_ms:.4f} ms",
            )

            ax.set_xlabel(
                "Dwell time (ms)"
            )

            ax.set_ylabel(
                "Number of events"
            )

            ax.legend()

            fig.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True,
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

                "Dwell time is on the x-axis. "
                "The default y-axis is peak segment ΔI "
                "calculated directly from event_fitting."
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
            # ADD ALL RAW DATASET COLUMNS
            # ------------------------------------------------

            for j in range(
                X.shape[1]
            ):

                # X[:,4] = dwell time,
                # already used on the x-axis

                if j == 4:
                    continue

                y_options[
                    f"Raw dataset X[:, {j}]"
                ] = X[:, j]

            controls_a, controls_b = st.columns(
                2
            )

            # ------------------------------------------------
            # LEFT CONTROL
            # ------------------------------------------------

            with controls_a:

                y_name = st.selectbox(

                    "Y-axis metric",

                    options=
                    list(
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

            # ------------------------------------------------
            # RIGHT CONTROL
            # ------------------------------------------------

            with controls_b:

                log_axis = st.toggle(

                    "Logarithmic dwell-time axis",

                    value=True,

                    key="density2d_log_axis",
                )

                gridsize = st.slider(

                    "Density resolution",

                    min_value=25,

                    max_value=120,

                    value=65,

                    step=5,
                )

            # ------------------------------------------------
            # GET Y VALUES
            # ------------------------------------------------

            y_values = np.asarray(

                y_options[
                    y_name
                ],

                dtype=float,
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
                        "Useful if a few extreme outliers "
                        "squash the main density cloud. "
                        "Keep 0–100% to display everything."
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

                # --------------------------------------------
                # DRAW HEXBIN DENSITY
                # --------------------------------------------

                fig, ax = plt.subplots(
                    figsize=(9, 5.4)
                )

                hb = ax.hexbin(

                    x_plot,

                    y_plot,

                    gridsize=
                    gridsize,

                    mincnt=1,

                    xscale=(
                        "log"
                        if log_axis
                        else "linear"
                    ),

                    norm=
                    LogNorm(),
                )

                # --------------------------------------------
                # CUT-OFF
                # --------------------------------------------

                ax.axvline(

                    cutoff_ms,

                    linestyle="--",

                    linewidth=2,

                    label=
                    f"Cutoff = "
                    f"{cutoff_ms:.4f} ms",
                )

                # --------------------------------------------
                # COLOUR BAR
                # --------------------------------------------

                cbar = fig.colorbar(
                    hb,
                    ax=ax,
                )

                cbar.set_label(
                    "Events per density bin "
                    "(log colour scale)"
                )

                # --------------------------------------------
                # LABELS
                # --------------------------------------------

                ax.set_xlabel(
                    "Dwell time (ms)"
                )

                ax.set_ylabel(
                    y_name
                )

                ax.set_title(
                    f"{population_view} events"
                )

                ax.legend()

                fig.tight_layout()

                st.pyplot(
                    fig,
                    clear_figure=True,
                )

                st.caption(

                    f"Showing **{len(x_plot):,}** finite events. "

                    "The dashed line is the same dwell-time "
                    "cutoff used for SHORT/LONG export."
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
    