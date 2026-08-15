"""Streamlit interface for the Nanopore Event Population Splitter."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
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


APP_VERSION = "1.2.0"


# ============================================================
# DERIVED EVENT METRICS
# ============================================================


def derive_event_metrics(event_fitting: dict[str, np.ndarray], n_events: int):
    """
    Derive useful per-event metrics from event_fitting for plotting.

    Returns a dict of arrays with length = n_events.
    """

    peak_segment_abs_delta_i = np.full(n_events, np.nan, dtype=float)
    weighted_segment_abs_delta_i = np.full(n_events, np.nan, dtype=float)
    number_of_segments = np.full(n_events, np.nan, dtype=float)

    for i in range(n_events):
        diff_key = f"SEGMENT_INFO_{i}_segment_mean_diffs"
        width_key = f"SEGMENT_INFO_{i}_segment_widths_time"
        n_key = f"SEGMENT_INFO_{i}_number_of_segments"

        # ----------------------------------------------------
        # Peak segment |ΔI|
        # ----------------------------------------------------
        if diff_key in event_fitting:
            diffs = np.asarray(event_fitting[diff_key], dtype=float).ravel()
            diffs = diffs[np.isfinite(diffs)]

            if diffs.size:
                peak_segment_abs_delta_i[i] = float(np.max(np.abs(diffs)))

        # ----------------------------------------------------
        # Time-weighted mean |ΔI|
        # ----------------------------------------------------
        if diff_key in event_fitting and width_key in event_fitting:
            diffs = np.asarray(event_fitting[diff_key], dtype=float).ravel()
            widths = np.asarray(event_fitting[width_key], dtype=float).ravel()

            n_pair = min(len(diffs), len(widths))
            diffs = diffs[:n_pair]
            widths = widths[:n_pair]

            valid = np.isfinite(diffs) & np.isfinite(widths) & (widths > 0)

            if np.any(valid):
                diffs_valid = np.abs(diffs[valid])
                widths_valid = widths[valid]
                total_width = float(np.sum(widths_valid))

                if total_width > 0:
                    weighted_segment_abs_delta_i[i] = float(
                        np.sum(diffs_valid * widths_valid) / total_width
                    )

        # ----------------------------------------------------
        # Number of segments
        # ----------------------------------------------------
        if n_key in event_fitting:
            value = np.asarray(event_fitting[n_key], dtype=float).ravel()
            if value.size and np.isfinite(value[0]):
                number_of_segments[i] = float(value[0])

    return {
        "Peak segment |ΔI| (from event_fitting)": peak_segment_abs_delta_i,
        "Time-weighted |ΔI| (from event_fitting)": weighted_segment_abs_delta_i,
        "Number of segments (from event_fitting)": number_of_segments,
    }


# ============================================================
# CACHED FILE LOADING
# ============================================================


@st.cache_resource(show_spinner=False)
def load_three_files(
    event_data_bytes: bytes,
    dataset_bytes: bytes,
    event_fitting_bytes: bytes,
):
    """Cache NPZ parsing and derived metrics."""
    event_data = load_npz_bytes(event_data_bytes)
    dataset = load_npz_bytes(dataset_bytes)
    event_fitting = load_npz_bytes(event_fitting_bytes)

    n_events = len(np.asarray(dataset["X"]))
    derived_metrics = derive_event_metrics(event_fitting, n_events)

    return event_data, dataset, event_fitting, derived_metrics


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def silverman_bandwidth(values: np.ndarray) -> float:
    """Robust automatic bandwidth for KDE."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return 0.1

    std = float(np.std(values, ddof=1))
    q25, q75 = np.percentile(values, [25, 75])

    if q75 > q25:
        iqr_sigma = float((q75 - q25) / 1.349)
    else:
        iqr_sigma = np.nan

    candidates = [v for v in (std, iqr_sigma) if np.isfinite(v) and v > 0]

    if candidates:
        sigma = min(candidates)
    else:
        sigma = max(abs(float(np.mean(values))), 1.0)

    bandwidth = 0.9 * sigma * len(values) ** (-1 / 5)

    return max(float(bandwidth), 1e-6)


def kde_curve(values: np.ndarray, log_space: bool, n_points: int = 600):
    """Calculate a smooth 1D KDE curve."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if log_space:
        values = values[values > 0]
        transformed = np.log10(values)
    else:
        transformed = values

    if len(transformed) < 2:
        return None, None

    lo = float(np.min(transformed))
    hi = float(np.max(transformed))

    if np.isclose(lo, hi):
        return None, None

    padding = 0.03 * (hi - lo)
    grid = np.linspace(lo - padding, hi + padding, n_points)

    bandwidth = silverman_bandwidth(transformed)

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(transformed.reshape(-1, 1))

    density = np.exp(kde.score_samples(grid.reshape(-1, 1)))

    if log_space:
        x = 10 ** grid
    else:
        x = grid

    return x, density


def smooth_2d_density(
    x: np.ndarray,
    y: np.ndarray,
    log_x: bool = True,
    grid_size: int = 180,
    bandwidth: float = 0.18,
):
    """
    Calculate a smooth 2D KDE density map for plotting.

    x = dwell time
    y = metric such as |ΔI|
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

    # --------------------------------------------------------
    # Transform x if log-scale
    # --------------------------------------------------------
    if log_x:
        x_work = np.log10(x)
    else:
        x_work = x.copy()

    # --------------------------------------------------------
    # Robust clipping for visualization
    # Avoid extreme outliers crushing the main density cloud
    # --------------------------------------------------------
    x_low, x_high = np.percentile(x_work, [0.5, 99.5])
    y_low, y_high = np.percentile(y, [0.5, 99.5])

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

    # --------------------------------------------------------
    # Standardize before KDE
    # --------------------------------------------------------
    x_mean = float(np.mean(x_work))
    x_std = float(np.std(x_work))
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))

    if x_std == 0:
        x_std = 1.0
    if y_std == 0:
        y_std = 1.0

    x_scaled = (x_work - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std

    points = np.column_stack([x_scaled, y_scaled])

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(points)

    gx = np.linspace(np.min(x_work), np.max(x_work), grid_size)
    gy = np.linspace(np.min(y), np.max(y), grid_size)
    GX, GY = np.meshgrid(gx, gy)

    GX_scaled = (GX - x_mean) / x_std
    GY_scaled = (GY - y_mean) / y_std

    grid_points = np.column_stack([GX_scaled.ravel(), GY_scaled.ravel()])
    log_density = kde.score_samples(grid_points)
    density = np.exp(log_density).reshape(GX.shape)

    if log_x:
        GX_display = 10 ** GX
    else:
        GX_display = GX

    return GX_display, GY, density


def estimate_dwell_resolution_ms(dwell_ms: np.ndarray) -> float:
    """
    Estimate the dwell-time resolution from unique dwell values.
    This helps choose a sensible histogram bin width and avoids fake gaps.
    """
    dwell_ms = np.asarray(dwell_ms, dtype=float)
    dwell_ms = dwell_ms[np.isfinite(dwell_ms)]
    dwell_ms = dwell_ms[dwell_ms > 0]

    if len(dwell_ms) < 2:
        return 0.005

    uniq = np.unique(np.round(dwell_ms, 6))
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        return 0.005

    resolution = float(np.min(diffs))

    # Keep within a sensible practical range
    resolution = max(min(resolution, 0.05), 0.001)
    return resolution


# ============================================================
# MAIN APP
# ============================================================


def main() -> None:
    st.set_page_config(
        page_title="Nanopore Event Population Splitter",
        page_icon="🧬",
        layout="wide",
    )

    st.title("🧬 Nanopore Event Population Splitter")
    st.caption(
        "Split short- and long-dwell event populations while keeping "
        "event_data, dataset, and event_fitting files synchronized."
    )

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:
        st.header("1 · Load matching files")

        f_event_data = st.file_uploader(
            "event_data (.npz)", type=["npz"], key="event_data"
        )
        f_dataset = st.file_uploader(
            "dataset (.npz)", type=["npz"], key="dataset"
        )
        f_fitting = st.file_uploader(
            "event_fitting (.npz)", type=["npz"], key="event_fitting"
        )

        st.divider()
        st.caption(f"Version {APP_VERSION}")

    if not all([f_event_data, f_dataset, f_fitting]):
        st.info("Upload the three matching NPZ files from one dataset to begin.")
        st.stop()

    # ========================================================
    # LOAD + VALIDATE
    # ========================================================

    with st.spinner("Loading and checking event synchronization..."):
        event_data, dataset, event_fitting, derived_metrics = load_three_files(
            f_event_data.getvalue(),
            f_dataset.getvalue(),
            f_fitting.getvalue(),
        )

        try:
            dwell_s, _, notes = validate_inputs(event_data, dataset, event_fitting)
        except Exception as exc:
            st.error(f"Validation failed: {exc}")
            st.stop()

    dwell_ms = np.asarray(dwell_s, dtype=float) * 1000.0
    n_events = len(dwell_ms)
    X = np.asarray(dataset["X"], dtype=float)

    dwell_resolution_ms = estimate_dwell_resolution_ms(dwell_ms)

    st.success(f"Files are synchronized correctly: **{n_events:,} events**.")
    st.caption("Checks: " + "; ".join(notes) + ".")

    # ========================================================
    # GMM SUGGESTION
    # ========================================================

    try:
        suggested_cutoff, gmm = auto_cutoff_gmm(dwell_ms)
    except Exception as exc:
        suggested_cutoff = float(np.median(dwell_ms))
        gmm = None
        st.warning(
            "Automatic two-population suggestion was unavailable "
            f"({exc}). The median is being used only as an initial value."
        )

    # ========================================================
    # LAYOUT
    # ========================================================

    plot_col, control_col = st.columns([2, 1], gap="large")

    # ========================================================
    # CONTROLS
    # ========================================================

    with control_col:
        st.subheader("2 · Choose the dwell-time boundary")
        st.write("**SHORT:** dwell ≤ cutoff  \n**LONG:** dwell > cutoff")

        st.caption(
            "The GMM value is only a starting point. Inspect the histogram, "
            "KDE, and density map and choose the valley that best separates "
            "the populations in that dataset."
        )

        cutoff_ms = st.number_input(
            "Cutoff (ms)",
            min_value=float(dwell_ms.min()),
            max_value=float(dwell_ms.max()),
            value=float(suggested_cutoff),
            step=0.005,
            format="%.5f",
        )

        if gmm is not None:
            st.caption(
                f"GMM suggestion: **{suggested_cutoff:.4f} ms**  \n"
                f"Approx. centres: {gmm['short_geometric_mean_ms']:.4f} ms and "
                f"{gmm['long_geometric_mean_ms']:.4f} ms"
            )

        short_idx = np.flatnonzero(dwell_ms <= cutoff_ms)
        long_idx = np.flatnonzero(dwell_ms > cutoff_ms)

        metric_a, metric_b = st.columns(2)
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

        if len(short_idx) and len(long_idx):
            st.write(
                f"Short median: **{np.median(dwell_ms[short_idx]):.4f} ms**  \n"
                f"Long median: **{np.median(dwell_ms[long_idx]):.4f} ms**"
            )
        else:
            st.warning("Move the cutoff so that both populations contain events.")

        st.caption(
            f"Estimated dwell-time resolution: **{dwell_resolution_ms:.4f} ms**"
        )

    # ========================================================
    # PLOTS
    # ========================================================

    with plot_col:
        st.subheader("Event population plots")

        plot_type = st.radio(
            "Plot type",
            [
                "Dwell histogram",
                "Dwell density (KDE)",
                "2D smooth density",
            ],
            horizontal=True,
        )

        # ----------------------------------------------------
        # 1. HISTOGRAM
        # ----------------------------------------------------
        if plot_type == "Dwell histogram":
            hist_view = st.radio(
                "Show",
                ["All events", "Short + Long overlay"],
                horizontal=True,
            )

            log_axis = st.toggle(
                "Logarithmic dwell-time axis",
                value=True,
                key="hist_log_axis",
            )

            fig, ax = plt.subplots(figsize=(9, 4.8))

            if log_axis:
                n_log_bins = st.slider(
                    "Number of log-spaced bins",
                    min_value=30,
                    max_value=150,
                    value=80,
                    step=5,
                )

                bins = np.logspace(
                    np.log10(dwell_ms.min()),
                    np.log10(dwell_ms.max()),
                    n_log_bins,
                )

                if hist_view == "All events":
                    ax.hist(dwell_ms, bins=bins, alpha=0.85, label="All events")
                else:
                    ax.hist(
                        dwell_ms[short_idx],
                        bins=bins,
                        alpha=0.6,
                        label=f"Short ({len(short_idx):,})",
                    )
                    ax.hist(
                        dwell_ms[long_idx],
                        bins=bins,
                        alpha=0.6,
                        label=f"Long ({len(long_idx):,})",
                    )

                ax.set_xscale("log")

            else:
                bin_width_ms = st.number_input(
                    "Histogram bin width (ms)",
                    min_value=0.001,
                    max_value=float(max(dwell_ms.max() / 5, 0.01)),
                    value=float(dwell_resolution_ms),
                    step=float(dwell_resolution_ms),
                    format="%.5f",
                    help=(
                        "Using a physical bin width avoids fake gaps, especially "
                        "for short events where dwell times are discrete."
                    ),
                )

                bins = np.arange(
                    dwell_ms.min(),
                    dwell_ms.max() + bin_width_ms,
                    bin_width_ms,
                )

                if len(bins) < 2:
                    bins = 50

                if hist_view == "All events":
                    ax.hist(dwell_ms, bins=bins, alpha=0.85, label="All events")
                else:
                    ax.hist(
                        dwell_ms[short_idx],
                        bins=bins,
                        alpha=0.6,
                        label=f"Short ({len(short_idx):,})",
                    )
                    ax.hist(
                        dwell_ms[long_idx],
                        bins=bins,
                        alpha=0.6,
                        label=f"Long ({len(long_idx):,})",
                    )

            ax.axvline(
                cutoff_ms,
                linestyle="--",
                linewidth=2,
                label=f"Cutoff = {cutoff_ms:.4f} ms",
            )
            ax.set_xlabel("Dwell time (ms)")
            ax.set_ylabel("Count")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

            st.caption(
                "If the short-event histogram previously showed gaps, that was "
                "mainly a binning artefact. This version uses a dwell-time bin "
                "width instead of arbitrary 100 bins."
            )

        # ----------------------------------------------------
        # 2. 1D KDE DENSITY
        # ----------------------------------------------------
        elif plot_type == "Dwell density (KDE)":
            density_view = st.radio(
                "Show",
                ["All events", "Short + Long"],
                horizontal=True,
            )

            log_axis = st.toggle(
                "Fit KDE in log10(dwell time)",
                value=True,
                key="kde_log_axis",
                help=(
                    "Recommended for nanopore dwell times because the distribution "
                    "is usually strongly right-skewed."
                ),
            )

            fig, ax = plt.subplots(figsize=(9, 4.8))

            if density_view == "All events":
                x_all, d_all = kde_curve(dwell_ms, log_axis)
                if x_all is not None:
                    ax.plot(x_all, d_all, linewidth=2, label="All events")

            else:
                if len(short_idx) >= 2:
                    x_short, d_short = kde_curve(dwell_ms[short_idx], log_axis)
                    if x_short is not None:
                        ax.plot(
                            x_short,
                            d_short,
                            linewidth=2,
                            label=f"Short ({len(short_idx):,})",
                        )

                if len(long_idx) >= 2:
                    x_long, d_long = kde_curve(dwell_ms[long_idx], log_axis)
                    if x_long is not None:
                        ax.plot(
                            x_long,
                            d_long,
                            linewidth=2,
                            label=f"Long ({len(long_idx):,})",
                        )

            ax.axvline(
                cutoff_ms,
                linestyle="--",
                linewidth=2,
                label=f"Cutoff = {cutoff_ms:.4f} ms",
            )

            if log_axis:
                ax.set_xscale("log")
                ax.set_ylabel("KDE density in log10(dwell time)")
            else:
                ax.set_ylabel("KDE density")

            ax.set_xlabel("Dwell time (ms)")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

            st.caption(
                "This is a smooth 1D density view of the dwell-time distribution. "
                "Use it together with the histogram and 2D map when choosing the cutoff."
            )

        # ----------------------------------------------------
        # 3. 2D SMOOTH DENSITY
        # ----------------------------------------------------
        elif plot_type == "2D smooth density":
            st.caption(
                "Smooth 2D density map with the visual style you asked for. "
                "Dwell time is on the x-axis."
            )

            y_options = {
                "Peak segment |ΔI| (from event_fitting)": derived_metrics[
                    "Peak segment |ΔI| (from event_fitting)"
                ],
                "Time-weighted |ΔI| (from event_fitting)": derived_metrics[
                    "Time-weighted |ΔI| (from event_fitting)"
                ],
                "Number of segments (from event_fitting)": derived_metrics[
                    "Number of segments (from event_fitting)"
                ],
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
                )

            with controls_b:
                log_axis = st.toggle(
                    "Logarithmic dwell-time axis",
                    value=True,
                    key="density2d_log_axis",
                )

                grid_size = st.slider(
                    "Density grid size",
                    min_value=100,
                    max_value=260,
                    value=180,
                    step=20,
                )

                bandwidth = st.slider(
                    "Smoothing",
                    min_value=0.05,
                    max_value=0.40,
                    value=0.18,
                    step=0.01,
                    help="Higher = smoother density cloud.",
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
                GX, GY, density = smooth_2d_density(
                    x_plot,
                    y_plot,
                    log_x=log_axis,
                    grid_size=grid_size,
                    bandwidth=bandwidth,
                )

                if GX is None:
                    st.warning("Not enough valid data to estimate a smooth density.")
                else:
                    fig, ax = plt.subplots(figsize=(9, 5.5))
                    fig.patch.set_facecolor("black")
                    ax.set_facecolor("black")

                    threshold = density.max() * 0.004
                    density_masked = np.ma.masked_where(density < threshold, density)

                    mesh = ax.pcolormesh(
                        GX,
                        GY,
                        density_masked,
                        shading="auto",
                        cmap="magma",
                    )

                    if log_axis:
                        ax.set_xscale("log")

                    ax.axvline(
                        cutoff_ms,
                        color="white",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.9,
                        label=f"Cutoff = {cutoff_ms:.4f} ms",
                    )

                    ax.set_xlabel("Dwell time (ms)", color="white")
                    ax.set_ylabel(y_name, color="white")
                    ax.set_title(f"{population_view} events", color="white")

                    ax.tick_params(colors="white")
                    for spine in ax.spines.values():
                        spine.set_color("white")

                    legend = ax.legend(
                        facecolor="black",
                        edgecolor="white",
                        framealpha=0.6,
                    )
                    for txt in legend.get_texts():
                        txt.set_color("white")

                    cbar = fig.colorbar(mesh, ax=ax)
                    cbar.set_label("Event density", color="white")
                    cbar.ax.yaxis.set_tick_params(color="white")
                    plt.setp(cbar.ax.get_yticklabels(), color="white")

                    fig.tight_layout()
                    st.pyplot(fig, clear_figure=True)

                    st.caption(
                        f"Showing **{len(x_plot):,}** finite events. "
                        "The dashed line is the same dwell-time cutoff used for export."
                    )

    # ========================================================
    # EXPORT
    # ========================================================

    st.subheader("3 · Export synchronized populations")
    st.write(
        "Each exported population contains its own **event_data**, **dataset**, "
        "and **event_fitting** NPZ file. IDs are re-numbered consistently from "
        "0 to N−1, and a CSV records the original IDs for traceability."
    )

    default_stem = clean_stem(f_dataset.name)
    stem = st.text_input(
        "Output dataset name",
        value=default_stem,
        help="This becomes the prefix of the six filtered NPZ files.",
    ).strip()
    stem = stem or default_stem

    if not len(short_idx) or not len(long_idx):
        st.warning("Both populations must contain at least one event before export.")
        st.stop()

    if st.button("Build filtered files", type="primary"):
        progress = st.progress(0, text="Building SHORT population...")
        short_files, short_map = build_filtered_files(
            short_idx, "SHORT", event_data, dataset, event_fitting
        )

        progress.progress(45, text="Building LONG population...")
        long_files, long_map = build_filtered_files(
            long_idx, "LONG", event_data, dataset, event_fitting
        )

        progress.progress(85, text="Packing ZIP...")
        zip_bytes = make_zip(
            stem,
            cutoff_ms,
            short_files,
            long_files,
            short_map,
            long_map,
        )

        progress.progress(100, text="Done")

        st.session_state["result_zip"] = zip_bytes
        st.session_state["result_name"] = (
            f"{stem}_dwell_split_{cutoff_ms:.4f}ms.zip"
        )
        st.session_state["result_summary"] = (
            len(short_idx),
            len(long_idx),
            cutoff_ms,
        )

    if "result_zip" in st.session_state:
        n_short, n_long, used_cutoff = st.session_state["result_summary"]

        st.success(
            f"Ready: **{n_short:,} short** + **{n_long:,} long** events at "
            f"**{used_cutoff:.4f} ms**."
        )

        st.download_button(
            "Download filtered populations (.zip)",
            data=st.session_state["result_zip"],
            file_name=st.session_state["result_name"],
            mime="application/zip",
        )


if __name__ == "__main__":
    main()