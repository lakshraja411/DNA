"""Streamlit interface for the Nanopore Event Population Splitter."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from splitter_core import (
    auto_cutoff_gmm,
    build_filtered_files,
    clean_stem,
    load_npz_bytes,
    make_zip,
    validate_inputs,
)


APP_VERSION = "1.0.0"


@st.cache_resource(show_spinner=False)
def load_three_files(
    event_data_bytes: bytes,
    dataset_bytes: bytes,
    event_fitting_bytes: bytes,
):
    """Cache NPZ parsing so moving the cutoff does not reload large files."""
    return (
        load_npz_bytes(event_data_bytes),
        load_npz_bytes(dataset_bytes),
        load_npz_bytes(event_fitting_bytes),
    )


def main() -> None:
    st.set_page_config(
        page_title="Nanopore Event Population Splitter",
        page_icon="🧬",
        layout="wide",
    )

    st.title("🧬 Nanopore Event Population Splitter")
    st.caption(
        "Split short- and long-dwell event populations while keeping event_data, "
        "dataset, and event_fitting files synchronized."
    )

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

    with st.spinner("Loading and checking event synchronization..."):
        event_data, dataset, event_fitting = load_three_files(
            f_event_data.getvalue(),
            f_dataset.getvalue(),
            f_fitting.getvalue(),
        )
        try:
            dwell_s, _, notes = validate_inputs(
                event_data, dataset, event_fitting
            )
        except Exception as exc:
            st.error(f"Validation failed: {exc}")
            st.stop()

    dwell_ms = dwell_s * 1000.0
    n_events = len(dwell_ms)

    st.success(f"Files are synchronized correctly: **{n_events:,} events**.")
    st.caption("Checks: " + "; ".join(notes) + ".")

    try:
        suggested_cutoff, gmm = auto_cutoff_gmm(dwell_ms)
    except Exception as exc:
        suggested_cutoff = float(np.median(dwell_ms))
        gmm = None
        st.warning(
            "Automatic two-population suggestion was unavailable "
            f"({exc}). The median is being used only as an initial value."
        )

    plot_col, control_col = st.columns([2, 1], gap="large")

    with control_col:
        st.subheader("2 · Choose the dwell-time boundary")
        st.write(
            "**SHORT:** dwell ≤ cutoff  \n"
            "**LONG:** dwell > cutoff"
        )
        st.caption(
            "The GMM value is only an automatic starting point. For scientific "
            "analysis, inspect the distribution and choose the valley that best "
            "separates the populations in that dataset."
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

    with plot_col:
        st.subheader("Dwell-time distribution")
        log_axis = st.toggle("Logarithmic x-axis", value=True)

        fig, ax = plt.subplots(figsize=(9, 4.8))
        if log_axis:
            bins = np.logspace(
                np.log10(dwell_ms.min()), np.log10(dwell_ms.max()), 80
            )
            ax.hist(dwell_ms, bins=bins)
            ax.set_xscale("log")
        else:
            ax.hist(dwell_ms, bins=100)

        ax.axvline(
            cutoff_ms,
            linestyle="--",
            linewidth=2,
            label=f"cutoff = {cutoff_ms:.4f} ms",
        )
        ax.set_xlabel("Dwell time (ms)")
        ax.set_ylabel("Number of events")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

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
