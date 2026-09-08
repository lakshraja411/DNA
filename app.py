"""Nanopore Analysis — a compact, scientific GMM analysis workspace."""
from __future__ import annotations

import hashlib
import io
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from analysis_core import (
    load_sources, get_feature, select_2d_gmm_components_bic,
    build_bic_selected_population_result, describe, export_results,
)
from plotting import (
    bic_plot, scatter_plot, histogram_plot, kde_plot, density_plot,
    export_bytes, colour,
)

st.set_page_config(page_title="Nanopore Analysis", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {max-width: 1380px; padding-top: 1.45rem; padding-bottom: 3rem;}
h1 {font-size: 1.85rem !important; letter-spacing: -.025em; font-weight: 600;}
h2 {font-size: 1.22rem !important; font-weight: 600; margin-top: .5rem;}
h3 {font-size: 1.02rem !important; font-weight: 600;}
[data-testid="stSidebar"] {border-right: 1px solid #E1E5E9;}
[data-testid="stMetric"] {border: 0; padding: .2rem 0;}
[data-testid="stMetricLabel"] {font-size: .82rem;}
[data-testid="stMetricValue"] {font-size: 1.45rem;}
div[data-baseweb="tab-list"] {gap: 1rem; border-bottom: 1px solid #DCE2E7;}
div[data-baseweb="tab"] {padding: .55rem .15rem; font-size: .92rem;}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {
 background: #286B75; border-color: #286B75; color: white;}
.stButton button[kind="primary"]:hover {background: #1E5660; color: white;}
[data-testid="stAlert"] {border-radius: 5px;}
hr {border-color: #E4E8EB;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_load(dataset_bytes, event_bytes, fitting_bytes, dwell_unit):
    return load_sources(dataset_bytes, event_bytes, fitting_bytes, dwell_unit)


@st.cache_resource(show_spinner=False, max_entries=4)
def cached_fit(dwell, delta, max_k):
    return select_2d_gmm_components_bic(dwell, delta, max_k)


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "analysis"


def figure_editor(key, default_title, xlabel, ylabel, log_x=False, can_log=True):
    """Editable presentation parameters. No plotting option changes the fit."""
    with st.expander("Figure settings", expanded=False):
        title = st.text_input("Title", default_title, key=f"{key}_title")
        c1, c2 = st.columns(2)
        with c1:
            xlab = st.text_input("X-axis label", xlabel, key=f"{key}_xlabel")
            width = st.number_input("Width (inches)", 4., 14., 7.6, .2,
                                    key=f"{key}_width")
        with c2:
            ylab = st.text_input("Y-axis label", ylabel, key=f"{key}_ylabel")
            height = st.number_input("Height (inches)", 3., 10., 5., .2,
                                     key=f"{key}_height")
        font = st.slider("Font size (pt)", 8, 18, 11, key=f"{key}_font")
        grid, legend = st.columns(2)
        with grid:
            show_grid = st.checkbox("Subtle Y grid", False, key=f"{key}_grid")
        with legend:
            show_legend = st.checkbox("Legend", True, key=f"{key}_legend")
        use_limits = st.checkbox("Set axis limits", False, key=f"{key}_limits")
        xmin = xmax = ymin = ymax = None
        if use_limits:
            l1, l2 = st.columns(2)
            with l1:
                xmin_text = st.text_input("X minimum", key=f"{key}_xmin")
                ymin_text = st.text_input("Y minimum", key=f"{key}_ymin")
            with l2:
                xmax_text = st.text_input("X maximum", key=f"{key}_xmax")
                ymax_text = st.text_input("Y maximum", key=f"{key}_ymax")
            try:
                xmin = float(xmin_text) if xmin_text.strip() else None
                xmax = float(xmax_text) if xmax_text.strip() else None
                ymin = float(ymin_text) if ymin_text.strip() else None
                ymax = float(ymax_text) if ymax_text.strip() else None
            except ValueError:
                st.warning("Axis limits must be numeric. Invalid limits were ignored.")
                xmin = xmax = ymin = ymax = None
            if log_x and xmin is not None and xmin <= 0:
                st.warning("A logarithmic X-axis requires a positive minimum.")
                xmin = None
    return dict(title=title, xlabel=xlab, ylabel=ylab, width=width,
                height=height, font_size=font, grid=show_grid,
                legend=show_legend, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


def show_figure(fig, stem):
    st.pyplot(fig, use_container_width=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    for column, fmt, mime in zip(
        (c1, c2, c3), ("png", "pdf", "svg"),
        ("image/png", "application/pdf", "image/svg+xml")
    ):
        with column:
            st.download_button(fmt.upper(), export_bytes(fig, fmt),
                               f"{stem}.{fmt}", mime=mime,
                               use_container_width=True, key=f"download_{stem}_{fmt}")
    plt.close(fig)


st.title("Nanopore Analysis")

with st.sidebar:
    st.markdown("### Data")
    uploaded_dataset = st.file_uploader("Dataset · required", type="npz",
                                       key="dataset_upload")
    with st.expander("Optional companion files"):
        uploaded_event = st.file_uploader("Event data", type="npz", key="event_upload")
        uploaded_fitting = st.file_uploader("Event fitting", type="npz", key="fitting_upload")
    dwell_unit = st.selectbox("Dwell-time unit in dataset", ["s", "ms", "us"],
                              format_func=lambda u: {"s":"Seconds", "ms":"Milliseconds",
                                                     "us":"Microseconds"}[u])
    st.divider()
    st.markdown("### GMM settings")
    max_k = st.slider("Maximum K", 2, 10, 6)
    st.caption("Full covariance · 10 initializations · random seed 0")
    st.caption("v2.1 · Scientific edition")

if uploaded_dataset is None:
    st.info("Upload a dataset.npz file to begin.")
    st.stop()

dataset_bytes = uploaded_dataset.getvalue()
event_bytes = uploaded_event.getvalue() if uploaded_event else None
fitting_bytes = uploaded_fitting.getvalue() if uploaded_fitting else None

try:
    data = cached_load(dataset_bytes, event_bytes, fitting_bytes, dwell_unit)
except Exception as exc:
    st.error(f"Data loading failed: {exc}")
    st.stop()

with st.sidebar:
    sources = ["Dataset ΔI"]
    if fitting_bytes is not None:
        sources += ["Peak segment ΔI", "Time-weighted segment ΔI"]
    feature_source = st.selectbox("ΔI source", sources)
    feature_column = 0
    if feature_source == "Dataset ΔI":
        columns = [j for j in range(data["X"].shape[1]) if j != 4]
        feature_column = st.selectbox("Dataset column", columns,
                                      format_func=lambda j: "X[:,0] · ΔI" if j == 0
                                      else f"X[:,{j}]")
    try:
        delta, delta_label = get_feature(data, feature_source, feature_column)
    except Exception as exc:
        st.error(str(exc))
        st.stop()
    if feature_source == "Dataset ΔI" and feature_column != 0:
        st.caption("Verify the physical units of the selected column.")

data["delta_label"] = delta_label
valid_mask = (np.isfinite(data["dwell_ms"]) & (data["dwell_ms"] > 0) &
              np.isfinite(delta))
n_total = len(delta)
n_valid = int(valid_mask.sum())
n_bad = n_total - n_valid
source_hash = hashlib.sha256(
    dataset_bytes + (event_bytes or b"") + (fitting_bytes or b"") +
    dwell_unit.encode() + feature_source.encode() + str(feature_column).encode()
).hexdigest()

if st.session_state.get("analysis_source") != source_hash:
    st.session_state["analysis_source"] = source_hash
    st.session_state.pop("analysis_fit", None)
    st.session_state.pop("analysis_selected_k", None)
    st.session_state.pop("analysis_export", None)

st.caption(uploaded_dataset.name)
m1, m2, m3 = st.columns(3)
m1.metric("Total events", f"{n_total:,}")
m2.metric("Valid 2D events", f"{n_valid:,}")
m3.metric("Excluded", f"{n_bad:,}")
if n_bad:
    st.warning(f"{n_bad:,} events cannot be fitted because dwell time or ΔI is invalid. "
               "Their original rows will be retained in the export log.")

with st.sidebar:
    fit_clicked = st.button("Fit GMM models", type="primary",
                            disabled=n_valid < 20, use_container_width=True)

if fit_clicked:
    try:
        with st.spinner("Fitting candidate models…"):
            bic = cached_fit(data["dwell_ms"], delta, max_k)
        st.session_state["analysis_fit"] = (source_hash, max_k, bic)
        st.session_state["analysis_selected_k"] = int(bic["best_k"])
        st.session_state.pop("analysis_export", None)
    except Exception as exc:
        st.error(f"GMM fitting failed: {exc}")

fit_state = st.session_state.get("analysis_fit")
has_fit = bool(fit_state and fit_state[0] == source_hash and fit_state[1] == max_k)
bic = fit_state[2] if has_fit else None
result = None
selected_k = None
if has_fit:
    tested = [int(k) for k in bic["table"]["Components (K)"]]
    if st.session_state.get("analysis_selected_k") not in tested:
        st.session_state["analysis_selected_k"] = int(bic["best_k"])
    with st.sidebar:
        selected_k = st.selectbox("Selected K", tested,
                                  key="analysis_selected_k",
                                  format_func=lambda k: f"{k}" +
                                  (" · BIC minimum" if k == bic["best_k"] else ""))
    result = build_bic_selected_population_result(
        bic, data["dwell_ms"], delta, selected_k=selected_k
    )

tab_data, tab_model, tab_export = st.tabs(["Data", "BIC & populations", "Export"])

with tab_data:
    st.subheader("Data overview")
    plot_type = st.selectbox("Plot", ["Dwell distribution", "ΔI distribution", "2D density"],
                             key="overview_type", label_visibility="collapsed")
    log_x = st.checkbox("Logarithmic dwell axis", value=True,
                        key="overview_log") if plot_type != "ΔI distribution" else False
    if plot_type == "2D density":
        bandwidth = st.slider("KDE bandwidth", .5, 2., 1., .1, key="overview_bw")
        spec = figure_editor("overview_2d", "Event density",
                             "Dwell time (ms)", delta_label, log_x)
        if n_valid >= 5:
            try:
                fig = density_plot(data, delta, spec, log_x, bandwidth)
                show_figure(fig, "event_density")
            except Exception as exc:
                st.warning(f"Density calculation unavailable: {exc}")
    else:
        metric = "Dwell time" if plot_type == "Dwell distribution" else "ΔI"
        values = data["dwell_ms"][valid_mask] if metric == "Dwell time" else delta[valid_mask]
        mode = st.selectbox("Y-axis", ["Count", "Density"], key="overview_mode")
        bins = st.slider("Number of bins", 10, 200, 70, 5, key="overview_bins")
        if len(values):
            edges = np.histogram_bin_edges(
                np.log10(values) if log_x else values, bins=bins
            )
            if log_x:
                edges = 10**edges
            spec = figure_editor("overview_hist", plot_type,
                                 "Dwell time (ms)" if metric == "Dwell time" else delta_label,
                                 "Number of events" if mode == "Count" else "Probability density",
                                 log_x)
            # Use the original valid parent population, not an arbitrary cutoff.
            overview_result = {"valid_idx": np.flatnonzero(valid_mask),
                               "cluster_indices": [], "labels": np.full(n_total, -1)}
            fig = histogram_plot(data, overview_result, delta, metric, [-1], mode,
                                 edges, spec, log_x)
            show_figure(fig, "data_distribution")
    with st.expander("Data validation"):
        st.write(f"Dataset shape: {data['X'].shape}")
        st.write(f"Feature: {delta_label}; dwell input unit: {dwell_unit}.")
        st.write(f"Companion files: event data {'loaded' if event_bytes else 'not loaded'}, "
                 f"event fitting {'loaded' if fitting_bytes else 'not loaded'}.")
        st.caption("Only open trusted NanoSense archives; NPZ object arrays require pickle loading.")

with tab_model:
    if not has_fit:
        st.info("Select the input files and click Fit GMM models.")
    else:
        best_k = int(bic["best_k"])
        table = bic["table"].copy()
        chosen_bic = float(table.loc[table["Components (K)"] == selected_k, "BIC"].iloc[0])
        minimum = float(table["BIC"].min())
        a, b, c = st.columns(3)
        a.metric("BIC minimum", f"K = {best_k}")
        b.metric("Selected model", f"K = {selected_k}")
        c.metric("ΔBIC", f"{chosen_bic-minimum:.1f}")
        if best_k == max(tested):
            st.warning("The BIC minimum is at the largest K tested. A wider candidate range "
                       "may be needed before interpreting the preferred component count.")
        if not all(model.converged_ for model in bic["models"].values()):
            st.warning("At least one candidate fit did not converge. Review the fit before "
                       "using its BIC for model selection.")
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            spec = figure_editor("bic_figure", "BIC model selection",
                                 "Number of Gaussian components, K", "BIC")
            fig = bic_plot(table, best_k, selected_k, spec)
            show_figure(fig, "bic_model_selection")
        with right:
            st.markdown("#### Candidate models")
            display_table = table.copy()
            display_table["Converged"] = [bic["models"][k].converged_ for k in tested]
            display_table["Iterations"] = [bic["models"][k].n_iter_ for k in tested]
            st.dataframe(display_table.style.format(
                {"BIC":"{:.1f}", "ΔBIC from best":"{:.1f}"}
            ), hide_index=True, use_container_width=True)
            st.caption("Lower BIC is preferred among the tested models.")
        st.divider()
        st.subheader("Population summary")
        summary = describe(result, data["dwell_ms"], delta)
        st.dataframe(summary.style.format({
            "% of valid events":"{:.1f}",
            "Median dwell (ms)":"{:.4f}",
            "Mean dwell (ms)":"{:.4f}",
            "Mean ΔI (nA)":"{:.4f}",
        }), hide_index=True, use_container_width=True)
        p = result["assignment_probability"]
        st.caption(f"Median posterior assignment probability: {100*np.median(p):.1f}%"
                   f"  ·  Below 70%: {100*np.mean(p < .7):.1f}%")
        st.download_button("Download summary CSV", summary.to_csv(index=False),
                           "cluster_summary.csv", "text/csv")
        st.divider()
        st.subheader("Population plots")
        plot_kind = st.selectbox("Plot", [
            "GMM classification", "Dwell distribution", "ΔI distribution",
            "Dwell KDE", "ΔI KDE", "2D density"
        ], key="population_plot", label_visibility="collapsed")
        is_dwell = "Dwell" in plot_kind or plot_kind in ("GMM classification", "2D density")
        log_plot = st.checkbox("Logarithmic dwell axis", True,
                               key="population_log") if is_dwell else False
        selection = []
        if plot_kind not in ("GMM classification", "2D density"):
            choices = [-1] + list(range(result["n_components"]))
            selection = st.multiselect(
                "Populations", choices, default=list(range(result["n_components"])),
                format_func=lambda k: "All events" if k == -1 else f"Cluster {k+1}",
                key="population_selection"
            )
        if plot_kind == "GMM classification":
            c1, c2 = st.columns(2)
            with c1:
                boundary = st.checkbox("Model boundaries", True)
            with c2:
                point_size = st.slider("Point size", 2, 25, 8)
            spec = figure_editor("classification", "GMM population classification",
                                 "Dwell time (ms)", delta_label, log_plot)
            fig = scatter_plot(data, result, delta, spec, log_plot, boundary, point_size)
            show_figure(fig, "gmm_classification")
        elif plot_kind == "2D density":
            bw = st.slider("KDE bandwidth", .5, 2., 1., .1, key="population_bw")
            spec = figure_editor("population_density", "Event density",
                                 "Dwell time (ms)", delta_label, log_plot)
            try:
                fig = density_plot(data, delta, spec, log_plot, bw)
                show_figure(fig, "population_density")
            except Exception as exc:
                st.warning(f"Density calculation unavailable: {exc}")
        elif selection:
            metric = "Dwell time" if "Dwell" in plot_kind else "ΔI"
            xlabel = "Dwell time (ms)" if metric == "Dwell time" else delta_label
            if "KDE" in plot_kind:
                bw = st.slider("KDE bandwidth", .5, 2., 1., .1, key="cluster_bw")
                spec = figure_editor("cluster_kde", f"{metric} density",
                                     xlabel, "Probability density", log_plot)
                fig = kde_plot(data, result, delta, metric, selection, spec, log_plot, bw)
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    mode = st.selectbox("Y-axis", ["Count", "Density", "Share of all events"],
                                        key="cluster_hist_mode")
                with c2:
                    bins = st.slider("Bins", 10, 250, 70, 5, key="cluster_bins")
                with c3:
                    style = st.selectbox("Style", ["Step", "Filled"], key="hist_style")
                with st.expander("Bin and range controls"):
                    bin_method = st.selectbox("Bin spacing", ["Automatic", "Fixed width"],
                                              key="bin_method")
                    manual_width = None
                    if bin_method == "Fixed width":
                        manual_width = st.number_input("Bin width (original units)",
                                                       min_value=1e-9, value=.005,
                                                       format="%.6f")
                    trim = st.checkbox("Set histogram range", False)
                    lower = upper = None
                    if trim:
                        lower = st.number_input("Range minimum", value=0., format="%.6f")
                        upper = st.number_input("Range maximum", value=1., format="%.6f")
                parent = data["dwell_ms"] if metric == "Dwell time" else delta
                parent = parent[result["valid_idx"]]
                parent = parent[np.isfinite(parent) & ((parent > 0) if log_plot else True)]
                if len(parent):
                    lo, hi = float(np.min(parent)), float(np.max(parent))
                    if trim:
                        if lower >= upper or (log_plot and lower <= 0):
                            st.warning("Invalid histogram range; using the full range.")
                        else:
                            lo, hi = float(lower), float(upper)
                    if hi <= lo:
                        hi = lo + max(abs(lo)*1e-6, 1e-9)
                    if manual_width:
                        if (hi-lo)/manual_width > 10000:
                            st.warning("Too many bins; increase the width or narrow the range.")
                            manual_width = (hi-lo)/10000
                        edges = np.arange(lo, hi+manual_width, manual_width)
                        if len(edges) < 2:
                            edges = np.array([lo, hi])
                    elif log_plot:
                        edges = np.geomspace(lo, hi, bins+1)
                    else:
                        edges = np.linspace(lo, hi, bins+1)
                    ylabel = {"Count":"Number of events", "Density":"Probability density",
                              "Share of all events":"Fraction of valid events per bin"}[mode]
                    spec = figure_editor("cluster_hist", f"{metric} distribution",
                                         xlabel, ylabel, log_plot)
                    fig = histogram_plot(data, result, delta, metric, selection,
                                         mode, edges, spec, log_plot, style)
                else:
                    fig = None
            if fig is not None:
                show_figure(fig, safe_name(plot_kind.lower().replace(" ", "_")))
        if plot_kind not in ("GMM classification", "2D density"):
            st.caption("Common bin edges are used across populations. Density normalizes "
                       "each cluster separately; counts and event shares preserve abundance.")

with tab_export:
    st.subheader("Export")
    if not has_fit:
        st.info("Fit and select a GMM model before exporting.")
    else:
        st.write(f"Selected K: **{selected_k}** · Valid events: **{n_valid:,}**")
        st.caption("The package includes the BIC table, cluster statistics, event mapping, "
                   "posterior probabilities, and available synchronized NPZ files.")
        if n_bad:
            st.checkbox("Exclude invalid 2D events from cluster files and record them "
                        "in excluded_events.csv", key="confirm_exclusions")
        ready = n_bad == 0 or st.session_state.get("confirm_exclusions", False)
        if st.button("Build export ZIP", type="primary", disabled=not ready):
            try:
                with st.spinner("Building cluster files…"):
                    stem = safe_name(uploaded_dataset.name.replace(".dataset.npz", "")
                                     .replace(".npz", ""))
                    payload = export_results(data, bic, result, delta, delta_label, stem)
                st.session_state["analysis_export"] = {
                    "source": source_hash, "K": selected_k, "payload": payload,
                    "name": f"{stem}_GMM_K{selected_k}.zip"
                }
            except Exception as exc:
                st.error(f"Export failed: {exc}")
        saved = st.session_state.get("analysis_export")
        if saved and saved["source"] == source_hash and saved["K"] == selected_k:
            st.download_button("Download analysis ZIP", saved["payload"],
                               saved["name"], "application/zip", type="primary")
        elif saved:
            st.caption("The model selection changed. Rebuild the export to update it.")
