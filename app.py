"""Nanopore Studio — BIC-guided event population analysis."""
from __future__ import annotations

import hashlib
import io
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import gaussian_kde

from nanopore_engine import (
    build_bic_selected_population_result,
    descriptive_table,
    derive_event_metrics,
    event_assignment_table,
    export_bundle,
    read_sources,
    reclassify_segment_counts_by_similarity,
    select_2d_gmm_components_bic,
    topology_statistics,
)
from splitter_core import clean_stem

VERSION = "2.0.0"
COLORS = [
    "#3CB8AE", "#E9A55D", "#8B8BEA", "#5D9FE6", "#E27586",
    "#83B966", "#C78CCC", "#D5BD65", "#67B9D5", "#AD9F93",
]

st.set_page_config(
    page_title="Nanopore Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { color-scheme: dark; }
.block-container { max-width: 1480px; padding-top: 1.65rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { border-right: 1px solid #263748; }
h1, h2, h3 { letter-spacing: -.025em; }
.stTabs [data-baseweb="tab-list"] { gap: .45rem; border-bottom: 1px solid #2A3B4C; }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: .6rem 1rem; }
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {
    background: #287F7B; border: 1px solid #3A9994; color: white;
}
.stButton button[kind="primary"]:hover, .stDownloadButton button[kind="primary"]:hover {
    background: #32918B; color: white;
}
.np-eyebrow { color: #80BDB7; font-size: .72rem; font-weight: 700;
    letter-spacing: .13em; text-transform: uppercase; margin-bottom: .4rem; }
.np-subtitle { color: #AABBC9; font-size: 1rem; line-height: 1.65; max-width: 780px; }
.np-rule { height: 1px; background: #27394A; margin: 1.2rem 0; }
.np-note { background: #172B3A; border: 1px solid #2C4556;
    border-radius: 10px; padding: .8rem 1rem; color: #C4D4DE; font-size: .89rem; }
.np-kicker { color: #92A9BA; font-size: .78rem; letter-spacing: .035em; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_cached(dataset_bytes, event_bytes, fitting_bytes, unit):
    return read_sources(dataset_bytes, event_bytes, fitting_bytes, unit)


@st.cache_data(show_spinner=False, max_entries=4)
def fit_cached(dwell_ms, delta_i, max_k):
    return select_2d_gmm_components_bic(
        dwell_ms, delta_i, max_components=max_k
    )


@st.cache_data(show_spinner=False, max_entries=8)
def topology_cached(fitting_bytes, n_events, threshold):
    from splitter_core import load_npz_bytes
    fitting = load_npz_bytes(fitting_bytes)
    return reclassify_segment_counts_by_similarity(
        fitting, n_events, threshold
    )


@st.cache_data(show_spinner=False, max_entries=4)
def metrics_cached(fitting_bytes, n_events):
    from splitter_core import load_npz_bytes
    return derive_event_metrics(load_npz_bytes(fitting_bytes), n_events)


def fmt(value, digits=3):
    return f"{value:,.{digits}f}" if np.isfinite(value) else "—"


def style_figure(fig, ax):
    fig.patch.set_facecolor("#101D2B")
    ax.set_facecolor("#101D2B")
    for spine in ax.spines.values():
        spine.set_color("#3D5062")
    ax.tick_params(colors="#B9C9D5", labelsize=9)
    ax.xaxis.label.set_color("#CEDAE3")
    ax.yaxis.label.set_color("#CEDAE3")
    ax.title.set_color("#F0F5F8")
    ax.grid(False)
    return fig, ax


def new_figure(figsize=(8.8, 5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    return style_figure(fig, ax)


def render_figure(fig, filename=None):
    st.pyplot(fig, clear_figure=False, use_container_width=True)
    if filename:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        st.download_button("Download figure · PNG", buf.getvalue(),
                           file_name=filename, mime="image/png")
    plt.close(fig)


def colour(index):
    return COLORS[index % len(COLORS)]


def raw_density(dwell, delta, log_x=False, smooth=1.0):
    """Smooth KDE is visual-only; bounded sample and grid protect Cloud memory."""
    valid = np.isfinite(dwell) & (dwell > 0) & np.isfinite(delta)
    x, y = dwell[valid], delta[valid]
    if len(x) < 5:
        return None
    if len(x) > 3500:
        rng = np.random.default_rng(0)
        chosen = rng.choice(len(x), 3500, replace=False)
        x, y = x[chosen], y[chosen]
    work = np.log10(x) if log_x else x
    if np.ptp(work) <= 0 or np.ptp(y) <= 0:
        return None
    lo_x, hi_x = np.percentile(work, [0.2, 99.8])
    lo_y, hi_y = np.percentile(y, [0.2, 99.8])
    keep = (work >= lo_x) & (work <= hi_x) & (y >= lo_y) & (y <= hi_y)
    work, y = work[keep], y[keep]
    if len(work) < 5:
        return None
    try:
        kde = gaussian_kde(np.vstack([work, y]), bw_method="scott")
        kde.set_bandwidth(kde.factor * smooth)
        gx = np.linspace(lo_x, hi_x, 150)
        gy = np.linspace(lo_y, hi_y, 150)
        xx, yy = np.meshgrid(gx, gy)
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    except (ValueError, np.linalg.LinAlgError):
        return None
    return (10**gx if log_x else gx), gy, zz, len(work)


def plot_density(dwell, delta, log_x=False, smooth=1.0):
    density = raw_density(dwell, delta, log_x, smooth)
    if density is None:
        st.info("A smooth KDE could not be calculated for these values.")
        return
    gx, gy, zz, n_plot = density
    fig, ax = new_figure()
    mesh = ax.pcolormesh(gx, gy, zz, shading="auto", cmap="magma")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Dwell time (ms)")
    ax.set_ylabel("ΔI (nA)")
    ax.set_title("Event density")
    cbar = fig.colorbar(mesh, ax=ax, pad=.025)
    cbar.set_label("KDE density", color="#CEDAE3")
    cbar.ax.tick_params(colors="#B9C9D5")
    fig.tight_layout()
    render_figure(fig, "event_density.png")
    st.caption(f"KDE fitted to {n_plot:,} events. A reproducible sample is used "
               "above 3,500 events. Smoothing and display limits do not alter clustering.")


def plot_bic(table, best_k, selected_k):
    fig, ax = new_figure((8.5, 4.1))
    x = table["Components (K)"].to_numpy()
    y = table["BIC"].to_numpy()
    ax.plot(x, y, color="#75A9B8", lw=2, marker="o", ms=5)
    ax.scatter([best_k], [float(np.min(y))], s=105, color="#3CB8AE",
               zorder=5, label=f"Minimum BIC · K={best_k}")
    if selected_k != best_k:
        val = float(table.loc[table["Components (K)"] == selected_k, "BIC"].iloc[0])
        ax.scatter([selected_k], [val], s=100, marker="D",
                   color="#E9A55D", zorder=6, label=f"Selected · K={selected_k}")
    ax.set_xticks(x)
    ax.set_xlabel("Number of Gaussian components (K)")
    ax.set_ylabel("BIC")
    ax.set_title("Model selection")
    ax.legend(frameon=False, labelcolor="#D3DEE7", fontsize=9)
    fig.tight_layout()
    render_figure(fig, "bic_model_selection.png")


def plot_clusters(data, result, log_x=False, show_boundary=True):
    idx = result["valid_idx"]
    dwell = data["dwell_ms"]
    delta = data["delta_i"]
    labels = result["labels"]
    fig, ax = new_figure((9.0, 5.7))
    # Plot-only sampling, identical assignments and model are retained.
    if len(idx) > 20000:
        idx = np.sort(np.random.default_rng(0).choice(idx, 20000, replace=False))
    if show_boundary and result["n_components"] > 1:
        scaler, model = result["scaler"], result["model"]
        log_d = np.log10(dwell[result["valid_idx"]])
        amp = delta[result["valid_idx"]]
        gx = np.linspace(np.percentile(log_d, .2), np.percentile(log_d, 99.8), 170)
        gy = np.linspace(np.percentile(amp, .2), np.percentile(amp, 99.8), 145)
        xx, yy = np.meshgrid(gx, gy)
        features = np.column_stack([xx.ravel(), yy.ravel()])
        pred = model.predict(scaler.transform(features))
        ordered = result["raw_to_ordered"][pred].reshape(xx.shape)
        ax.contour(10**xx, yy, ordered, levels=np.arange(result["n_components"]-1)+.5,
                   colors="#8495A5", linewidths=.8, alpha=.55)
    for k, indices in enumerate(result["cluster_indices"]):
        selected = idx[labels[idx] == k]
        ax.scatter(dwell[selected], delta[selected], s=8, alpha=.47,
                   color=colour(k), edgecolors="none",
                   label=f"C{k+1} · {len(indices):,}")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Dwell time (ms)")
    ax.set_ylabel("ΔI (nA)")
    ax.set_title("Selected-K population map")
    ax.legend(frameon=False, labelcolor="#D3DEE7", fontsize=8,
              loc="best", ncol=2, markerscale=2)
    fig.tight_layout()
    render_figure(fig, "gmm_population_map.png")


def plot_histogram(data, result, metric, mode, bins, selected_clusters):
    values = data["dwell_ms"] if metric == "Dwell time" else data["delta_i"]
    arrays = [values[result["cluster_indices"][k]] for k in selected_clusters]
    arrays = [a[np.isfinite(a)] for a in arrays]
    nonempty = [a for a in arrays if len(a)]
    if not nonempty:
        st.info("No events are available for this histogram.")
        return
    all_values = values[result["valid_idx"]]
    all_values = all_values[np.isfinite(all_values)]
    low, high = np.min(all_values), np.max(all_values)
    if low == high:
        high = low + 1e-9
    edges = np.linspace(low, high, bins + 1)
    fig, ax = new_figure()
    for k, arr in zip(selected_clusters, arrays):
        if not len(arr):
            continue
        weights = None
        density = mode == "Density · each cluster"
        if mode == "Share of all events":
            weights = np.full(len(arr), 1.0 / len(result["valid_idx"]))
        ax.hist(arr, bins=edges, weights=weights, density=density,
                histtype="step", linewidth=1.7, color=colour(k),
                label=f"C{k+1} · {len(arr):,}")
    ax.set_xlabel("Dwell time (ms)" if metric == "Dwell time" else "ΔI (nA)")
    ax.set_ylabel({
        "Count": "Number of events",
        "Density · each cluster": "Probability density",
        "Share of all events": "Fraction of all valid events / bin",
    }[mode])
    ax.set_title(f"{metric} distributions")
    ax.legend(frameon=False, labelcolor="#D3DEE7", fontsize=9)
    fig.tight_layout()
    render_figure(fig, "cluster_distributions.png")


def display_cluster_table(table):
    st.dataframe(
        table.style.format({
            "% of valid events": "{:.1f}",
            "Median dwell (ms)": "{:.4f}",
            "Mean dwell (ms)": "{:.4f}",
            "Mean ΔI (nA)": "{:.4f}",
        }),
        hide_index=True, use_container_width=True
    )


def display_topology_table(table):
    st.dataframe(table.style.format({
        "Mean original segments": "{:.3f}",
        "Mean reclassified segments": "{:.3f}",
        "% linear": "{:.1f}", "% folded": "{:.1f}", "% complex": "{:.1f}",
    }), hide_index=True, use_container_width=True)


def plot_topology(table):
    fig, ax = new_figure((9, 4.8))
    x = np.arange(len(table))
    width = .25
    labels = ["Linear-like", "Folded-like", "Complex"]
    columns = ["% linear", "% folded", "% complex"]
    colours = ["#3CB8AE", "#E9A55D", "#8B8BEA"]
    for i, (name, column, c) in enumerate(zip(labels, columns, colours)):
        vals = table[column].fillna(0).to_numpy()
        ax.bar(x + (i-1)*width, vals, width=.24, color=c, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(table["Cluster"])
    ax.set_ylabel("Events with usable topology (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Segment-derived event classes")
    ax.legend(frameon=False, labelcolor="#D3DEE7", ncol=3, fontsize=9)
    fig.tight_layout()
    render_figure(fig, "topology_by_cluster.png")


def safe_stem(filename):
    stem = clean_stem(filename)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "nanopore"


# ─────────────────────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="np-eyebrow">Research analysis workspace</div>',
            unsafe_allow_html=True)
st.title("Nanopore Studio")
st.markdown(
    '<div class="np-subtitle">Explore event distributions, select Gaussian '
    'mixture models with BIC, and characterise DNA transport populations '
    'without changing the original measurements.</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("### Data sources")
    st.caption("Only the dataset is required for clustering.")
    f_dataset = st.file_uploader("Dataset · required", type=["npz"], key="np_dataset")
    with st.expander("Optional companion files", expanded=False):
        f_event = st.file_uploader("Event data · optional", type=["npz"], key="np_event")
        f_fitting = st.file_uploader("Event fitting · optional", type=["npz"], key="np_fitting")
        st.caption("Event data enables original-ID mapping and synchronized exports. "
                   "Event fitting enables segment and topology analysis.")
    unit = st.selectbox("Stored dwell-time unit", ["s", "ms", "us"], index=0,
                        format_func=lambda u: {"s":"Seconds (NanoSense default)",
                                               "ms":"Milliseconds", "us":"Microseconds"}[u])
    st.divider()
    st.caption(f"Nanopore Studio · v{VERSION}")
    st.caption("Files are processed by the Streamlit server hosting this app. Source files are not modified.")

if f_dataset is None:
    st.markdown('<div class="np-rule"></div>', unsafe_allow_html=True)
    st.info("Upload a NanoSense dataset.npz to begin. The other two files are optional.")
    st.markdown("**The streamlined workflow**")
    st.write("Inspect your data → compare candidate K with BIC → select and "
             "characterise clusters → optionally reclassify segment topology → export.")
    st.stop()

dataset_bytes = f_dataset.getvalue()
event_bytes = f_event.getvalue() if f_event else None
fitting_bytes = f_fitting.getvalue() if f_fitting else None
source_id = hashlib.sha256(
    dataset_bytes + unit.encode() + (event_bytes or b"") + (fitting_bytes or b"")
).hexdigest()

try:
    with st.spinner("Reading and validating the dataset…"):
        data = load_cached(dataset_bytes, event_bytes, fitting_bytes, unit)
except Exception as exc:
    st.error(f"Unable to load these sources: {exc}")
    st.stop()

n_total = len(data["dwell_ms"])
n_valid = int(data["valid"].sum())
n_excluded = n_total - n_valid

if st.session_state.get("np_source_id") != source_id:
    st.session_state["np_source_id"] = source_id
    st.session_state.pop("np_fit", None)
    st.session_state.pop("np_export", None)
    st.session_state.pop("np_selected_k", None)

# All controls are separate from source data and statistical model parameters.
st.markdown('<div class="np-rule"></div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Detected events", f"{n_total:,}")
m2.metric("Valid for GMM", f"{n_valid:,}")
m3.metric("Excluded from 2D", f"{n_excluded:,}")
m4.metric("Companion files", f"{int(event_bytes is not None)+int(fitting_bytes is not None)} / 2")

if n_excluded:
    st.warning(f"{n_excluded:,} events have non-finite ΔI, invalid dwell time, or "
               "non-positive dwell. They remain in the source data and will be "
               "recorded in the export; they are not assigned to a GMM cluster.")

tabs = st.tabs(["01  Data", "02  Model selection", "03  Populations",
                "04  Topology", "05  Export"])

# ── DATA ──────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Dataset overview")
    st.caption("Inspect the original measurements before fitting a model.")
    with st.expander("Source validation and column mapping"):
        for note in data["notes"]:
            st.write("✓", note)
        st.write(f"X shape: {data['X'].shape}; dwell input unit: {unit}.")
        st.write("The default feature mapping is X[:,4] for dwell and X[:,0] for ΔI.")
        st.caption("Only load NPZ files from trusted sources because NanoSense "
                   "archives may contain Python object arrays.")
    a, b = st.columns([1.1, 2.2], gap="large")
    with a:
        st.markdown("#### Raw distributions")
        metric = st.selectbox("Histogram variable", ["Dwell time", "ΔI"],
                              key="raw_metric")
        values = data["dwell_ms"] if metric == "Dwell time" else data["delta_i"]
        values = values[data["valid"]]
        fig, ax = new_figure((5.8, 4.1))
        ax.hist(values, bins=70, color="#3CB8AE", alpha=.85, edgecolor="none")
        ax.set_xlabel("Dwell time (ms)" if metric == "Dwell time" else "ΔI (nA)")
        ax.set_ylabel("Number of events")
        ax.set_title("All valid events")
        fig.tight_layout()
        render_figure(fig)
        st.caption(f"Median dwell: {fmt(np.median(data['dwell_ms'][data['valid']]),4)} ms")
    with b:
        st.markdown("#### Dwell–blockade landscape")
        c1, c2 = st.columns(2)
        with c1:
            density_log = st.toggle("Logarithmic dwell axis", value=False)
        with c2:
            density_smooth = st.slider("KDE smoothing", .5, 2.0, 1.0, .1)
        plot_density(data["dwell_ms"], data["delta_i"], density_log, density_smooth)

# ── MODEL SELECTION ───────────────────────────────────────────
with tabs[1]:
    st.subheader("BIC-guided model selection")
    st.caption("Fit independent full-covariance GMMs to standardized "
               "[log₁₀(dwell ms), ΔI]. No forced two-component analysis.")
    with st.form("fit_form"):
        c1, c2 = st.columns([1, 2])
        with c1:
            max_k = st.slider("Maximum K to test", 2, 10, 6)
        with c2:
            st.markdown("**Fixed fitting settings**")
            st.caption("Full covariance · 10 initializations · random state 0 · "
                       "regularization 10⁻⁶. The same settings are used for every K.")
        run_fit = st.form_submit_button("Fit and compare models", type="primary",
                                        disabled=n_valid < 20)
    if run_fit:
        try:
            with st.spinner("Fitting candidate Gaussian mixtures…"):
                fitted = fit_cached(data["dwell_ms"], data["delta_i"], max_k)
            st.session_state["np_fit"] = {"source": source_id, "fit": fitted}
            st.session_state.pop("np_selected_k", None)
            st.session_state.pop("np_export", None)
        except Exception as exc:
            st.error(f"Model fitting failed: {exc}")

    fit_state = st.session_state.get("np_fit")
    if fit_state and fit_state["source"] == source_id:
        bic = fit_state["fit"]
        best_k = int(bic["best_k"])
        tested = [int(k) for k in bic["table"]["Components (K)"]]
        if st.session_state.get("np_selected_k") not in tested:
            st.session_state["np_selected_k"] = best_k
        selected_k = st.selectbox(
            "K to use for population analysis", tested,
            format_func=lambda k: f"K = {k}" + (" · BIC minimum" if k == best_k else ""),
            key="np_selected_k"
        )
        selected_k = int(selected_k)
        result = build_bic_selected_population_result(
            bic, data["dwell_ms"], data["delta_i"], selected_k=selected_k
        )
        selected_bic = float(bic["table"].loc[
            bic["table"]["Components (K)"] == selected_k, "BIC"].iloc[0])
        delta_bic = selected_bic - float(bic["table"]["BIC"].min())
        x1, x2, x3 = st.columns(3)
        x1.metric("BIC-preferred K", str(best_k))
        x2.metric("Selected K", str(selected_k))
        x3.metric("ΔBIC from minimum", f"{delta_bic:.1f}")
        if best_k == max(tested):
            st.warning("The lowest BIC occurs at the largest K tested. "
                       "The exact preferred component count is not established; "
                       "test a wider range if scientifically appropriate.")
        if selected_k != best_k:
            st.info("You selected a different K from the BIC minimum. "
                    "Both values will be recorded in the export.")
        bic_table = bic["table"].copy()
        bic_table["Converged"] = [bic["models"][k].converged_ for k in tested]
        bic_table["EM iterations"] = [bic["models"][k].n_iter_ for k in tested]
        left, right = st.columns([1.1, 1], gap="large")
        with left:
            plot_bic(bic_table, best_k, selected_k)
        with right:
            st.markdown("#### Candidate models")
            st.dataframe(bic_table.style.format(
                {"BIC": "{:.1f}", "ΔBIC from best": "{:.1f}"}
            ), hide_index=True, use_container_width=True)
            if not bic_table["Converged"].all():
                st.warning("One or more fits did not converge. Treat the affected "
                           "BIC values cautiously and consider reviewing the fit.")
        with st.expander("How to interpret BIC"):
            st.write("BIC balances model likelihood against the number of fitted "
                     "parameters. Lower BIC is preferred among the candidate models.")
            st.latex(r"\mathrm{BIC}=-2\ln L+p\ln N")
            st.caption("A Gaussian component is a statistical description, not "
                       "automatically a separate DNA transport mechanism.")
    else:
        st.info("Run the BIC comparison to unlock population analysis and export.")

# Shared selected result — no additional GMM refitting on a K selection.
fit_state = st.session_state.get("np_fit")
has_fit = bool(fit_state and fit_state["source"] == source_id)
if has_fit:
    bic = fit_state["fit"]
    chosen_k = int(st.session_state.get("np_selected_k", bic["best_k"]))
    result = build_bic_selected_population_result(
        bic, data["dwell_ms"], data["delta_i"], selected_k=chosen_k
    )
    cluster_table = descriptive_table(result, data["dwell_ms"], data["delta_i"])
else:
    bic = result = cluster_table = None

# ── POPULATIONS ───────────────────────────────────────────────
with tabs[2]:
    st.subheader("Population characterisation")
    if not has_fit:
        st.info("Fit the candidate models in Model selection first.")
    else:
        st.caption("Clusters are ordered by median original dwell time. Means are "
                   "calculated from hard-assigned raw events, not GMM centres.")
        display_cluster_table(cluster_table)
        st.download_button("Download cluster summary · CSV",
                           cluster_table.to_csv(index=False),
                           "cluster_summary.csv", "text/csv")
        p = result["assignment_probability"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Median assignment confidence", f"{100*np.median(p):.1f}%")
        c2.metric("Assignments below 70%", f"{100*np.mean(p < .70):.1f}%")
        c3.metric("Number of populations", str(result["n_components"]))
        st.markdown("#### Population map")
        c1, c2 = st.columns(2)
        with c1:
            plot_log = st.toggle("Logarithmic dwell axis", key="cluster_log")
        with c2:
            boundary = st.toggle("Show model boundaries", value=True)
        plot_clusters(data, result, plot_log, boundary)
        st.markdown("#### Compare distributions")
        h1, h2, h3 = st.columns(3)
        with h1:
            hist_metric = st.selectbox("Variable", ["Dwell time", "ΔI"], key="cluster_metric")
        with h2:
            hist_mode = st.selectbox("Y-axis", ["Count", "Density · each cluster",
                                                "Share of all events"])
        with h3:
            n_bins = st.slider("Common bin count", 20, 150, 70, 5)
        selected_clusters = st.multiselect(
            "Populations to display",
            options=list(range(result["n_components"])),
            default=list(range(result["n_components"])),
            format_func=lambda k: f"Cluster {k+1}",
        )
        if selected_clusters:
            plot_histogram(data, result, hist_metric, hist_mode, n_bins, selected_clusters)
        st.caption("All curves use the same bin edges. Density normalizes each "
                   "population separately; count and share preserve abundance.")

# ── TOPOLOGY ──────────────────────────────────────────────────
topology_table = None
effective_counts = None
topology_threshold = None
with tabs[3]:
    st.subheader("Optional segment-topology reclassification")
    if not has_fit:
        st.info("Fit a GMM before characterising its clusters.")
    elif fitting_bytes is None:
        st.info("Upload the optional event_fitting.npz in the sidebar to enable "
                "segment and topology analysis. Clustering and other exports remain available.")
    else:
        st.caption("Post-clustering sensitivity analysis. The GMM assignments stay "
                   "fixed while the similarity threshold changes.")
        c1, c2 = st.columns([1, 2])
        with c1:
            enable_topology = st.toggle("Enable reclassification", value=False)
        with c2:
            st.caption("Adjacent fitted segment amplitudes are merged when their "
                       "similarity reaches the selected threshold.")
        if enable_topology:
            topology_threshold = st.slider(
                "Adjacent-segment amplitude similarity threshold (%)",
                0.0, 100.0, 80.0, 1.0, format="%.0f%%"
            )
            st.latex(r"S(a,b)=100\frac{\min(|a|,|b|)}{\max(|a|,|b|)}")
            st.caption("100% means identical blockade magnitudes. Merged levels are "
                       "duration-weighted when widths are available. This reproduces "
                       "the exploratory v1.4.8 rule; it is not a NanoSense-native "
                       "threshold unless independently verified.")
            with st.spinner("Reclassifying segment sequences…"):
                derived = metrics_cached(fitting_bytes, n_total)
                original_counts = derived["Number of segments"]
                effective_counts = topology_cached(
                    fitting_bytes, n_total, topology_threshold
                )
                topology_table = topology_statistics(
                    result, original_counts, effective_counts
                )
            display_topology_table(topology_table)
            st.download_button("Download topology summary · CSV",
                               topology_table.to_csv(index=False),
                               "topology_summary.csv", "text/csv")
            plot_topology(topology_table)
            with st.expander("Definitions and limitations"):
                st.write("One effective segment is linear-like; two are folded-like; "
                         "three or more are complex. These are segment-derived, "
                         "putative classes rather than confirmed molecular conformations.")
                st.write("Percentages use only events with usable segment information. "
                         "Missing values are reported separately. Longer events may "
                         "offer more opportunity for a segmentation algorithm to find "
                         "multiple levels, so this is supporting evidence, not "
                         "independent proof of a physical mechanism.")
                st.write("The similarity rule is applied after the original NanoSense "
                         "segmentation. It does not refit the current trace or modify "
                         "the original NPZ files.")

# ── EXPORT ────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Export analysis and synchronized populations")
    if not has_fit:
        st.info("Run a GMM analysis to enable export.")
    else:
        st.caption("All original rows are traceable. Invalid 2D events are logged, "
                   "and only uploaded source types are included in each cluster folder.")
        st.markdown("#### Included in the analysis package")
        st.write("Cluster summary, full event-to-cluster mapping, posterior "
                 "probabilities, BIC table, model settings, and excluded-event log.")
        if topology_table is not None:
            st.write("The current topology reclassification table and threshold "
                     "will also be included.")
        sources = ["dataset"]
        if event_bytes is not None:
            sources.append("event_data")
        if fitting_bytes is not None:
            sources.append("event_fitting")
        st.markdown("**Available NPZ source types:** " + ", ".join(sources))
        stem = safe_stem(f_dataset.name)
        if st.button("Build analysis package", type="primary"):
            with st.spinner("Building synchronized cluster files…"):
                try:
                    zip_bytes = export_bundle(
                        data, bic, result, topology_table, effective_counts,
                        topology_threshold, stem
                    )
                    st.session_state["np_export"] = {
                        "source": source_id,
                        "K": chosen_k,
                        "threshold": topology_threshold,
                        "bytes": zip_bytes
                    }
                except Exception as exc:
                    st.error(f"Export failed: {exc}")
        export_state = st.session_state.get("np_export")
        if export_state and export_state["source"] == source_id \
                and export_state["K"] == chosen_k \
                and export_state["threshold"] == topology_threshold:
            st.success("Analysis package is ready.")
            st.download_button(
                "Download analysis + clusters · ZIP",
                export_state["bytes"],
                file_name=f"{stem}_GMM_K{chosen_k}_analysis.zip",
                mime="application/zip", type="primary"
            )
        elif export_state and export_state["source"] == source_id:
            st.caption("The selection has changed. Build the package again to "
                       "include the current K and topology settings.")
        with st.expander("Export compatibility"):
            st.write("Dataset-only exports contain filtered dataset.npz files and "
                     "original-row mapping. If event_data and/or event_fitting were "
                     "uploaded, their corresponding synchronized, reindexed NPZ files "
                     "are included too. A complete three-file export retains the "
                     "original NanoSense source structure.")
            st.write("The source archives are never overwritten. Original IDs and "
                     "row indices are retained in event_id_mapping.csv.")
