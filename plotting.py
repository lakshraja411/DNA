"""White, publication-style plots with editable labels and export."""
from __future__ import annotations
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, NullLocator
from scipy.stats import gaussian_kde

PALETTE = ["#3B6EA8", "#D4813B", "#578F6D", "#9B6AA6",
           "#C05C64", "#4F9B9B", "#A18B41", "#6E7DAD",
           "#B07959", "#708B55"]

def colour(k):
    return PALETTE[k % len(PALETTE)]

def figure(spec):
    font = float(spec.get("font_size", 11))
    fig, ax = plt.subplots(figsize=(spec.get("width", 7.6),
                                    spec.get("height", 5.0)), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#303030")
        ax.spines[side].set_linewidth(.9)
    ax.tick_params(which="major", direction="out", length=4, width=.8,
                   colors="#292929", labelsize=font-1, top=False, right=False)
    ax.tick_params(which="minor", direction="out", length=2, width=.6,
                   colors="#292929", top=False, right=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if spec.get("grid", False):
        ax.grid(axis="y", color="#E4E7EA", linewidth=.6, zorder=0)
    return fig, ax

def finish(fig, ax, spec, title="", xlabel="", ylabel="", legend=True):
    font = float(spec.get("font_size", 11))
    ax.set_title(spec.get("title", title), fontsize=font+2, pad=13,
                 fontweight="normal", color="#202020")
    ax.set_xlabel(spec.get("xlabel", xlabel), fontsize=font, labelpad=8)
    ax.set_ylabel(spec.get("ylabel", ylabel), fontsize=font, labelpad=8)
    for axis, name in ((ax.xaxis, "x"), (ax.yaxis, "y")):
        low = spec.get(f"{name}min")
        high = spec.get(f"{name}max")
        if low is not None or high is not None:
            current = axis.get_view_interval()
            if low is None:
                low = current[0]
            if high is None:
                high = current[1]
            if float(low) < float(high):
                (ax.set_xlim if name == "x" else ax.set_ylim)(float(low), float(high))
    handles, labels = ax.get_legend_handles_labels()
    if legend and spec.get("legend", True) and handles:
        ax.legend(frameon=False, fontsize=font-1, loc="best",
                  handlelength=1.8, borderaxespad=.5)
    fig.tight_layout(pad=1.4)
    return fig

def export_bytes(fig, format="png"):
    buf = io.BytesIO()
    kw = {"format": format, "bbox_inches": "tight", "facecolor": "white"}
    if format == "png":
        kw["dpi"] = 600
    fig.savefig(buf, **kw)
    return buf.getvalue()

def save_figure(fig, path):
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")

def bic_plot(table, best_k, selected_k, spec):
    fig, ax = figure(spec)
    x = table["Components (K)"].to_numpy()
    y = table["BIC"].to_numpy()
    ax.plot(x, y, "-o", color=colour(0), linewidth=1.5, markersize=5,
            markerfacecolor="white", markeredgewidth=1.3, label="BIC")
    ax.scatter([best_k], [np.min(y)], color=colour(2), marker="o", s=60,
               zorder=5, label=f"Minimum · K = {best_k}")
    if selected_k != best_k:
        chosen = float(table.loc[table["Components (K)"] == selected_k, "BIC"].iloc[0])
        ax.scatter([selected_k], [chosen], color=colour(1), marker="D", s=50,
                   zorder=6, label=f"Selected · K = {selected_k}")
    ax.set_xticks(x)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    return finish(fig, ax, spec, "BIC model selection",
                  "Number of Gaussian components, K", "BIC")

def scatter_plot(data, result, delta_i, spec, log_x=True, boundaries=True,
                 point_size=8, alpha=.50, maximum=25000):
    dwell = np.asarray(data["dwell_ms"])
    delta_i = np.asarray(delta_i)
    fig, ax = figure(spec)
    valid = result["valid_idx"]
    # Only the display may be sampled. The model and exported rows are complete.
    if len(valid) > maximum:
        valid = np.sort(np.random.default_rng(0).choice(valid, maximum, replace=False))
    labels = result["labels"]
    if boundaries and result["n_components"] > 1:
        all_idx = result["valid_idx"]
        log_d = np.log10(dwell[all_idx])
        amp = delta_i[all_idx]
        gx = np.linspace(*np.percentile(log_d, [.2, 99.8]), 180)
        gy = np.linspace(*np.percentile(amp, [.2, 99.8]), 150)
        xx, yy = np.meshgrid(gx, gy)
        features = np.column_stack((xx.ravel(), yy.ravel()))
        pred = result["model"].predict(result["scaler"].transform(features))
        ordered = result["raw_to_ordered"][pred].reshape(xx.shape)
        ax.contour(10**xx, yy, ordered,
                   levels=np.arange(result["n_components"]-1)+.5,
                   colors="#656565", linewidths=.75, linestyles="dashed", alpha=.7)
    for k, idx in enumerate(result["cluster_indices"]):
        selected = valid[labels[valid] == k]
        ax.scatter(dwell[selected], delta_i[selected], s=point_size,
                   alpha=alpha, color=colour(k), rasterized=True,
                   linewidths=0, label=f"Cluster {k+1} (n = {len(idx):,})")
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
    return finish(fig, ax, spec, "GMM population classification",
                  "Dwell time (ms)", data["delta_label"])

def histogram_plot(data, result, delta_i, metric, selection, mode,
                   edges, spec, log_x=False, style="Step"):
    values = (np.asarray(data["dwell_ms"]) if metric == "Dwell time"
              else np.asarray(delta_i))
    valid = result["valid_idx"]
    fig, ax = figure(spec)
    density = mode == "Density"
    for k in selection:
        idx = valid if k == -1 else result["cluster_indices"][k]
        a = values[idx]
        a = a[np.isfinite(a) & ((not log_x) | (a > 0))]
        if not len(a):
            continue
        if mode == "Share of all events":
            weights = np.full(len(a), 1. / len(valid))
        else:
            weights = None
        kw = {"bins": edges, "density": density, "weights": weights,
              "color": "#444444" if k == -1 else colour(k),
              "label": "All events" if k == -1 else f"Cluster {k+1} (n = {len(idx):,})"}
        if style == "Filled":
            ax.hist(a, histtype="bar", alpha=.4, edgecolor="none", **kw)
        else:
            ax.hist(a, histtype="step", linewidth=1.6, **kw)
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
    ylabel = {"Count":"Number of events", "Density":"Probability density",
              "Share of all events":"Fraction of valid events per bin"}[mode]
    xlabel = "Dwell time (ms)" if metric == "Dwell time" else data["delta_label"]
    return finish(fig, ax, spec, f"{metric} distribution", xlabel, ylabel)

def kde_plot(data, result, delta_i, metric, selection, spec, log_x=False,
             bandwidth=1., limit=5000):
    values = (np.asarray(data["dwell_ms"]) if metric == "Dwell time"
              else np.asarray(delta_i))
    fig, ax = figure(spec)
    indices = result["valid_idx"]
    for k in selection:
        idx = indices if k == -1 else result["cluster_indices"][k]
        a = values[idx]
        a = a[np.isfinite(a) & ((not log_x) | (a > 0))]
        if len(a) > limit:
            a = np.random.default_rng(0).choice(a, limit, replace=False)
        if len(a) < 3:
            continue
        work = np.log10(a) if log_x else a
        if np.std(work) <= 0:
            continue
        try:
            kde = gaussian_kde(work, bw_method="scott")
            kde.set_bandwidth(kde.factor * bandwidth)
            grid = np.linspace(*np.percentile(work, [.2, 99.8]), 500)
            y = kde(grid)
        except (ValueError, np.linalg.LinAlgError):
            continue
        # Change of variables: density in raw dwell units, even on a log axis.
        if log_x:
            x = 10**grid
            y = y / (x * np.log(10))
        else:
            x = grid
        ax.plot(x, y, linewidth=1.6, color="#444444" if k == -1 else colour(k),
                label="All events" if k == -1 else f"Cluster {k+1} (n = {len(idx):,})")
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
    xlabel = "Dwell time (ms)" if metric == "Dwell time" else data["delta_label"]
    return finish(fig, ax, spec, f"{metric} density", xlabel, "Probability density")

def density_plot(data, delta_i, spec, log_x=True, bandwidth=1., maximum=3500):
    dwell = np.asarray(data["dwell_ms"])
    delta = np.asarray(delta_i)
    valid = np.isfinite(dwell) & (dwell > 0) & np.isfinite(delta)
    x, y = dwell[valid], delta[valid]
    if len(x) > maximum:
        chosen = np.random.default_rng(0).choice(len(x), maximum, replace=False)
        x, y = x[chosen], y[chosen]
    if len(x) < 5:
        raise ValueError("At least five valid events are required for the density map.")
    work = np.log10(x) if log_x else x
    xlo, xhi = np.percentile(work, [.2, 99.8])
    ylo, yhi = np.percentile(y, [.2, 99.8])
    keep = (work >= xlo) & (work <= xhi) & (y >= ylo) & (y <= yhi)
    work, y = work[keep], y[keep]
    if np.std(work) <= 0 or np.std(y) <= 0:
        raise ValueError("The data do not have sufficient two-dimensional spread for KDE.")
    kde = gaussian_kde(np.vstack((work, y)), bw_method="scott")
    kde.set_bandwidth(kde.factor * bandwidth)
    gx = np.linspace(xlo, xhi, 180)
    gy = np.linspace(ylo, yhi, 170)
    xx, yy = np.meshgrid(gx, gy)
    z = kde(np.vstack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("white")
    fig, ax = figure(spec)
    mesh = ax.pcolormesh(10**gx if log_x else gx, gy, z,
                         shading="auto", cmap=cmap, rasterized=True)
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
    cbar = fig.colorbar(mesh, ax=ax, pad=.025, fraction=.045)
    cbar.set_label("KDE density", fontsize=spec.get("font_size",11)-1)
    cbar.ax.tick_params(labelsize=spec.get("font_size",11)-2, direction="out")
    return finish(fig, ax, spec, "Event density",
                  "Dwell time (ms)", data["delta_label"], legend=False)
