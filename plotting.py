"""White, publication-style plots with editable labels and export."""
from __future__ import annotations
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, NullLocator
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

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
    kw = {"format": format, "bbox_inches": "tight", "facecolor": fig.get_facecolor()}
    if format == "png":
        kw["dpi"] = 600
    fig.savefig(buf, **kw)
    return buf.getvalue()

def save_figure(fig, path):
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())

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

def _display_subset(data, result, selection, maximum=25000):
    """Return complete selected rows, optionally capped only for scatter rendering."""
    valid = np.asarray(result["valid_idx"], dtype=int)
    if selection is None:
        indices = valid
    else:
        indices = np.asarray(selection, dtype=int)
    if maximum is not None and len(indices) > maximum:
        indices = np.sort(np.random.default_rng(0).choice(
            indices, maximum, replace=False))
    return indices


def scatter_plot(data, result, delta_i, spec, log_x=False, boundaries=True,
                 point_size=13, alpha=.45, maximum=25000, unit_scale=1.0):
    """Original selected-K scatter and dashed maximum-posterior boundaries."""
    dwell = np.asarray(data["dwell_ms"], dtype=float)
    delta = np.asarray(delta_i, dtype=float)
    fig, ax = figure(spec)
    valid = _display_subset(data, result, None, maximum)
    if boundaries and result["n_components"] > 1:
        all_idx = result["valid_idx"]
        d = np.log10(dwell[all_idx])
        a = delta[all_idx]
        gx = np.linspace(*np.percentile(d, [.2, 99.8]), 220)
        gy = np.linspace(*np.percentile(a, [.2, 99.8]), 220)
        xx, yy = np.meshgrid(gx, gy)
        features = np.column_stack((xx.ravel(), yy.ravel()))
        predicted = result["model"].predict(result["scaler"].transform(features))
        ordered = result["raw_to_ordered"][predicted].reshape(xx.shape)
        ax.contour(10**xx, yy*unit_scale, ordered,
                   levels=np.arange(result["n_components"]-1)+.5,
                   colors="#555555", linewidths=1.25, linestyles="--")
    for k, idx in enumerate(result["cluster_indices"]):
        selected = valid[result["labels"][valid] == k]
        ax.scatter(dwell[selected], delta[selected]*unit_scale,
                   s=point_size, alpha=alpha, color=colour(k),
                   linewidths=0, rasterized=True,
                   label=f"Cluster {k+1} ({len(idx):,})")
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
    return finish(fig, ax, spec, f"Selected-K 2D GMM classification (K = {result['n_components']})",
                  "Dwell time (ms)", data["delta_label"])


def histogram_plot(data, result, delta_i, metric, selection, mode,
                   edges, spec, log_x=False, style="Filled",
                   unit_scale=1.0):
    """Filled, overlapping histograms with common edges, as in the reference."""
    values = np.asarray(data["dwell_ms"] if metric == "Dwell time" else delta_i,
                        dtype=float)
    if metric != "Dwell time":
        values = values*unit_scale
    valid = result["valid_idx"]
    fig, ax = figure(spec)
    density = mode == "Density"
    for k in selection:
        idx = valid if k == -1 else result["cluster_indices"][k]
        a = values[idx]
        a = a[np.isfinite(a) & ((a > 0) if log_x else True)]
        if not len(a):
            continue
        weights = (np.full(len(a), 1.0/len(valid))
                   if mode == "Share of all events" else None)
        kw = dict(bins=edges, weights=weights, density=density,
                  color="#555555" if k == -1 else colour(k),
                  label="All events" if k == -1 else f"Cluster {k+1} ({len(idx):,})")
        if style == "Filled":
            ax.hist(a, histtype="bar", alpha=.55 if len(selection)>1 else .85,
                    edgecolor="none", **kw)
        else:
            ax.hist(a, histtype="step", linewidth=1.7, **kw)
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
    ylabel = {"Count":"Number of events", "Density":"Probability density",
              "Share of all events":"Fraction of valid events per bin"}[mode]
    xlabel = "Dwell time (ms)" if metric == "Dwell time" else data["delta_label"]
    return finish(fig, ax, spec, f"{metric} distribution", xlabel, ylabel)


def kde_plot(data, result, delta_i, metric, selection, spec, log_x=True,
             bandwidth=1.0, limit=10000, unit_scale=1.0):
    """Reference KDE: log-dwell fitting and log-dwell density when enabled."""
    values = np.asarray(data["dwell_ms"] if metric == "Dwell time" else delta_i,
                        dtype=float)
    if metric != "Dwell time":
        values = values*unit_scale
    fig, ax = figure(spec)
    used = []
    for k in selection:
        idx = result["valid_idx"] if k == -1 else result["cluster_indices"][k]
        a = values[idx]
        a = a[np.isfinite(a) & ((a > 0) if log_x else True)]
        if len(a) > limit:
            a = np.random.default_rng(0).choice(a, limit, replace=False)
        if len(a) < 3:
            continue
        work = np.log10(a) if log_x else a
        if np.std(work) <= 0:
            continue
        if bandwidth == 1.0:
            # Use the original Silverman KDE implementation unchanged.
            x, y = kde_curve(a, log_x)
        else:
            try:
                grid = np.linspace(np.min(work), np.max(work), 600)
                base = silverman_bandwidth(work)
                kde = KernelDensity(kernel="gaussian", bandwidth=base*bandwidth)
                kde.fit(work.reshape(-1, 1))
                x, y = (10**grid if log_x else grid), np.exp(
                    kde.score_samples(grid.reshape(-1, 1)))
            except (ValueError, np.linalg.LinAlgError):
                continue
        if x is None:
            continue
        ax.plot(x, y, linewidth=2, color="#555555" if k == -1 else colour(k),
                label="All events" if k == -1 else f"Cluster {k+1} ({len(idx):,})")
        used.append(len(a))
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
    ylabel = ("KDE density in log10(dwell time)" if log_x
              else "KDE density")
    xlabel = "Dwell time (ms)" if metric == "Dwell time" else data["delta_label"]
    return finish(fig, ax, spec, f"{metric} density", xlabel, ylabel)


def density_plot(data, delta_i, spec, log_x=False, bandwidth=.90,
                 maximum=5000, selected_indices=None, grid_size=220,
                 cutoff_percent=.15, y_percentiles=(0.,100.),
                 unit_scale=1.0, theme="Classic dark", return_details=False):
    """Reference 2D Gaussian KDE with peak-relative background masking."""
    dwell = np.asarray(data["dwell_ms"], dtype=float)
    delta = np.asarray(delta_i, dtype=float)*unit_scale
    valid = np.isfinite(dwell) & (dwell > 0) & np.isfinite(delta)
    if selected_indices is not None:
        mask = np.zeros(len(dwell), dtype=bool)
        mask[np.asarray(selected_indices, dtype=int)] = True
        valid &= mask
    x, y = dwell[valid], delta[valid]
    n_selected = len(x)
    if len(x) > maximum:
        chosen = np.random.default_rng(0).choice(len(x), maximum, replace=False)
        x, y = x[chosen], y[chosen]
    if len(x) < 5:
        raise ValueError("At least five valid events are required for the density map.")
    low_p, high_p = y_percentiles
    if low_p > 0 or high_p < 100:
        low, high = np.percentile(y, [low_p, high_p])
        keep = (y >= low) & (y <= high)
        x, y = x[keep], y[keep]
    gx, gy, z = true_2d_kde_density(
        x, y, log_x=log_x, grid_size=grid_size,
        bandwidth_scale=bandwidth, x_percentiles=(.2,99.8),
        y_percentiles=(.2,99.8))
    if z is None:
        raise ValueError("The selected data do not have sufficient spread for KDE.")
    if theme == "Classic dark":
        fig, ax = plt.subplots(figsize=(spec.get("width",7.6),
                                        spec.get("height",5)), dpi=120)
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        foreground = "white"
    else:
        fig, ax = figure(spec)
        foreground = "#202020"
    threshold = cutoff_percent/100.0*float(np.max(z))
    masked = np.ma.masked_where(z <= threshold, z)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("black" if theme == "Classic dark" else "white")
    mesh = ax.pcolormesh(gx, gy, masked, shading="auto",
                         cmap=cmap, rasterized=True)
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
    cbar = fig.colorbar(mesh, ax=ax, pad=.025, fraction=.045)
    cbar.set_label("KDE density", color=foreground,
                   fontsize=spec.get("font_size",11)-1)
    cbar.ax.tick_params(colors=foreground, direction="out",
                        labelsize=spec.get("font_size",11)-2)
    if theme == "Classic dark":
        font = spec.get("font_size",11)
        ax.set_title(spec.get("title", "Event density"), color="white",
                     fontsize=font+2, pad=13)
        ax.set_xlabel(spec.get("xlabel","Dwell time (ms)"),
                      color="white", fontsize=font)
        ax.set_ylabel(spec.get("ylabel",data["delta_label"]),
                      color="white", fontsize=font)
        ax.tick_params(colors="white", direction="out")
        for spine in ax.spines.values():
            spine.set_color("white")
        for axis, name in ((ax.xaxis,"x"),(ax.yaxis,"y")):
            lo, hi = spec.get(name+"min"), spec.get(name+"max")
            if lo is not None or hi is not None:
                current = axis.get_view_interval()
                lo = current[0] if lo is None else lo
                hi = current[1] if hi is None else hi
                if float(lo) < float(hi):
                    (ax.set_xlim if name=="x" else ax.set_ylim)(float(lo),float(hi))
        fig.tight_layout(pad=1.4)
    else:
        finish(fig, ax, spec, "Event density", "Dwell time (ms)",
               data["delta_label"], legend=False)
    details = {"selected_events":n_selected, "kde_events":len(x),
               "background_cutoff_percent":cutoff_percent}
    return (fig, details) if return_details else fig
