"""Pure plotting functions: every one takes already-computed DataFrames/arrays
(never a live GravPop model object) plus labels/colors/result_folder, and
returns/saves a matplotlib figure. This is what plot_subpop_results.py calls;
none of it should ever need to resample from a model -- see
src/plotting/ppd_compute.py for the stages that produce the inputs these
functions consume.
"""
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from matplotlib.ticker import LogFormatterMathtext, MaxNLocator
from scipy.ndimage import gaussian_filter1d

from plot_helper import add_contours, make_2D_comparison2, make_corner_plot

from .style import format_figure_ticks, fix_legend, APJ_HALF_COLUMN_WIDTH_IN, AASTEX_TEXT_WIDTH_IN, LATEX_FONT_SIZES_PT
from .relabeling import get_component_ppd_samples
from .stats import hpd_array

# -----------------------------------------------------------------------------
# Fraction (eta) vs mass_1
# -----------------------------------------------------------------------------


def plot_eta_small_multiples(m1, bands_frame, letters, labels, colors, figsize=None, sharey=True):
    """One panel per component: median eta_k(m) line + smoothed 90% band."""
    n_components = len(letters)
    if figsize is None:
        figsize = (8, 2.1 * n_components)

    fig, axes = plt.subplots(n_components, 1, figsize=figsize, sharex=True, sharey=sharey)
    if n_components == 1:
        axes = [axes]

    for k, (ax, letter, label, color) in enumerate(zip(axes, letters, labels, colors)):
        lo = gaussian_filter1d(bands_frame[f"{letter}_lower"].to_numpy(), sigma=1.0)
        med = bands_frame[f"{letter}_median"].to_numpy()
        hi = gaussian_filter1d(bands_frame[f"{letter}_upper"].to_numpy(), sigma=1.0)

        line = ax.plot(m1, med, lw=2.0, label=label, color=color)[0]
        ax.fill_between(m1, lo, hi, alpha=0.25, color=line.get_color(), linewidth=0)
        ax.set_ylabel(r"$\eta_k$")
        ax.set_ylim(0, 1)
        ax.text(0.02, 0.88, label, transform=ax.transAxes, ha="left", va="top")

    axes[-1].set_xlabel(r"$m_1$")
    axes[-1].set_xlim(np.min(m1), np.max(m1))
    fig.tight_layout()
    return fig, axes


def eta_vs_m1_plot(bands_frame, letters, result_folder, labels, colors, output_name, figwidth=None, dpi=200):
    m1 = bands_frame["mass_1_source"].to_numpy()
    fig, axes = plot_eta_small_multiples(m1, bands_frame, letters, labels, colors)
    if figwidth is not None:
        fig.set_figwidth(figwidth)
    fig.set_dpi(dpi)
    format_figure_ticks(fig)
    fig.savefig(f"{result_folder}/{output_name}", dpi=dpi, bbox_inches="tight")
    return fig


def original_label_fraction_spin_magnitude_plot(
    bands_frame_original_labels,
    raw_component_ppds,
    output_folder,
    letters,
    labels,
    colors,
):
    """Pair original-label fraction curves with their spin-magnitude PPDs."""
    mass_grid = bands_frame_original_labels["mass_1_source"].to_numpy()
    fig, axes = plt.subplots(
        len(letters), 2, figsize=(AASTEX_TEXT_WIDTH_IN, 8.0),
        gridspec_kw={"width_ratios": [1.35, 1.0]}, squeeze=False,
    )

    for component_index, (letter, label, color, component_ppd) in enumerate(
        zip(letters, labels, colors, raw_component_ppds)
    ):
        fraction_ax, spin_ax = axes[component_index]
        lower = gaussian_filter1d(bands_frame_original_labels[f"{letter}_lower"].to_numpy(), sigma=1.0)
        median = bands_frame_original_labels[f"{letter}_median"].to_numpy()
        upper = gaussian_filter1d(bands_frame_original_labels[f"{letter}_upper"].to_numpy(), sigma=1.0)
        fraction_ax.fill_between(mass_grid, lower, upper, color=color, alpha=0.25, linewidth=0)
        fraction_ax.plot(mass_grid, median, color=color, linewidth=2.0)
        fraction_ax.text(0.025, 0.9, label, transform=fraction_ax.transAxes, ha="left", va="top")
        fraction_ax.set_xlim(float(np.min(mass_grid)), float(np.max(mass_grid)))
        fraction_ax.set_ylim(0.0, 1.0)
        fraction_ax.set_ylabel(r"$\eta_k$")

        sampled = component_ppd.sample(10_000, replace=True, random_state=10)
        try:
            add_contours(
                spin_ax, sampled["chi_1"].to_numpy(), sampled["chi_2"].to_numpy(),
                a=np.array([0.0, 0.0]), b=np.array([1.0, 1.0]), color=color,
                quantiles=[0.9, 0.5, 0.1], boundary_method="truncnorm", fill=True,
            )
        except ValueError as error:
            if "Contour levels must be increasing" not in str(error):
                raise
            spin_ax.text(0.5, 0.5, "Degenerate KDE", transform=spin_ax.transAxes, ha="center", va="center")
        spin_ax.set_xlim(0.0, 1.0)
        spin_ax.set_ylim(0.0, 1.0)
        spin_ax.set_ylabel(r"$\chi_2$")
        spin_ax.text(0.025, 0.9, label, transform=spin_ax.transAxes, ha="left", va="top")

        if component_index != len(letters) - 1:
            fraction_ax.tick_params(axis="x", labelbottom=False)
            spin_ax.tick_params(axis="x", labelbottom=False)

    axes[-1, 0].set_xlabel(r"$m_1\,[M_\odot]$")
    axes[-1, 1].set_xlabel(r"$\chi_1$")
    fig.align_ylabels(axes)
    fig.tight_layout()
    format_figure_ticks(fig)
    folder_name = Path(output_folder).name
    seed_match = re.search(r"seed[_-]?(\d+)", folder_name, flags=re.IGNORECASE) or re.search(r"-(\d+)$", folder_name)
    seed_suffix = f"-seed{seed_match.group(1)}" if seed_match else ""
    output_stem = Path(output_folder) / f"eta-spin-mag-PPD-original-labels{seed_suffix}"
    fig.savefig(output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    return fig


# -----------------------------------------------------------------------------
# chi_eff / chi_p
# -----------------------------------------------------------------------------


def component_chi_eff_chi_p_ppd_plot(component_ppds, result_folder, labels, colors, scatter, output_name):
    sampled = [sample.sample(10_000, replace=True, random_state=10) for sample in component_ppds]
    if scatter:
        fig = make_2D_comparison2(
            sampled, variables=["chi_eff", "chi_p"],
            variable_labels=[r"$\chi_{\rm eff}$", r"$\chi_p$"], a=[-1, 0], b=[1, 1],
            model_labels=labels, colors=colors, scatter=True, marginals=False,
            legend_location="upper left", boundary_method="truncnorm",
            figsize=(APJ_HALF_COLUMN_WIDTH_IN, APJ_HALF_COLUMN_WIDTH_IN), dpi=400,
        )
    else:
        fig, ax = plt.subplots(figsize=(APJ_HALF_COLUMN_WIDTH_IN, APJ_HALF_COLUMN_WIDTH_IN), dpi=400)
        contour_labels = []
        for sample, label, color in zip(sampled, labels, colors):
            try:
                add_contours(
                    ax, sample["chi_eff"].to_numpy(), sample["chi_p"].to_numpy(),
                    a=np.array([-1.0, 0.0]), b=np.array([1.0, 1.0]), color=color,
                    quantiles=[0.9, 0.5, 0.1], boundary_method="truncnorm", fill=False,
                )
            except ValueError as error:
                if "Contour levels must be increasing" not in str(error):
                    raise
                continue
            contour_labels.append((label, color))
        if not contour_labels:
            plt.close(fig)
            return None
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$\chi_{\rm eff}$")
        ax.set_ylabel(r"$\chi_p$")
        ax.legend(
            handles=[Line2D([], [], color=color, label=label) for label, color in contour_labels],
            loc="upper left", frameon=True,
        )
    format_figure_ticks(fig)
    fig.savefig(f"{result_folder}/{output_name}", dpi=400, bbox_inches="tight")
    return fig


def component_chi_eff_chi_p_filled_composite_plot(
    component_ppds, bands_frame, letters, result_folder, labels, colors,
    output_name="component-chi-eff-chi-p-PPD-composite.png",
):
    """Filled chi_eff-chi_p contours beside stacked chi_eff/chi_p PPD marginals.

    Left panel: filled component-by-component KDE contours in the
    (chi_eff, chi_p) plane, layered back-to-front so the medium-spin
    component's typically narrow, high-density contour is never hidden
    underneath the others; the slow (low) spin component's native scale is
    usually a tiny fraction of the full domain, leaving its 2D contour
    numerically degenerate here, so it is skipped in this panel (it still
    shows up cleanly in the 1D marginal panels, which is what matters).
    Right panels: precomputed chi_eff/chi_p PPD bands, stacked chi_eff-over-chi_p.
    """
    sampled = [sample.sample(10_000, replace=True, random_state=10) for sample in component_ppds]

    fig = plt.figure(figsize=(AASTEX_TEXT_WIDTH_IN, 0.55 * AASTEX_TEXT_WIDTH_IN), dpi=200)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], hspace=0.3, wspace=0.32)
    contour_ax = fig.add_subplot(gs[:, 0])
    chi_eff_ax = fig.add_subplot(gs[0, 1])
    chi_p_ax = fig.add_subplot(gs[1, 1])

    # Draw order back-to-front: highest spin down to index 2, then medium
    # spin (index 1) last/on top; index 0 (slow spin) is skipped -- see
    # docstring.
    z_order = [*range(len(letters) - 1, 1, -1), 1]
    contour_labels = {}
    for component_index in z_order:
        sample, label, color = sampled[component_index], labels[component_index], colors[component_index]
        try:
            add_contours(
                contour_ax, sample["chi_eff"].to_numpy(), sample["chi_p"].to_numpy(),
                a=np.array([-1.0, 0.0]), b=np.array([1.0, 1.0]), color=color,
                quantiles=[0.9, 0.5, 0.1], boundary_method="truncnorm", fill=True,
            )
        except ValueError as error:
            if "Contour levels must be increasing" not in str(error):
                raise
            continue
        contour_labels[component_index] = (label, color)

    contour_ax.set_xlim(-1, 1)
    contour_ax.set_ylim(0, 1)
    contour_ax.set_xlabel(r"$\chi_{\rm eff}$")
    contour_ax.set_ylabel(r"$\chi_p$")

    marginal_axes = {"chi_eff": chi_eff_ax, "chi_p": chi_p_ax}
    marginal_xlabels = {"chi_eff": r"$\chi_{\rm eff}$", "chi_p": r"$\chi_p$"}
    marginal_ylabels = {"chi_eff": r"$p(\chi_{\rm eff})$", "chi_p": r"$p(\chi_p)$"}
    for name in ("chi_eff", "chi_p"):
        ax = marginal_axes[name]
        rows = bands_frame[bands_frame["grid_name"] == name]
        centers = rows["bin_center"].to_numpy()
        non_low_spin_upper = []
        for component_index, (letter, color) in enumerate(zip(letters, colors)):
            lower = rows[f"{letter}_lower"].to_numpy()
            median = rows[f"{letter}_median"].to_numpy()
            upper = rows[f"{letter}_upper"].to_numpy()
            if component_index > 0:
                non_low_spin_upper.append(np.max(upper))
            ax.fill_between(centers, lower, upper, color=color, alpha=0.25, linewidth=0)
            ax.plot(centers, median, color=color, linewidth=1.5)
        ax.set_xlim(centers.min(), centers.max())
        if non_low_spin_upper:
            ax.set_ylim(0, 1.05 * max(non_low_spin_upper))
        ax.set_xlabel(marginal_xlabels[name])
        ax.set_ylabel(marginal_ylabels[name])

    chi_p_ax.legend(
        handles=[Line2D([], [], color=color, label=label) for label, color in zip(labels, colors)],
        loc="upper right", frameon=True,
    )

    fig.align_ylabels([chi_eff_ax, chi_p_ax])
    format_figure_ticks(fig)
    fig.savefig(f"{result_folder}/{output_name}", dpi=200, bbox_inches="tight")
    return fig


def component_derived_spin_ppd_bands_plot(bands_frame, letters, result_folder, labels, colors):
    """Standalone chi_eff/chi_p PPD band figures, from a precomputed bands frame."""
    for name, xlabel, ylabel in (
        ("chi_eff", r"$\chi_{\rm eff}$", r"$p(\chi_{\rm eff})$"),
        ("chi_p", r"$\chi_p$", r"$p(\chi_p)$"),
    ):
        rows = bands_frame[bands_frame["grid_name"] == name]
        centers = rows["bin_center"].to_numpy()
        fig, ax = plt.subplots(figsize=(APJ_HALF_COLUMN_WIDTH_IN, APJ_HALF_COLUMN_WIDTH_IN), dpi=400)
        non_low_spin_upper = []
        for component_index, (letter, label, color) in enumerate(zip(letters, labels, colors)):
            lower = rows[f"{letter}_lower"].to_numpy()
            median = rows[f"{letter}_median"].to_numpy()
            upper = rows[f"{letter}_upper"].to_numpy()
            if component_index > 0:
                non_low_spin_upper.append(np.max(upper))
            ax.fill_between(centers, lower, upper, color=color, alpha=0.25, linewidth=0)
            ax.plot(centers, median, color=color, linewidth=1.5, label=label)
        ax.set_xlim(centers.min(), centers.max())
        if non_low_spin_upper:
            ax.set_ylim(0, 1.05 * max(non_low_spin_upper))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", frameon=True)
        format_figure_ticks(fig)
        fig.savefig(
            f"{result_folder}/component-by-component-{name.replace('_', '-')}-PPD.pdf",
            dpi=400, bbox_inches="tight",
        )


# -----------------------------------------------------------------------------
# Mass-ratio hyperparameters and PPD
# -----------------------------------------------------------------------------


def mass_ratio_hyperparameter_scatter(posteriors, result_folder, letters, labels, colors):
    fig, ax = plt.subplots(figsize=(APJ_HALF_COLUMN_WIDTH_IN, APJ_HALF_COLUMN_WIDTH_IN), dpi=400)
    for letter, label, color in zip(letters, labels, colors):
        ax.scatter(
            posteriors[f"mu_mass_ratio_{letter}"], posteriors[f"sigma_mass_ratio_{letter}"],
            color=color, alpha=0.20, s=8, linewidths=0, rasterized=True, label=label,
        )
    ax.set_xlabel(r"$\mu_q$")
    ax.set_ylabel(r"$\sigma_q$")
    ax.legend(loc="upper right", frameon=True)
    format_figure_ticks(fig)
    fig.savefig(f"{result_folder}/mass-ratio-hyperparameters-scatter.pdf", dpi=400, bbox_inches="tight")
    return fig


def mass_ratio_hyperparameter_kde(posteriors, result_folder, letters, labels, colors):
    samples = []
    for letter in letters:
        sample = posteriors[[f"mu_mass_ratio_{letter}", f"sigma_mass_ratio_{letter}"]].rename(
            columns={f"mu_mass_ratio_{letter}": "mu_q", f"sigma_mass_ratio_{letter}": "sigma_q"}
        )
        samples.append(sample)
    active = [
        (sample, label, color) for sample, label, color in zip(samples, labels, colors)
        if np.ptp(sample["mu_q"].to_numpy()) >= 1e-3 and np.ptp(sample["sigma_q"].to_numpy()) >= 1e-3
    ]
    if not active:
        return None
    active_samples, active_labels, active_colors = map(list, zip(*active))
    fig = make_2D_comparison2(
        active_samples, variables=["mu_q", "sigma_q"], variable_labels=[r"$\mu_q$", r"$\sigma_q$"],
        a=[0.4, 0.2], b=[1.2, 0.8], model_labels=active_labels, colors=active_colors,
        quantiles=[0.9, 0.5, 0.1], scatter=False, marginals=False, legend_location="upper right",
        boundary_method="truncnorm", figsize=(APJ_HALF_COLUMN_WIDTH_IN, APJ_HALF_COLUMN_WIDTH_IN), dpi=400,
    )
    format_figure_ticks(fig)
    fig.savefig(f"{result_folder}/mass-ratio-hyperparameters-kde.pdf", dpi=400, bbox_inches="tight")
    return fig


def mass_ratio_ppd_band_plot(bands_frame, letters, result_folder, labels, colors):
    q_grid = bands_frame["q"].to_numpy()
    fig, ax = plt.subplots(figsize=(APJ_HALF_COLUMN_WIDTH_IN, APJ_HALF_COLUMN_WIDTH_IN), dpi=400)
    for letter, label, color in zip(letters, labels, colors):
        ax.fill_between(
            q_grid, bands_frame[f"{letter}_lower"], bands_frame[f"{letter}_upper"],
            color=color, alpha=0.25, linewidth=0,
        )
        ax.plot(q_grid, bands_frame[f"{letter}_median"], color=color, linewidth=1.5, label=label)
    ax.set_xlim(0.01, 1.0)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$p(q)$")
    ax.legend(loc="upper left", frameon=True)
    format_figure_ticks(fig)
    fig.savefig(f"{result_folder}/mass-ratio-PPD.pdf", dpi=400, bbox_inches="tight")
    return fig


# -----------------------------------------------------------------------------
# Mass-contribution PPD
# -----------------------------------------------------------------------------


def component_mass_contribution_ppd_plot(
    bands_frame, letters, result_folder, labels, colors, output_path, figsize, mass_limits=(3.0, 300.0),
):
    mass_grid = bands_frame["mass_1_source"].to_numpy()
    fig, ax = plt.subplots(figsize=figsize, dpi=400)
    y_upper = 0.0
    for letter, label, color in zip(letters, labels, colors):
        lower = bands_frame[f"{letter}_lower"].to_numpy()
        median = bands_frame[f"{letter}_median"].to_numpy()
        upper = bands_frame[f"{letter}_upper"].to_numpy()
        y_upper = max(y_upper, np.max(upper))
        ax.fill_between(mass_grid, lower, upper, color=color, alpha=0.25, linewidth=0)
        ax.plot(mass_grid, median, color=color, linewidth=1.75, label=label)

    total_lower = bands_frame["total_lower"].to_numpy()
    total_median = bands_frame["total_median"].to_numpy()
    total_upper = bands_frame["total_upper"].to_numpy()
    y_upper = max(y_upper, np.max(total_upper))
    ax.plot(mass_grid, total_median, color="black", linestyle=":", linewidth=1.75, label="Total")
    ax.plot(mass_grid, total_lower, color="black", linestyle=":", linewidth=0.9, alpha=0.65)
    ax.plot(mass_grid, total_upper, color="black", linestyle=":", linewidth=0.9, alpha=0.65)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*mass_limits)
    ax.set_ylim(bottom=np.interp(200.0, mass_grid, total_median), top=1.1 * y_upper)
    ax.set_xlabel(r"$m_1\,[M_\odot]$")
    ax.set_ylabel(r"$p(m_1)\,[M_\odot^{-1}]$")
    ax.legend(loc="upper right", frameon=True)
    format_figure_ticks(fig)
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    return fig


# -----------------------------------------------------------------------------
# Component-by-component 2D spin scatter/KDE (magnitude and tilt)
# -----------------------------------------------------------------------------


def component_by_component_mag_ppd_plot(
    ppds, result_folder, ordered_source_indices, model_labels, colors, legend_location="upper left", **kwargs
):
    component_samples = get_component_ppd_samples(ppds, component_indices=ordered_source_indices)
    sampled = [s.sample(10_000, replace=True, random_state=10) for s in component_samples]
    fig = make_2D_comparison2(
        sampled, variables=["chi_1", "chi_2"], variable_labels=["$\\chi_1$", "$\\chi_2$"],
        a=[0, 0], b=[1, 1], model_labels=model_labels, colors=colors,
        scatter=True, alpha=0.05, legend_location=legend_location, boundary_method="truncnorm", **kwargs,
    )
    fig = fix_legend(fig, model_labels=model_labels, legend_location=legend_location)
    format_figure_ticks(fig)
    fig.savefig(f"{result_folder}/component-by-component-mag-PPD.pdf", dpi=200)
    return fig


def component_by_component_tilt_ppd_plot(
    ppds, result_folder, ordered_source_indices, model_labels, colors, legend_location="upper left", **kwargs
):
    component_samples = get_component_ppd_samples(ppds, component_indices=ordered_source_indices)
    sampled = [s.sample(10_000, replace=True, random_state=10) for s in component_samples]
    fig = make_2D_comparison2(
        sampled, variables=["cos_tilt_1", "cos_tilt_2"], variable_labels=["$\\cos\\theta_1$", "$\\cos\\theta_2$"],
        a=[-1, -1], b=[1, 1], model_labels=model_labels, colors=colors,
        scatter=True, alpha=0.05, legend_location=legend_location, boundary_method="truncnorm", **kwargs,
    )
    fig = fix_legend(fig, model_labels=model_labels, legend_location=legend_location)
    ax = fig.axes[0]
    for artist_index, artist in enumerate(ax.collections[: len(colors)]):
        artist.set_zorder(len(colors) - artist_index)
    format_figure_ticks(fig)
    fig.savefig(f"{result_folder}/component-by-component-tilt-PPD.pdf", dpi=200, bbox_inches="tight")
    return fig


def new_component_by_component_mag_ppd_plot(
    ppds, result_folder, ordered_source_indices, model_labels, colors, legend_location="upper left", **kwargs
):
    component_samples = get_component_ppd_samples(ppds, component_indices=ordered_source_indices)
    sampled = [s.sample(10_000, replace=True, random_state=10) for s in component_samples]
    fig = make_2D_comparison2(
        sampled, variables=["chi_1", "chi_2"], variable_labels=["$\\chi_1$", "$\\chi_2$"],
        a=[0, 0], b=[1, 1], model_labels=model_labels, quantiles=[0.9, 0.5, 0.1],
        scatter=False, alpha=0.05, legend_location=legend_location, boundary_method="truncnorm", **kwargs,
    )
    fig = fix_legend(fig, model_labels=model_labels, legend_location=legend_location)
    format_figure_ticks(fig)
    fig.savefig(f"{result_folder}/component-by-component-mag-PPD_kde.pdf", dpi=200)
    return fig


def new_component_by_component_tilt_ppd_plot(
    ppds, result_folder, ordered_source_indices, model_labels, colors, legend_location="upper left", **kwargs
):
    """KDE variant; the last (highest-spin) component is drawn first, at
    reduced alpha, so it never visually swamps the others' typically much
    narrower, higher-density contours."""
    component_samples = get_component_ppd_samples(ppds, component_indices=ordered_source_indices)
    sampled = [s.sample(10_000, replace=True, random_state=10) for s in component_samples]
    n_components = len(colors)
    draw_order = list(range(n_components - 1, -1, -1))
    figsize = kwargs.pop("figsize", (APJ_HALF_COLUMN_WIDTH_IN, APJ_HALF_COLUMN_WIDTH_IN))
    dpi = kwargs.pop("dpi", 400)
    boundary_method = kwargs.pop("boundary_method", "truncnorm")
    quantiles = kwargs.pop("quantiles", [0.9, 0.5, 0.1])
    high_spin_line_alpha = 0.45
    high_spin_fill_alpha_map = {0.9: 0.10, 0.5: 0.18, 0.1: 0.26}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for component_index in draw_order:
        sample = sampled[component_index]
        fill_alpha_map = None
        line_color = colors[component_index]
        if component_index == n_components - 1:
            line_color = to_rgba(colors[component_index], alpha=high_spin_line_alpha)
            fill_alpha_map = high_spin_fill_alpha_map

        add_contours(
            ax, sample["cos_tilt_1"].to_numpy(), sample["cos_tilt_2"].to_numpy(),
            a=np.array([-1, -1]), b=np.array([1, 1]), color=line_color,
            quantiles=quantiles, boundary_method=boundary_method, fill=True, fill_alpha_map=fill_alpha_map,
        )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("$\\cos\\theta_1$")
    ax.set_ylabel("$\\cos\\theta_2$")
    legend_handles = [
        Line2D([], [], color=colors[i], ls="solid", label=model_labels[i], linewidth=1.75)
        for i in range(n_components)
    ]
    leg = ax.legend(handles=legend_handles, loc=legend_location, fontsize=plt.rcParams["legend.fontsize"], frameon=True)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.7)
    leg.get_frame().set_edgecolor("none")
    format_figure_ticks(fig)
    fig.savefig(f"{result_folder}/component-by-component-tilt-PPD_kde.pdf", dpi=200, bbox_inches="tight")
    return fig


# -----------------------------------------------------------------------------
# Overlaid histogram / posterior-band generics
# -----------------------------------------------------------------------------


def overlaid_histogram_plot(
    series_list, labels, colors, bins, xlabel, ylabel, output_path, xlim=None,
    figsize=(APJ_HALF_COLUMN_WIDTH_IN, APJ_HALF_COLUMN_WIDTH_IN), legend_location="upper left",
    y_break=None, exclude_first_from_ylim=False,
):
    if y_break is not None:
        return overlaid_broken_y_histogram_plot(
            series_list, labels, colors, bins, xlabel, ylabel, output_path,
            xlim=xlim, figsize=figsize, legend_location=legend_location, y_break=y_break,
        )

    fig, ax = plt.subplots(1, dpi=400)
    for series, label, color in zip(series_list, labels, colors):
        values = np.asarray(series)
        ax.hist(values, bins=bins, density=True, color=color, alpha=0.3, histtype="stepfilled")
        ax.hist(values, bins=bins, density=True, color=color, histtype="step", linewidth=1.5, label=label)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if exclude_first_from_ylim and len(series_list) > 1:
        density_maxima = [
            np.histogram(np.asarray(series), bins=bins, density=True)[0].max() for series in series_list[1:]
        ]
        ax.set_ylim(0, 1.05 * max(density_maxima))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_location)
    fig = fix_legend(fig, ax=ax, model_labels=labels, legend_location=legend_location)
    fig.set_size_inches(figsize)
    format_figure_ticks(fig)
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    return fig


def overlaid_broken_y_histogram_plot(
    series_list, labels, colors, bins, xlabel, ylabel, output_path, xlim, figsize, legend_location, y_break,
):
    """Draw a density histogram with a fixed omitted interval on the y-axis."""
    lower_max, upper_min = y_break
    densities = [np.histogram(np.asarray(series), bins=bins, density=True)[0] for series in series_list]
    upper_max = max(upper_min * 1.05, max(density.max() for density in densities) * 1.05)
    fig, (ax_upper, ax_lower) = plt.subplots(
        2, 1, sharex=True, figsize=(figsize[0], figsize[1] * 1.2), dpi=400,
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.06},
    )
    for series, label, color in zip(series_list, labels, colors):
        for ax in (ax_upper, ax_lower):
            ax.hist(series, bins=bins, density=True, color=color, alpha=0.3, histtype="stepfilled")
            ax.hist(
                series, bins=bins, density=True, color=color, histtype="step",
                linewidth=1.5, label=label if ax is ax_upper else None,
            )

    if xlim is not None:
        ax_lower.set_xlim(*xlim)
    ax_lower.set_ylim(0, lower_max)
    ax_upper.set_ylim(upper_min, upper_max)
    ax_upper.spines["bottom"].set_visible(False)
    ax_lower.spines["top"].set_visible(False)
    ax_upper.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_upper.legend(loc=legend_location)
    ax_lower.set_xlabel(xlabel)
    fig.supylabel(ylabel)

    marker_size = 0.015
    for ax, y in ((ax_upper, 0), (ax_lower, 1)):
        ax.plot((-marker_size, marker_size), (y - marker_size, y + marker_size),
                transform=ax.transAxes, color="k", clip_on=False, linewidth=0.8)
        ax.plot((1 - marker_size, 1 + marker_size), (y - marker_size, y + marker_size),
                transform=ax.transAxes, color="k", clip_on=False, linewidth=0.8)

    format_figure_ticks(fig)
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    return fig


def overlaid_posterior_band_plot(
    band_specs, xlabel, ylabel, output_path, xlim=None,
    figsize=(APJ_HALF_COLUMN_WIDTH_IN, APJ_HALF_COLUMN_WIDTH_IN), legend_location="upper left",
):
    fig, ax = plt.subplots(1, dpi=400)
    for spec in band_specs:
        ax.fill_between(spec["x_grid"], spec["lower"], spec["upper"], color=spec["color"], alpha=0.25, linewidth=0)
        ax.plot(spec["x_grid"], spec["median"], color=spec["color"], linewidth=1.75, label=spec["label"])

    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_location)
    fig = fix_legend(fig, ax=ax, model_labels=[spec["label"] for spec in band_specs], legend_location=legend_location)
    fig.set_size_inches(figsize)
    format_figure_ticks(fig)
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    return fig


def zeta_cos_tilt_histogram_plot(df, letters, labels, colors, result_folder):
    columns = [f"zeta_cos_tilt_{letter}" for letter in letters]
    if not set(columns).issubset(df.columns):
        return None
    return overlaid_histogram_plot(
        series_list=[df[c] for c in columns], labels=labels, colors=colors,
        bins=np.linspace(0, 1, 50), xlabel=r"$\zeta_{\cos \theta}$", ylabel=r"$p(\zeta_{\cos \theta})$",
        output_path=f"{result_folder}/zeta-cos-tilt-hyperposterior.pdf", xlim=(0, 1),
        exclude_first_from_ylim=True,
    )


def high_spin_tilt_marginal_plot(bands_frame, high_spin_label, colors, result_folder):
    tilt_grid = bands_frame["cos_tilt"].to_numpy()
    return overlaid_posterior_band_plot(
        band_specs=[
            {
                "x_grid": tilt_grid, "lower": bands_frame["tilt1_lower"], "median": bands_frame["tilt1_median"],
                "upper": bands_frame["tilt1_upper"], "color": colors[-1],
                "label": rf"{high_spin_label} $\cos(\theta_1)$",
            },
            {
                "x_grid": tilt_grid, "lower": bands_frame["tilt2_lower"], "median": bands_frame["tilt2_median"],
                "upper": bands_frame["tilt2_upper"], "color": sns.desaturate(colors[-1], 0.45),
                "label": rf"{high_spin_label} $\cos(\theta_2)$",
            },
        ],
        xlabel=r"$\cos(\theta)$", ylabel=r"$p(\cos(\theta))$",
        output_path=f"{result_folder}/high-spin-costilt-marginals-PPD.pdf", xlim=(-1, 1),
    )


# -----------------------------------------------------------------------------
# Spin-hyperparameter comparison corner
# -----------------------------------------------------------------------------


def spin_hyperparameter_comparison_plot(df, letters, labels, result_folder, n_plot_samples=1000):
    """Compare mu_chi/sigma_chi across every component's primary and secondary."""
    n_plot_samples = min(n_plot_samples, len(df))
    all_samples, all_labels = [], []
    # Reverse component order to match historical high-to-low spin plotting order.
    for letter, label in reversed(list(zip(letters, labels))):
        for body in ("1", "2"):
            sample = df[[f"mu_chi_{body}_{letter}", f"sigma_chi_{body}_{letter}"]].rename(
                columns={f"mu_chi_{body}_{letter}": "mu_chi", f"sigma_chi_{body}_{letter}": "sigma_chi"}
            ).sample(n_plot_samples)
            body_name = "Primary" if body == "1" else "Secondary"
            all_samples.append(sample)
            all_labels.append(f"{label} {body_name}")

    active = [
        (sample, label) for sample, label in zip(all_samples, all_labels)
        if np.ptp(sample["mu_chi"].to_numpy()) >= 1e-3 and np.ptp(sample["sigma_chi"].to_numpy()) >= 1e-3
    ]
    output_path = f"{result_folder}/spin-hyperparameter-comparison-one-plot.pdf"
    if not active:
        fig, ax = plt.subplots(figsize=(APJ_HALF_COLUMN_WIDTH_IN, APJ_HALF_COLUMN_WIDTH_IN), dpi=200)
        ax.axis("off")
        ax.text(0.5, 0.5, "KDE omitted:\nall spin hyperposteriors are degenerate.", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_path, dpi=200)
        return fig

    active_samples, active_labels = map(list, zip(*active))
    try:
        fig = make_2D_comparison2(
            active_samples, variables=["mu_chi", "sigma_chi"],
            variable_labels=[r"$\mu^{A,B,C}_{\chi_{1,2}}$", r"$\sigma^{A,B,C}_{\chi_{1,2}}$"],
            model_labels=active_labels, legend=False, quantiles=[0.9, 0.5, 0.1], marginals=False,
            scatter=False, legend_location="upper left", boundary_method="truncnorm",
        )
        format_figure_ticks(fig)
        fig.savefig(output_path, dpi=200)
    except ValueError as exc:
        if "Contour levels must be increasing" not in str(exc):
            raise
        print("Skipping degenerate spin-hyperparameter KDE comparison:", exc)
        fig = None
    return fig


def slow_spin_hyperparameter_corner_plot(df, low_spin_letter, color, result_folder):
    parameter_labels = {
        f"mu_chi_1_{low_spin_letter}": r"$\mu_{\chi_1}^A$",
        f"sigma_chi_1_{low_spin_letter}": r"$\sigma_{\chi_1}^A$",
        f"mu_chi_2_{low_spin_letter}": r"$\mu_{\chi_2}^A$",
        f"sigma_chi_2_{low_spin_letter}": r"$\sigma_{\chi_2}^A$",
        f"rho_chi_{low_spin_letter}": r"$\rho_{\chi}^A$",
    }
    variables = []
    for name in parameter_labels:
        if name not in df:
            continue
        values = df[name].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) > 1 and np.ptp(finite) > 1e-8:
            variables.append(name)

    try:
        fig, axes = make_corner_plot(
            all_data=[{k: df[k].values for k in df.columns}], variables=variables,
            variable_labels=[parameter_labels[name] for name in variables],
            kde=True, quantiles=[0.9, 0.5, 0.1], boundary_method="truncnorm",
            scatter=False, legend=False, colors=[color], hist_linewidth=1, kde_linewidth=1,
            figsize=(AASTEX_TEXT_WIDTH_IN, AASTEX_TEXT_WIDTH_IN),
        )
    except ValueError as exc:
        if "NaN" not in str(exc):
            raise
        fig, ax = plt.subplots(figsize=(AASTEX_TEXT_WIDTH_IN, AASTEX_TEXT_WIDTH_IN), dpi=200)
        ax.axis("off")
        ax.text(0.5, 0.5, "KDE corner omitted:\ndegenerate hyperposterior samples.", ha="center", va="center", transform=ax.transAxes)
        axes = np.array([[ax]])

    for ax in fig.axes:
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=LATEX_FONT_SIZES_PT["normalsize"])
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=LATEX_FONT_SIZES_PT["normalsize"])
        ax.tick_params(axis="x", labelsize=LATEX_FONT_SIZES_PT["footnotesize"])
        ax.tick_params(axis="y", labelsize=LATEX_FONT_SIZES_PT["footnotesize"])
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))

    for row_index in range(axes.shape[0]):
        for col_index in range(axes.shape[1]):
            ax = axes[row_index, col_index]
            if col_index > row_index:
                continue
            if row_index != axes.shape[0] - 1:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
            if col_index != 0:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)

    format_figure_ticks(fig)
    for ax in fig.axes:
        ax.tick_params(axis="x", labelsize=8)
        for text in ax.texts:
            if text.get_text():
                text.set_fontsize(LATEX_FONT_SIZES_PT["normalsize"])

    fig.set_dpi(200)
    fig.savefig(f"{result_folder}/slow_spin_hyperparameters.png", dpi=200, bbox_inches="tight")
    # Preserve the historical misspelling for existing manuscripts and plot browsers.
    fig.savefig(f"{result_folder}/slow_spin_hyperparameteres.png", dpi=200, bbox_inches="tight")
    return fig


def slow_spin_magnitude_tilt_kde_plot(samples, result_folder, color):
    """Plot the two requested low-spin magnitude--tilt KDE projections."""
    panel_specs = [
        ("chi_1", "cos_tilt_1", r"$\chi_1$", r"$\cos(\theta_1)$"),
        ("chi_2", "cos_tilt_2", r"$\chi_2$", r"$\cos(\theta_2)$"),
    ]
    fig, axes = plt.subplots(
        2, 1, figsize=(APJ_HALF_COLUMN_WIDTH_IN, 1.75 * APJ_HALF_COLUMN_WIDTH_IN), dpi=400, constrained_layout=True,
    )
    for ax, (x_name, y_name, x_label, y_label) in zip(axes, panel_specs):
        add_contours(
            ax, samples[x_name].to_numpy(), samples[y_name].to_numpy(),
            a=np.array([0.0, -0.2]), b=np.array([0.2, 0.2]), color=color,
            quantiles=[0.99, 0.9, 0.5, 0.1], boundary_method="truncnorm", fill=True,
        )
        ax.set_xlim(0.0, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_xticks([0.0, 0.1, 0.2])
        ax.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
        ax.set_xlabel(x_label, fontsize=LATEX_FONT_SIZES_PT["large"])
        ax.set_ylabel(y_label, fontsize=LATEX_FONT_SIZES_PT["large"])
        ax.tick_params(labelsize=LATEX_FONT_SIZES_PT["large"])
    format_figure_ticks(fig)
    fig.savefig(f"{result_folder}/slow_spin_PPD.png", dpi=400, bbox_inches="tight")
    return fig


# -----------------------------------------------------------------------------
# Mass-break hyperparameter corner
# -----------------------------------------------------------------------------


def shade_ordered_mass_break_region(axes, limits, zorder=0):
    """Shade the physically excluded region m_i^eta > m_j^eta in each panel."""
    for row_index in range(axes.shape[0]):
        for col_index in range(row_index):
            ax = axes[row_index, col_index]
            x_lower, x_upper = limits[col_index]
            y_lower, y_upper = limits[row_index]
            x_values = np.linspace(x_lower, x_upper, 256)
            y_upper_boundary = np.clip(x_values, y_lower, y_upper)
            ax.fill_between(
                x_values, y_lower, y_upper_boundary, where=x_values > y_lower,
                color="0.75", edgecolor="0.48", linewidth=0.8, zorder=zorder,
            )


def format_corner_outer_axes(axes):
    """Keep labels and tick labels only on the conventional outer corner axes."""
    for row_index in range(axes.shape[0]):
        for col_index in range(axes.shape[1]):
            ax = axes[row_index, col_index]
            if col_index > row_index:
                continue
            if row_index != axes.shape[0] - 1:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
            if col_index != 0:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)


def add_mass_break_kde_contours(axes, data, limits, mass_break_columns, color):
    """Add three visibly distinct KDE credible regions to the mass-break corner."""
    for row_index in range(axes.shape[0]):
        for col_index in range(row_index):
            x_name = mass_break_columns[col_index]
            y_name = mass_break_columns[row_index]
            x_values = np.asarray(data[x_name], dtype=float)
            y_values = np.asarray(data[y_name], dtype=float)
            finite = np.isfinite(x_values) & np.isfinite(y_values)
            x_values = x_values[finite]
            y_values = y_values[finite]
            x_width = np.ptp(x_values) if x_values.size else 0.0
            y_width = np.ptp(y_values) if y_values.size else 0.0
            if x_values.size < 3 or x_width <= 1e-10 or y_width <= 1e-10:
                print(f"Skipping degenerate mass-break KDE panel: {x_name} vs {y_name}")
                continue
            add_contours(
                axes[row_index, col_index], x_values, y_values,
                a=np.array([limits[col_index][0], limits[row_index][0]]),
                b=np.array([limits[col_index][1], limits[row_index][1]]),
                color=color, quantiles=[0.9, 0.5, 0.1], linewidths=1, boundary_method="truncnorm",
                fill=True, fill_alpha_map={0.9: 0.16, 0.5: 0.24, 0.1: 0.34},
            )


def mass_break_corner_plots(df, color, result_folder):
    """KDE and scatter mass-break corners; no-op if the run has no break columns."""
    mass_break_columns = sorted(
        (c for c in df.columns if c.startswith("spin_fraction_break_mass_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    if not mass_break_columns:
        return None, None

    mass_break_limits = []
    for column in mass_break_columns:
        lower, _, upper = hpd_array(df[column].to_numpy(), alpha=0.05)
        mass_break_limits.append((lower, upper))
    mass_break_labels = [rf"$m_{{{i + 1}}}^{{\eta}}\,[M_\odot]$" for i in range(len(mass_break_columns))]
    mass_break_data = {column: df[column].to_numpy() for column in mass_break_columns}

    fig_kde, axes = make_corner_plot(
        all_data=[mass_break_data], variables=mass_break_columns, variable_labels=mass_break_labels,
        limits=mass_break_limits, kde=False, scatter=False, legend=False, colors=[color],
        hist_linewidth=1, kde_linewidth=1, figsize=(AASTEX_TEXT_WIDTH_IN, AASTEX_TEXT_WIDTH_IN),
    )
    add_mass_break_kde_contours(axes, mass_break_data, mass_break_limits, mass_break_columns, color)
    shade_ordered_mass_break_region(axes, mass_break_limits, zorder=10)
    format_corner_outer_axes(axes)
    format_figure_ticks(fig_kde)
    fig_kde.savefig(f"{result_folder}/mass-break-hyperparameters-corner.pdf", dpi=400, bbox_inches="tight")

    fig_scatter, axes = make_corner_plot(
        all_data=[mass_break_data], variables=mass_break_columns, variable_labels=mass_break_labels,
        limits=mass_break_limits, kde=False, scatter=True, alpha=0.12, scatter_kwargs={"s": 3},
        legend=False, colors=[color], hist_linewidth=1, figsize=(AASTEX_TEXT_WIDTH_IN, AASTEX_TEXT_WIDTH_IN),
    )
    shade_ordered_mass_break_region(axes, mass_break_limits)
    format_corner_outer_axes(axes)
    format_figure_ticks(fig_scatter)
    fig_scatter.savefig(f"{result_folder}/mass-break-hyperparameters-corner-scatter.pdf", dpi=400, bbox_inches="tight")
    return fig_kde, fig_scatter
