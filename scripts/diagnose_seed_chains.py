#!/usr/bin/env python
"""Diagnose independent single-chain runs and make an exploratory thinned merge."""

import argparse
import json
import math
import shutil
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SELECTED_PARAMETERS = [
    "eta_spin_comp0_regime0",
    "eta_spin_comp1_regime0",
    "eta_spin_comp2_regime0",
    "mu_chi_1_a",
    "mu_chi_1_b",
    "mu_chi_1_c",
]
COMPONENT_LETTERS = ("a", "b", "c")
MIN_DRAWS_FOR_IAT = 20
EXPECTED_CONSTANT_PARAMETERS = {"mmax"}


def canonicalize_component_labels(samples, posterior):
    """Relabel each chain by increasing median spin before chains are combined."""
    scores = {
        letter: float(np.median(0.5 * (posterior[f"mu_chi_1_{letter}"] + posterior[f"mu_chi_2_{letter}"])))
        for letter in COMPONENT_LETTERS
    }
    ordered_letters = tuple(sorted(COMPONENT_LETTERS, key=scores.get))
    source_to_target_letter = dict(zip(ordered_letters, COMPONENT_LETTERS))
    source_to_target_index = {
        COMPONENT_LETTERS.index(source): target_index
        for target_index, source in enumerate(ordered_letters)
    }

    def rename(name):
        renamed = name
        for source, target in source_to_target_letter.items():
            if renamed.endswith(f"_{source}"):
                renamed = f"{renamed[:-2]}_{target}"
                break
        for source, target in source_to_target_index.items():
            token = f"eta_spin_comp{source}_"
            if token in renamed:
                renamed = renamed.replace(token, f"eta_spin_comp{target}_")
                break
        return renamed

    renamed_samples = samples.rename(columns={column: rename(column) for column in samples.columns})
    renamed_posterior = {rename(name): values for name, values in posterior.items()}
    mapping = {target.upper(): source.upper() for source, target in source_to_target_letter.items()}
    return renamed_samples, renamed_posterior, mapping


def autocorrelation(values, max_lag):
    """Return the normalized FFT autocorrelation for a scalar trace."""
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    variance = np.dot(centered, centered)
    if not np.isfinite(variance) or variance <= 0:
        return None
    fft_values = np.fft.rfft(centered, n=2 * len(values))
    acf = np.fft.irfft(fft_values * np.conjugate(fft_values), n=2 * len(values))[: len(values)]
    acf /= acf[0]
    return acf[: min(max_lag + 1, len(acf))]


def integrated_autocorrelation_time(acf, c=5.0):
    """Sokal-window IAT estimate; returns NaN for constant or invalid traces."""
    if acf is None:
        return np.nan
    cumulative_tau = 1.0 + 2.0 * np.cumsum(acf[1:])
    cumulative_tau = np.maximum(cumulative_tau, 1.0)
    windows = np.arange(1, len(cumulative_tau) + 1)
    valid = np.flatnonzero(windows >= c * cumulative_tau)
    return float(cumulative_tau[valid[0]]) if len(valid) else float(cumulative_tau[-1])


def read_chain(folder):
    with h5py.File(folder / "chain_samples.nc", "r") as handle:
        posterior = {name: np.asarray(values[0]) for name, values in handle["posterior"].items()}
        stats = {name: np.asarray(values[0]) for name, values in handle["sample_stats"].items()}
    samples = pd.read_csv(folder / "result.csv")
    if len(samples) != len(next(iter(posterior.values()))):
        raise ValueError(f"{folder.name}: result.csv and NetCDF draw counts differ")
    return samples, posterior, stats


def plot_traces(chains, output):
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(len(SELECTED_PARAMETERS), 1, sharex=True, figsize=(15, 14), constrained_layout=True)
    for parameter, axis in zip(SELECTED_PARAMETERS, axes):
        values_all = [chain["posterior"][parameter] for chain in chains]
        low, high = np.nanquantile(np.concatenate(values_all), [0.005, 0.995])
        padding = max(0.02 * (high - low), 0.02)
        marker_y = high + padding
        for index, chain in enumerate(chains):
            values = chain["posterior"][parameter]
            draw = np.arange(len(values))
            axis.plot(draw, values, color=colors[index], alpha=0.75, linewidth=0.55, label=chain["label"])
            divergent = chain["stats"].get("diverging", np.zeros(len(values), dtype=bool)).astype(bool)
            axis.plot(draw[divergent], np.full(divergent.sum(), marker_y), "|", color=colors[index], markersize=4, alpha=0.8)
        axis.set_ylim(low - padding, marker_y + padding)
        axis.set_ylabel(parameter)
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Retained draw index")
    axes[0].legend(ncol=4, fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"{len(chains)} independent seeds; top ticks mark divergent transitions", fontsize=13)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_autocorrelation(chains, parameters, tau_table, output, title, max_lag=500):
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(len(parameters), 1, sharex=True, figsize=(14, 2.25 * len(parameters)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for parameter, axis in zip(parameters, axes):
        for index, chain in enumerate(chains):
            acf = autocorrelation(chain["posterior"][parameter], max_lag)
            if acf is not None:
                tau = tau_table.loc[parameter, chain["label"]]
                axis.plot(np.arange(len(acf)), acf, color=colors[index], linewidth=1.0, label=f"{chain['label']} (tau={tau:.0f})")
        axis.axhline(0, color="0.4", linewidth=0.7)
        axis.set_ylabel(parameter)
        axis.grid(alpha=0.22)
    axes[-1].set_xlabel("Lag")
    axes[0].legend(ncol=4, fontsize=7, frameon=False, loc="upper right")
    fig.suptitle(title, fontsize=13)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="FinalResults")
    parser.add_argument("--prefix", required=True, help="Run-folder prefix before '-seedNN'.")
    parser.add_argument(
        "--exclude-seed",
        action="append",
        default=[],
        help="Seed label or number to exclude, e.g. --exclude-seed 87.",
    )
    parser.add_argument(
        "--no-relabel",
        action="store_true",
        help="Preserve the original sampled component labels in every chain.",
    )
    parser.add_argument(
        "--exclude-pathological",
        action="store_true",
        help=(
            "Exclude chains with constant sampled parameters or no finite IAT from the "
            "combined result while retaining diagnostics. Divergences are never filtered."
        ),
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    excluded = {seed if seed.startswith("seed") else f"seed{seed}" for seed in args.exclude_seed}
    folders = sorted(
        path for path in root.glob(f"{args.prefix}-seed*")
        if (path / "result.csv").is_file()
        and (path / "chain_samples.nc").is_file()
        and path.name.rsplit("-", 1)[-1] not in excluded
    )
    if len(folders) < 2:
        raise ValueError("Need at least two completed seed folders with result.csv and chain_samples.nc")

    chains = []
    for folder in folders:
        samples, posterior, stats = read_chain(folder)
        if args.no_relabel:
            component_mapping = {letter.upper(): letter.upper() for letter in COMPONENT_LETTERS}
        else:
            samples, posterior, component_mapping = canonicalize_component_labels(samples, posterior)
        missing = [name for name in SELECTED_PARAMETERS if name not in posterior]
        if missing:
            raise KeyError(f"{folder.name}: missing {missing}")
        chains.append({
            "folder": folder,
            "label": folder.name.rsplit("-", 1)[-1],
            "samples": samples,
            "posterior": posterior,
            "stats": stats,
            "component_mapping": component_mapping,
        })

    label_suffix = "-original-labels" if args.no_relabel else ""
    output = root / f"{args.prefix}-combined-{len(chains)}-chains-thinned{label_suffix}"
    output.mkdir(exist_ok=True)
    shutil.copy2(folders[0] / "model_and_priors.py", output / "model_and_priors.py")

    expected_draws = len(chains[0]["samples"])
    parameter_names = [
        name
        for name, values in chains[0]["posterior"].items()
        if name != "total_log_variance"
        and np.asarray(values).ndim == 1
        and len(values) == expected_draws
    ]
    for chain in chains[1:]:
        if set(chain["posterior"]) != set(chains[0]["posterior"]):
            raise ValueError("Posterior columns differ between chains")

    tau_table = pd.DataFrame(index=parameter_names, columns=[chain["label"] for chain in chains], dtype=float)
    for parameter in parameter_names:
        for chain in chains:
            tau_table.loc[parameter, chain["label"]] = integrated_autocorrelation_time(
                autocorrelation(chain["posterior"][parameter], max_lag=1000)
            )

    worst_by_parameter = tau_table.max(axis=1).sort_values(ascending=False)
    worst_parameters_pre_thinning = worst_by_parameter.head(5).index.tolist()
    worst_tau_by_chain = tau_table.max(axis=0)
    worst_five_by_chain_pre_thinning = {}
    for chain in chains:
        label = chain["label"]
        ranked = tau_table[label].dropna().sort_values(ascending=False).head(5)
        worst_five_by_chain_pre_thinning[label] = [
            {"parameter": parameter, "iat": float(tau)}
            for parameter, tau in ranked.items()
        ]
    pathological_exclusions = {}
    for chain in chains:
        label = chain["label"]
        reasons = []
        constant_parameters = sorted(
            name
            for name in parameter_names
            if name not in EXPECTED_CONSTANT_PARAMETERS
            and np.nanstd(chain["posterior"][name]) <= 1e-12
        )
        if constant_parameters:
            reasons.append(f"constant sampled parameters: {', '.join(constant_parameters)}")
        if not np.isfinite(worst_tau_by_chain[label]):
            reasons.append("no finite integrated autocorrelation time")
        if reasons:
            pathological_exclusions[label] = reasons

    contributing_chains = [
        chain
        for chain in chains
        if not (args.exclude_pathological and chain["label"] in pathological_exclusions)
    ]
    if not contributing_chains:
        raise ValueError("Pathology filter excluded every chain; no combined result can be made")
    thinning = {
        chain["label"]: max(1, int(math.ceil(worst_tau_by_chain[chain["label"]])))
        for chain in contributing_chains
    }

    plot_traces(chains, output / "traceplot-selected-hyperparameters.png")
    plot_autocorrelation(chains, SELECTED_PARAMETERS, tau_table, output / "autocorrelation-selected-hyperparameters.png", "Autocorrelation: lowest-regime fractions and primary-spin means")
    plot_autocorrelation(chains, worst_parameters_pre_thinning, tau_table, output / "autocorrelation-worst-five-hyperparameters.png", "Autocorrelation: five worst hyperparameters before thinning")

    thinned_frames = []
    provenance = []
    for chain_id, chain in enumerate(contributing_chains):
        step = thinning[chain["label"]]
        selected = chain["samples"].iloc[::step].copy()
        start = sum(len(frame) for frame in thinned_frames)
        thinned_frames.append(selected)
        provenance.extend(
            {
                "combined_draw": start + row,
                "chain_id": chain_id,
                "seed": chain["label"],
                "source_draw": int(draw),
                "thin_interval": step,
                "diverging": bool(
                    chain["stats"].get(
                        "diverging", np.zeros(len(chain["samples"]), dtype=bool)
                    )[draw]
                ),
            }
            for row, draw in enumerate(range(0, len(chain["samples"]), step))
        )

    combined = pd.concat(thinned_frames, ignore_index=True)
    combined.to_csv(output / "result.csv", index=False)
    pd.DataFrame(provenance).to_csv(output / "chain_index.csv", index=False)
    tau_table.to_csv(output / "integrated_autocorrelation_times.csv")

    thinned_tau_table = pd.DataFrame(
        index=parameter_names,
        columns=[chain["label"] for chain in contributing_chains],
        dtype=float,
    )
    thinned_draws_by_chain = {}
    for chain in contributing_chains:
        label = chain["label"]
        step = thinning[label]
        draw_count = len(chain["posterior"][parameter_names[0]][::step])
        thinned_draws_by_chain[label] = draw_count
        if draw_count < MIN_DRAWS_FOR_IAT:
            continue
        for parameter in parameter_names:
            values = chain["posterior"][parameter][::step]
            thinned_tau_table.loc[parameter, label] = integrated_autocorrelation_time(
                autocorrelation(values, max_lag=min(1000, draw_count - 1))
            )
    thinned_tau_table.to_csv(output / "integrated_autocorrelation_times_after_thinning.csv")

    worst_after_thinning = (
        thinned_tau_table.max(axis=1).dropna().sort_values(ascending=False)
    )
    worst_parameters = worst_after_thinning.head(5).index.tolist()
    worst_five_by_chain = {}
    for chain in contributing_chains:
        label = chain["label"]
        ranked = thinned_tau_table[label].dropna().sort_values(ascending=False).head(5)
        worst_five_by_chain[label] = {
            "draws_after_thinning": thinned_draws_by_chain[label],
            "iat_estimate_available": thinned_draws_by_chain[label] >= MIN_DRAWS_FOR_IAT,
            "parameters": [
                {"parameter": parameter, "iat": float(tau)}
                for parameter, tau in ranked.items()
            ],
        }

    divergence_counts = {
        chain["label"]: int(
            chain["stats"].get("diverging", np.zeros(len(chain["samples"]))).sum()
        )
        for chain in chains
    }
    total_divergences = sum(divergence_counts.values())
    retained_divergences = {
        chain["label"]: sum(
            bool(row["diverging"])
            for row in provenance
            if row["seed"] == chain["label"]
        )
        for chain in contributing_chains
    }
    warning = (
        "Thinning reduces autocorrelation but does not repair divergent or unconverged chains."
        if total_divergences
        else "No divergent transitions were recorded; thinning is used only to reduce autocorrelation."
    )
    diagnostics = {
        "warning": warning,
        "iat_method": "FFT autocorrelation with Sokal c=5 window; max lag 1000.",
        "source_folders": [folder.name for folder in folders],
        "excluded_seeds": sorted(excluded),
        "contributing_seeds": [chain["label"] for chain in contributing_chains],
        "pathological_chain_filter_enabled": args.exclude_pathological,
        "pathological_chain_exclusions": pathological_exclusions if args.exclude_pathological else {},
        "component_label_strategy": "original" if args.no_relabel else "median_spin_relabelled",
        "component_relabelling": {chain["label"]: chain["component_mapping"] for chain in chains},
        "divergences": divergence_counts,
        "total_divergences": total_divergences,
        "retained_divergences_after_thinning": retained_divergences,
        "worst_iat_by_chain": {
            label: float(value) if np.isfinite(value) else None
            for label, value in worst_tau_by_chain.items()
        },
        "worst_iat_by_chain_pre_thinning": {
            label: float(value) if np.isfinite(value) else None
            for label, value in worst_tau_by_chain.items()
        },
        "thinning_interval_by_chain": thinning,
        "post_thinning_iat_minimum_draws": MIN_DRAWS_FOR_IAT,
        "worst_five_hyperparameters": worst_parameters,
        "worst_five_hyperparameters_by_chain": worst_five_by_chain,
        "worst_five_hyperparameters_pre_thinning": worst_parameters_pre_thinning,
        "worst_five_hyperparameters_by_chain_pre_thinning": worst_five_by_chain_pre_thinning,
        "combined_draws": len(combined),
    }
    (output / "diagnostic_summary.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
