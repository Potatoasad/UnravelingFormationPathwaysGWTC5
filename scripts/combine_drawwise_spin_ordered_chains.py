#!/usr/bin/env python3
"""Relabel unconstrained mixture chains draw-by-draw and combine them safely.

Component A/B/C in the output are ordered at every draw by increasing
0.5 * (mu_chi_1 + mu_chi_2). Raw seed folders are never modified.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arviz_base import from_dict
from arviz_stats import summary
from arviz_stats.base import array_stats


EXPECTED_CONSTANTS = {"mmax"}

# Explanation: match any name ending with 
# an underscore and a single lowercase letter, 
# capturing the prefix and the letter separately.
SUFFIX_RE = re.compile(r"^(.*)_([a-z])$")

# Explanation: match any name starting with "eta_spin_comp",
# followed by one or more digits, and then an underscore and any characters,
# capturing the prefix, the digits, and the suffix separately.
ETA_RE = re.compile(r"^(eta_spin_comp)(\d+)(_.+)$")


def effective_sample_sizes(values: np.ndarray) -> tuple[float, float]:
    """Return ArviZ single-chain bulk and tail ESS for one parameter."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all():
        return np.nan, np.nan
    bulk = array_stats.ess(values, chain_axis=None, method="bulk")
    tail = array_stats.ess(values, chain_axis=None, method="tail", prob=0.95)
    return float(bulk), float(tail)


def read_chain(folder: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Read draw-aligned table, posterior, and sample statistics."""
    with h5py.File(folder / "chain_samples.nc", "r") as handle:
        posterior = {
            name: np.asarray(values[0])
            for name, values in handle["posterior"].items()
            if values.ndim == 2
        }
        stats = {
            name: np.asarray(values[0])
            for name, values in handle["sample_stats"].items()
            if values.ndim == 2 and name not in {"chain", "draw"}
        }
    samples = pd.read_csv(folder / "result.csv")
    samples = samples.loc[:, ~samples.columns.str.startswith("Unnamed:")]
    n_draws = len(next(iter(posterior.values())))
    if len(samples) != n_draws:
        raise ValueError(f"{folder.name}: result.csv has {len(samples)} rows but NetCDF has {n_draws}")
    if any(len(values) != n_draws for values in stats.values()):
        raise ValueError(f"{folder.name}: sample statistics are not draw-aligned")
    return samples, posterior, stats


def spin_order(
    posterior: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Infer components and return their increasing mean-spin order for every draw."""
    first = {name.removeprefix("mu_chi_1_") for name in posterior if name.startswith("mu_chi_1_")}
    second = {name.removeprefix("mu_chi_2_") for name in posterior if name.startswith("mu_chi_2_")}
    components = tuple(sorted(first & second))
    if len(components) < 2:
        raise KeyError("Cannot infer at least two components from mu_chi_1/2 columns")
    missing = [
        f"mu_chi_{body}_{component}"
        for body in (1, 2)
        for component in components
        if f"mu_chi_{body}_{component}" not in posterior
    ]
    if missing:
        raise KeyError(f"Cannot define spin ordering; missing {missing}")
    scores = np.column_stack([
        0.5 * (posterior[f"mu_chi_1_{c}"] + posterior[f"mu_chi_2_{c}"])
        for c in components
    ])
    return np.argsort(scores, axis=1, kind="stable"), scores, components


def relabel_mapping(
    data: dict[str, np.ndarray], order: np.ndarray, components: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Move every complete component-valued block according to a per-draw order."""
    result = {name: np.asarray(values).copy() for name, values in data.items()}

    suffix_groups: dict[str, dict[str, str]] = {}
    eta_groups: dict[tuple[str, str], dict[int, str]] = {}
    for name in data:
        suffix_match = SUFFIX_RE.match(name)
        if suffix_match and suffix_match.group(2) in components:
            suffix_groups.setdefault(suffix_match.group(1), {})[suffix_match.group(2)] = name
        eta_match = ETA_RE.match(name)
        if eta_match:
            key = (eta_match.group(1), eta_match.group(3))
            eta_groups.setdefault(key, {})[int(eta_match.group(2))] = name

    for base, names in suffix_groups.items():
        if set(names) != set(components):
            present = ", ".join(sorted(names))
            raise ValueError(
                f"Cannot relabel partial component block '{base}' (found {present}). "
                "This usually means the components use different model families."
            )
        values = np.column_stack([data[names[c]] for c in components])
        for target_index, target in enumerate(components):
            result[f"{base}_{target}"] = values[np.arange(len(order)), order[:, target_index]]

    for (prefix, suffix), names in eta_groups.items():
        if set(names) != set(range(len(components))):
            present = ", ".join(str(index) for index in sorted(names))
            raise ValueError(
                f"Cannot relabel partial fraction block '{prefix}*{suffix}' (found {present})."
            )
        values = np.column_stack([data[names[i]] for i in range(len(components))])
        for target_index in range(len(components)):
            result[f"{prefix}{target_index}{suffix}"] = values[
                np.arange(len(order)), order[:, target_index]
            ]
    return result


def relabel_dataframe(
    samples: pd.DataFrame, order: np.ndarray, components: tuple[str, ...]
) -> pd.DataFrame:
    arrays = relabel_mapping(
        {name: samples[name].to_numpy() for name in samples}, order, components
    )
    return pd.DataFrame(arrays, columns=samples.columns)


def scalar_parameters(posterior: dict[str, np.ndarray], n_draws: int) -> list[str]:
    return sorted(
        name for name, values in posterior.items()
        if name != "total_log_variance"
        and name not in EXPECTED_CONSTANTS
        and np.asarray(values).ndim == 1
        and len(values) == n_draws
        and np.isfinite(values).all()
    )


def compute_iats(chains: list[dict], key: str, parameter_names: list[str]) -> pd.DataFrame:
    """IAT = draws / min(bulk ESS, tail ESS), the same ESS basis as ArviZ thin_factor."""
    rows = []
    for chain in chains:
        for parameter in parameter_names:
            if parameter not in chain[key]:
                continue
            values = chain[key][parameter]
            ess_bulk, ess_tail = effective_sample_sizes(values)
            rows.append({
                "seed": chain["label"],
                "parameter": parameter,
                "ess_bulk": ess_bulk,
                "ess_tail": ess_tail,
                "iat": len(values) / min(ess_bulk, ess_tail),
            })
    return pd.DataFrame(rows)


def bfmi(energy: np.ndarray | None) -> float | None:
    if energy is None or len(energy) < 2 or np.var(energy) <= 0:
        return None
    return float(array_stats.bfmi(np.asarray(energy, dtype=float), chain_axis=None)[0])


def permutation_metadata(
    order: np.ndarray,
    scores: np.ndarray,
    stats: dict[str, np.ndarray],
    components: tuple[str, ...],
) -> pd.DataFrame:
    states = np.array(["".join(components[index].upper() for index in row) for row in order])
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    frame = pd.DataFrame({
        "source_draw": np.arange(len(order)),
        "raw_order_low_to_high": states,
        "permutation_changed": np.r_[False, states[1:] != states[:-1]],
        "minimum_adjacent_spin_gap": np.min(np.diff(sorted_scores, axis=1), axis=1),
    })
    for index, component in enumerate(components):
        frame[f"raw_mean_spin_{component}"] = scores[:, index]
        frame[f"ordered_mean_spin_{component.upper()}"] = sorted_scores[:, index]
    for name, values in stats.items():
        if np.asarray(values).ndim == 1 and len(values) == len(frame):
            frame[f"sample_stat__{name}"] = values
    return frame


def make_permutation_plot(chains: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(len(chains), 1, figsize=(15, 2.2 * len(chains)), sharex=False,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    all_states = sorted({state for chain in chains for state in chain["permutation"]["raw_order_low_to_high"]})
    state_number = {state: index for index, state in enumerate(all_states)}
    for axis, chain in zip(axes, chains):
        metadata = chain["permutation"]
        y = metadata["raw_order_low_to_high"].map(state_number)
        axis.step(metadata["source_draw"], y, where="post", linewidth=0.7)
        divergent = metadata.get("sample_stat__diverging", pd.Series(False, index=metadata.index)).astype(bool)
        axis.plot(metadata.loc[divergent, "source_draw"], y[divergent], "|", color="crimson", ms=6)
        axis.set_yticks(range(len(all_states)), all_states)
        axis.set_ylabel(chain["label"])
        axis.grid(axis="x", alpha=0.2)
    axes[-1].set_xlabel("Retained draw index")
    fig.suptitle("Raw component order from lowest to highest mean spin; red ticks are divergences")
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(samples: dict[str, np.ndarray], header: str, stem: str, output: Path) -> None:
    """Write ArviZ summary statistics for samples shaped (chain, draw)."""
    posterior = from_dict({"posterior": samples})
    with (output / f"{stem}.txt").open("w", encoding="utf-8") as stream:
        stream.write(header + "\n\n")
        stream.write(summary(posterior, ci_prob=0.9).to_string() + "\n")
    summary(posterior, ci_prob=0.9, round_to="none").to_csv(
        output / f"{stem}.csv", index_label="parameter"
    )


def write_multichain_summary(chains: list[dict], parameters: list[str], output: Path) -> None:
    min_draws = min(len(chain["posterior_relabelled"][parameters[0]]) for chain in chains)
    samples = {
        name: np.stack([chain["posterior_relabelled"][name][:min_draws] for chain in chains])
        for name in parameters
    }
    header = (
        f"Draw-wise mean-spin relabelled diagnostics: {len(chains)} chains, "
        f"{min_draws} draws per chain. Thinning was not applied."
    )
    write_summary(samples, header, "mcmc_summary_relabelled_unthinned", output)


def write_single_chain_summary(chain: dict, parameters: list[str], output: Path) -> None:
    """Write ArviZ diagnostics for one relabelled seed (R-hat is undefined for one chain)."""
    samples = {
        name: np.asarray(chain["posterior_relabelled"][name])[None, :]
        for name in parameters
    }
    header = (
        f"Draw-wise mean-spin relabelled single-chain diagnostics: "
        f"{chain['label']}, {len(next(iter(samples.values()))[0])} draws."
    )
    write_summary(samples, header, "mcmc_summary_relabelled", output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw-wise mean-spin relabeling, diagnostics, thinning, and chain combination."
    )
    parser.add_argument("--results-root", type=Path, default=Path("FinalResults"))
    parser.add_argument("--prefix", required=True, help="Folder prefix before -seedNN.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--exclude-seed", action="append", default=[])
    parser.add_argument(
        "--no-thin",
        action="store_true",
        help="Put every draw-wise relabelled sample in result.csv; still calculate IAT diagnostics.",
    )
    parser.add_argument(
        "--weighting", choices=("equal-chain", "effective"), default="equal-chain",
        help="Equal-chain trims thinned chains to equal counts; effective retains all thinned draws.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excluded = {value if value.startswith("seed") else f"seed{value}" for value in args.exclude_seed}
    folders = sorted(
        folder for folder in args.results_root.glob(f"{args.prefix}-seed*")
        if (folder / "result.csv").is_file()
        and (folder / "chain_samples.nc").is_file()
        and folder.name.rsplit("-", 1)[-1] not in excluded
    )
    if len(folders) < 2:
        raise ValueError("Need at least two completed seed folders with result.csv and chain_samples.nc")

    chains = []
    expected_components = None
    for folder in folders:
        samples, posterior, stats = read_chain(folder)
        order, scores, components = spin_order(posterior)
        if expected_components is None:
            expected_components = components
        elif components != expected_components:
            raise ValueError(f"{folder.name}: component labels differ between chains")
        relabelled_posterior = relabel_mapping(posterior, order, components)
        relabelled_samples = relabel_dataframe(samples, order, components)
        ordered_scores = np.column_stack([
            0.5 * (relabelled_posterior[f"mu_chi_1_{c}"] + relabelled_posterior[f"mu_chi_2_{c}"])
            for c in components
        ])
        if np.any(np.diff(ordered_scores, axis=1) < -1e-12):
            raise AssertionError(f"{folder.name}: draw-wise spin relabeling failed")
        chains.append({
            "folder": folder,
            "label": folder.name.rsplit("-", 1)[-1],
            "samples_relabelled": relabelled_samples,
            "posterior_raw": posterior,
            "posterior_relabelled": relabelled_posterior,
            "stats": stats,
            "permutation": permutation_metadata(order, scores, stats, components),
        })

    output_kind = "unthinned" if args.no_thin else "thinned"
    output = args.output or args.results_root / (
        f"{args.prefix}-combined-{len(chains)}-chains-drawwise-spin-ordered-{output_kind}"
    )
    output.mkdir(parents=True, exist_ok=True)
    for candidate in ("model_and_priors.py", "config.json"):
        source = folders[0] / candidate
        if source.exists():
            shutil.copy2(source, output / candidate)

    common_parameters = set(scalar_parameters(chains[0]["posterior_relabelled"], len(chains[0]["samples_relabelled"])))
    for chain in chains[1:]:
        common_parameters &= set(scalar_parameters(chain["posterior_relabelled"], len(chain["samples_relabelled"])))
    parameters = sorted(common_parameters)
    if not parameters:
        raise ValueError("No common nonconstant scalar posterior parameters")

    relabelled_iat = compute_iats(chains, "posterior_relabelled", parameters)
    raw_iat = compute_iats(chains, "posterior_raw", parameters)
    relabelled_iat.to_csv(output / "integrated_autocorrelation_times_relabelled.csv", index=False)
    raw_iat.to_csv(output / "integrated_autocorrelation_times_raw_labels.csv", index=False)

    worst_iat = relabelled_iat.groupby("seed")["iat"].max()
    missing_iat = worst_iat.index[~np.isfinite(worst_iat)].tolist()
    if missing_iat:
        raise ValueError(f"No finite IAT could be estimated for chains: {missing_iat}")
    thinning = {
        chain["label"]: 1 if args.no_thin else max(1, array_stats.thin_factor(
            np.stack([chain["posterior_relabelled"][name] for name in parameters]),
            reduce_func="min", chain_axis=None,
        ))
        for chain in chains
    }
    thinned_indices = {
        chain["label"]: np.arange(0, len(chain["samples_relabelled"]), thinning[chain["label"]])
        for chain in chains
    }
    if args.weighting == "equal-chain":
        target = min(len(indices) for indices in thinned_indices.values())
        for label, indices in thinned_indices.items():
            positions = np.linspace(0, len(indices) - 1, target).round().astype(int)
            thinned_indices[label] = indices[positions]

    unthinned_frames = []
    thinned_frames = []
    unthinned_provenance = []
    thinned_provenance = []
    permutation_frames = []
    island_occupancy_rows = []
    individual_root = output / "individual-spin-ordered"
    individual_root.mkdir(exist_ok=True)
    for chain_id, chain in enumerate(chains):
        label = chain["label"]
        metadata = chain["permutation"].copy()
        metadata.insert(0, "seed", label)
        metadata.insert(0, "chain_id", chain_id)
        permutation_frames.append(metadata)

        individual_output = individual_root / label
        individual_output.mkdir(exist_ok=True)
        chain["samples_relabelled"].to_csv(individual_output / "result.csv", index=False)
        metadata.to_csv(individual_output / "permutation_states.csv", index=False)
        for candidate in ("model_and_priors.py", "config.json"):
            source = chain["folder"] / candidate
            if source.exists():
                shutil.copy2(source, individual_output / candidate)
        source_chain_file = chain["folder"] / "chain_samples.nc"
        relabelled_chain_file = individual_output / "chain_samples.nc"
        shutil.copy2(source_chain_file, relabelled_chain_file)
        with h5py.File(relabelled_chain_file, "r+") as handle:
            for name, values in chain["posterior_relabelled"].items():
                handle["posterior"][name][0, :] = values
        (individual_output / "source_provenance.json").write_text(json.dumps({
            "source_folder": chain["folder"].name,
            "component_label_rule": "draw-wise increasing mean(mu_chi_1, mu_chi_2)",
            "draws": len(metadata),
        }, indent=2) + "\n")
        write_single_chain_summary(chain, parameters, individual_output)

        unthinned_frames.append(chain["samples_relabelled"])
        for draw in range(len(chain["samples_relabelled"])):
            unthinned_provenance.append({
                "chain_id": chain_id, "seed": label, "source_draw": draw,
                "raw_order_low_to_high": metadata.loc[draw, "raw_order_low_to_high"],
                "diverging": bool(chain["stats"].get("diverging", np.zeros(len(metadata)))[draw]),
            })

        indices = thinned_indices[label]
        thinned_frames.append(chain["samples_relabelled"].iloc[indices])
        selected_state_counts = Counter(metadata.loc[indices, "raw_order_low_to_high"])
        for state in sorted(selected_state_counts):
            count = selected_state_counts[state]
            island_occupancy_rows.append({
                "seed": label,
                "raw_label_island": state,
                "selected_draws": count,
                "percentage_of_selected_chain": 100.0 * count / len(indices),
                "total_selected_draws_for_chain": len(indices),
            })
        for draw in indices:
            thinned_provenance.append({
                "chain_id": chain_id, "seed": label, "source_draw": int(draw),
                "thin_interval": thinning[label],
                "raw_order_low_to_high": metadata.loc[draw, "raw_order_low_to_high"],
                "diverging": bool(chain["stats"].get("diverging", np.zeros(len(metadata)))[draw]),
            })

    pd.concat(unthinned_frames, ignore_index=True).to_csv(output / "result_relabelled_unthinned.csv", index=False)
    pd.DataFrame(unthinned_provenance).to_csv(output / "chain_index_relabelled_unthinned.csv", index=False)
    combined = pd.concat(thinned_frames, ignore_index=True)
    combined.to_csv(output / "result.csv", index=False)
    pd.DataFrame(thinned_provenance).to_csv(output / "chain_index.csv", index=False)
    pd.concat(permutation_frames, ignore_index=True).to_csv(output / "permutation_states.csv", index=False)
    occupancy = pd.DataFrame(island_occupancy_rows)
    occupancy.to_csv(output / "island_occupancy.csv", index=False)
    if not args.no_thin:
        occupancy.to_csv(output / "thinned_island_occupancy.csv", index=False)

    write_multichain_summary(chains, parameters, output)
    make_permutation_plot(chains, output / "raw-label-permutation-trace.png")

    chain_diagnostics = {}
    for chain in chains:
        label = chain["label"]
        metadata = chain["permutation"]
        stats = chain["stats"]
        ranked = (
            relabelled_iat.loc[relabelled_iat["seed"] == label]
            .sort_values("iat", ascending=False).head(5)
        )
        states = Counter(metadata["raw_order_low_to_high"])
        transition_gaps = metadata.loc[
            metadata["permutation_changed"], "minimum_adjacent_spin_gap"
        ]
        constant_parameters = sorted(
            parameter for parameter in parameters
            if np.ptp(chain["posterior_relabelled"][parameter]) <= 1e-12
        )
        selected_states = metadata.loc[thinned_indices[label], "raw_order_low_to_high"]
        selected_counts = Counter(selected_states)
        chain_diagnostics[label] = {
            "source_folder": chain["folder"].name,
            "draws": len(metadata),
            "divergences": int(np.asarray(stats.get("diverging", [])).sum()),
            "max_tree_depth_fraction": float(np.mean(stats["reached_max_tree_depth"]))
            if "reached_max_tree_depth" in stats else None,
            "bfmi": bfmi(stats.get("energy")),
            "constant_sampled_parameters": constant_parameters,
            "permutation_transition_count": int(metadata["permutation_changed"].sum()),
            "permutation_state_counts": dict(states),
            "thinned_permutation_state_percentages": {
                state: 100.0 * count / len(selected_states)
                for state, count in sorted(selected_counts.items())
            },
            "median_minimum_spin_gap_after_transition": float(transition_gaps.median())
            if len(transition_gaps) else None,
            "maximum_minimum_spin_gap_after_transition": float(transition_gaps.max())
            if len(transition_gaps) else None,
            "fraction_minimum_spin_gap_below_0p01": float(
                np.mean(metadata["minimum_adjacent_spin_gap"] < 0.01)
            ),
            "worst_five_relabelled_iats": [
                {"parameter": row.parameter, "iat": float(row.iat)}
                for row in ranked.itertuples()
            ],
            "thinning_interval": thinning[label],
            "retained_thinned_draws": len(thinned_indices[label]),
        }

    report = {
        "warning": (
            "R-hat/ESS are computed on unthinned draw-wise relabelled chains. "
            + (
                "result.csv also preserves every draw; no thinning was applied."
                if args.no_thin else
                "Thinning is only for a compact plotting table and does not repair bad sampling. "
                "No draw is selectively removed because it diverged; regular thinning may omit "
                "both divergent and non-divergent draws. The unthinned output preserves every draw."
            )
        ),
        "component_label_rule": (
            "At every draw, A/B/C are assigned by increasing "
            "0.5*(mu_chi_1 + mu_chi_2); complete component parameter vectors move together."
        ),
        "weighting": args.weighting,
        "thinning_enabled": not args.no_thin,
        "excluded_seeds": sorted(excluded),
        "source_folders": [folder.name for folder in folders],
        "iat_method": (
            "IAT = draws / min(ArviZ bulk ESS, ArviZ tail ESS at prob 0.95), per chain. "
            "Thinning interval from ArviZ thin_factor(reduce_func='min') on relabelled parameters."
        ),
        "combined_thinned_draws": len(combined),
        "combined_selected_draws": len(combined),
        "chain_diagnostics": chain_diagnostics,
    }
    (output / "diagnostic_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {output}")
    print(json.dumps({
        "combined_selected_draws": len(combined),
        "thinning_interval_by_chain": thinning,
        "permutation_transitions_by_chain": {
            label: values["permutation_transition_count"] for label, values in chain_diagnostics.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
