"""Assemble the LaTeX-macro-ready summary numbers written to macros.json.

Parameterized over component_letters throughout -- the old makeFourSubpopPlots.py
hardcoded labels = ["a", "b", "c"] here even for its 4-component runs, so
component "d"'s macros were silently never computed. Passing the real
component_letters fixes that as a natural side effect of sharing this module.
"""
import numpy as np

from .stats import hpd_array

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _closest_mass_index(m, m1s):
    return np.argmin(np.abs(np.asarray(m1s) - m))


def get_hpd_at_mass(m1, m1s, w0s, alpha=0.1):
    """w0s: (n_hyper, n_grid, n_components), physically ordered."""
    from src import print_ranges_precision

    mass_hpd = {}
    m1_index = _closest_mass_index(m1, m1s)
    for i in range(w0s.shape[-1]):
        eta_traces_i = w0s[:, m1_index, i]
        hpd_tuple = hpd_array(eta_traces_i, alpha=alpha, axis=0)
        mass_hpd[ALPHABET[i]] = print_ranges_precision(*hpd_tuple, uncertainty_digits=1, use_sci=False)
    return mass_hpd


def fractions_at_masses(m1s, w0s, masses=(10, 20, 35, 60), alpha=0.1):
    return {
        f"fraction_at_mass_{int(mass)}": get_hpd_at_mass(mass, m1s, w0s, alpha=alpha)
        for mass in masses
    }


def mass_breaks_and_ranges(df):
    """Formatted break-mass HPDs, plus the Flat/Transition regime range strings."""
    from src import print_ranges_precision

    break_point_col_names = sorted(
        (c for c in df.columns if "spin_fraction_break_mass_" in c),
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    if not break_point_col_names:
        return [], {"Flat": [], "Transition": []}

    mass_breaks = [
        print_ranges_precision(*hpd_array(df[col], alpha=0.1), uncertainty_digits=1, use_sci=False)
        for col in break_point_col_names
    ]

    flat_or_transition = ["Flat"] + ["Transition", "Flat"] * len(break_point_col_names)
    ranges = {"Flat": [], "Transition": []}

    ranges[flat_or_transition[0]].append(f"<{mass_breaks[0]}")
    for i in range(1, len(break_point_col_names)):
        ranges[flat_or_transition[i]].append(f"{mass_breaks[i - 1]} - {mass_breaks[i]}")
    ranges[flat_or_transition[len(break_point_col_names)]].append(f">{mass_breaks[-1]}")

    return mass_breaks, ranges


def subpop_hyperposteriors(df, component_letters):
    """Per-component HPD summaries of chi/cos_tilt mu/sigma/rho and mass_ratio mu/sigma."""
    from src import print_ranges_precision

    variables = ["chi", "cos_tilt"]
    hypers = ["mu", "sigma"]
    bodies = ["1", "2"]

    result = {letter: {hyper: {} for hyper in hypers} for letter in component_letters}

    for letter in component_letters:
        for var in variables:
            for hyper in hypers:
                for body in bodies:
                    key = f"{hyper}_{var}_{body}_{letter}"
                    if key not in df.columns:
                        continue
                    result[letter][hyper][f"{var}_{body}"] = print_ranges_precision(
                        *hpd_array(df[key].values, alpha=0.1), uncertainty_digits=1, use_sci=False
                    )

        for var in variables:
            key = f"rho_{var}_{letter}"
            if key not in df.columns:
                continue
            result[letter]["rho"] = print_ranges_precision(
                *hpd_array(df[key].values, alpha=0.1), uncertainty_digits=1, use_sci=False
            )

        for var in ["mass_ratio"]:
            key = f"mu_{var}_{letter}"
            if key not in df.columns:
                continue
            result[letter].setdefault("mu", {})[var] = print_ranges_precision(
                *hpd_array(df[key].values, alpha=0.1), uncertainty_digits=1, use_sci=False
            )
            key = f"sigma_{var}_{letter}"
            result[letter].setdefault("sigma", {})[var] = print_ranges_precision(
                *hpd_array(df[key].values, alpha=0.1), uncertainty_digits=1, use_sci=False
            )

    return result


def subpop_ppds(ppds, component_letters, ordered_source_indices):
    """Per-component HPD summaries of chi/cos_tilt and mass_ratio drawn from the PPD."""
    from src import print_ranges_precision

    variables = ["chi", "cos_tilt"]
    bodies = ["1", "2"]
    subpop_ppd_dfs = {
        letter: ppds[ppds[f"chi_1_{index}"]]
        for letter, index in zip(component_letters, ordered_source_indices)
    }

    result = {letter: {} for letter in component_letters}
    for letter in component_letters:
        for var in variables:
            for body in bodies:
                key = f"{var}_{body}"
                result[letter][key] = print_ranges_precision(
                    *hpd_array(subpop_ppd_dfs[letter][key].values, alpha=0.1), uncertainty_digits=1, use_sci=False
                )
        for var in ["mass_ratio"]:
            result[letter][var] = print_ranges_precision(
                *hpd_array(subpop_ppd_dfs[letter][var].values, alpha=0.1), uncertainty_digits=1, use_sci=False
            )
    return result


def low_spin_bayes_factor(df, low_spin_letter, bayes_factor_sample_size=100):
    """log10 Bayes factor in favor of the low-spin component's spin being ~zero.

    Uses truncatedgaussianmixtures (juliacall/PythonCall.jl); requires
    JULIA_DEPOT_PATH to include a depot with the real PythonCall.jl package
    (a project-local depot with only stale compiled/ cache and no packages/
    will fail with "PythonCall.jl did not start properly" -- point the second
    JULIA_DEPOT_PATH entry at $HOME/.julia, not a project-local .julia dir).
    """
    from src import compute_bayes_factors, print_ranges_precision

    try:
        log_bfs = compute_bayes_factors(
            df, preprocess=None,
            columns=[f"mu_chi_1_{low_spin_letter}", f"mu_chi_2_{low_spin_letter}",
                     f"sigma_chi_1_{low_spin_letter}", f"sigma_chi_2_{low_spin_letter}"],
            limits=[(0.0, 1.0)] * 4, loc=(0.0, 0.0, 0.0, 0.0), prior_value=1.0, in_favor=True,
            bayes_factor_sample_size=bayes_factor_sample_size, subsampling=None, seed=None,
        )
        return print_ranges_precision(*hpd_array(log_bfs, alpha=0.1), uncertainty_digits=1, use_sci=False)
    except Exception as exc:
        print(f"Skipping low-spin Bayes-factor macro: {exc}")
        return "unavailable"


def assemble_all_macros(df, ppds, m1s, w0s, component_letters, ordered_source_indices, low_spin_letter):
    """Build the full ALL_MACROS dict written to macros.json."""
    mass_breaks, ranges = mass_breaks_and_ranges(df)
    return {
        "mass_breakpoints": mass_breaks,
        "fraction_at_masses": fractions_at_masses(m1s, w0s),
        "mass_ranges": ranges,
        "subpop_hyperposteriors": subpop_hyperposteriors(df, component_letters),
        "subpop_ppds": subpop_ppds(ppds, component_letters, ordered_source_indices),
        "log_bayes_factors": {
            "in_favor_of_component_a_being_zero_spin": low_spin_bayes_factor(df, low_spin_letter),
        },
    }
