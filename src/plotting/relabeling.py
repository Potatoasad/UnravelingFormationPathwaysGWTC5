"""Physical component ordering/relabeling and PPD component-splitting.

A run's raw hyperparameter columns are suffixed with arbitrary letters (the
labels the sampler happened to explore in). These helpers reorder them by
posterior median spin magnitude so "a" is always the lowest-spin component,
"b" the next, etc. -- see references/scientific-findings.md in the
gravpop-model-architect skill for why this matters (label degeneracy).
"""
import numpy as np


def infer_spin_order(posteriors, component_letters):
    """Rank raw component letters by posterior median spin magnitude."""
    scores = {
        letter: np.median(
            0.5 * (posteriors[f"mu_chi_1_{letter}"] + posteriors[f"mu_chi_2_{letter}"])
        )
        for letter in component_letters
    }
    ordered_letters = tuple(sorted(component_letters, key=lambda letter: scores[letter]))
    return ordered_letters, scores


def relabel_component_hyperparameters(posteriors, ordered_letters, component_letters):
    """Rename raw-letter hyperparameter columns to the physically-ordered letters.

    Returns (relabeled_df, source_to_target_index) where source_to_target_index
    maps each raw component's positional index (0-based, in component_letters
    order) to its physically-ordered target index -- needed to reorder
    fraction-model outputs (e.g. w0s[:, :, source_to_target_index]) the same way.
    """
    source_to_target_letter = {
        source_letter: target_letter
        for source_letter, target_letter in zip(ordered_letters, component_letters)
    }
    source_to_target_index = {
        component_letters.index(source_letter): target_index
        for target_index, source_letter in enumerate(ordered_letters)
    }

    renamed_columns = {}
    for column in posteriors.columns:
        renamed_column = column
        for source_letter, target_letter in source_to_target_letter.items():
            if renamed_column.endswith(f"_{source_letter}"):
                renamed_column = f"{renamed_column[:-2]}_{target_letter}"
                break

        for source_index, target_index in source_to_target_index.items():
            component_token = f"eta_spin_comp{source_index}_"
            if component_token in renamed_column:
                renamed_column = renamed_column.replace(
                    component_token, f"eta_spin_comp{target_index}_"
                )
                break

        renamed_columns[column] = renamed_column

    return posteriors.rename(columns=renamed_columns), source_to_target_index


def ordered_source_indices_from_letters(ordered_letters, component_letters):
    return tuple(component_letters.index(letter) for letter in ordered_letters)


def get_component_ppd_samples(ppds, component_indices, indicator_column_template="chi_1_{}"):
    """Split a combined PPD DataFrame into one sub-DataFrame per component."""
    return [ppds[ppds[indicator_column_template.format(i)]] for i in component_indices]


def add_chi_eff_chi_p(samples):
    """Add standard BBH effective and precessing spin parameters to PPD samples."""
    derived_samples = []
    for sample in samples:
        sample = sample.copy()
        q = sample["mass_ratio"].to_numpy()
        chi_1 = sample["chi_1"].to_numpy()
        chi_2 = sample["chi_2"].to_numpy()
        cos_tilt_1 = sample["cos_tilt_1"].to_numpy()
        cos_tilt_2 = sample["cos_tilt_2"].to_numpy()
        sample["chi_eff"] = (chi_1 * cos_tilt_1 + q * chi_2 * cos_tilt_2) / (1.0 + q)
        chi_p_1 = chi_1 * np.sqrt(np.clip(1.0 - cos_tilt_1**2, 0.0, None))
        chi_p_2 = q * (4.0 * q + 3.0) / (4.0 + 3.0 * q) * chi_2 * np.sqrt(
            np.clip(1.0 - cos_tilt_2**2, 0.0, None)
        )
        sample["chi_p"] = np.maximum(chi_p_1, chi_p_2)
        derived_samples.append(sample)
    return derived_samples
