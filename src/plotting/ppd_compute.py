"""Expensive PPD-computation stages: everything that resamples from
spin_model/mass_model or evaluates them over a grid for every hyperposterior
draw. Nothing in this module touches matplotlib -- these functions return
plain numpy arrays / pandas DataFrames that the compute script saves as CSV
and the plot script (src/plotting/figures.py) later reads back in.

Moved out of the old makeThreeSubpopPlots.py / makeFourSubpopPlots.py
monoliths. Previously component_derived_spin_ppd_draws was computed twice per
run (once directly, once again inside the composite plot) -- callers here
should compute each band once and pass the resulting DataFrame to every plot
function that needs it.
"""
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from gravpop.models.utils import coerce_random_state, rng_choice, topological_sort
from gravpop import product

from .stats import hpd_array

# -----------------------------------------------------------------------------
# PPD sampling (ppds = full_model.sample(df_raw, oversample=...))
# -----------------------------------------------------------------------------


def _sample_conditional_mixture_aligned(model, hyper_samples, rng):
    """Sample mixture components that produce different auxiliary columns."""
    weights = np.asarray(model.mixing_model.weights_normalized(hyper_samples, hyper_samples))
    component_index = np.asarray([
        rng_choice(rng, len(model.component_models), p=weights[row])
        for row in range(len(hyper_samples))
    ])
    component_frames = []
    all_columns = []
    for models in model.component_models:
        accumulated = hyper_samples.copy()
        new_columns = []
        for component_model in topological_sort(models):
            sampled = component_model.sample(
                accumulated, oversample=1, random_state=rng
            ).reset_index(drop=True)
            for column in sampled.columns:
                accumulated[column] = sampled[column].values
                if column not in new_columns:
                    new_columns.append(column)
                if column not in all_columns:
                    all_columns.append(column)
        component_frames.append(accumulated[new_columns].reset_index(drop=True))

    result_columns = {}
    for column in all_columns:
        present = [frame[column] for frame in component_frames if column in frame]
        is_boolean = present and all(series.dtype == bool for series in present)
        values = (
            np.zeros(len(hyper_samples), dtype=bool)
            if is_boolean
            else np.full(len(hyper_samples), np.nan, dtype=float)
        )
        for index, frame in enumerate(component_frames):
            if column not in frame:
                continue
            mask = component_index == index
            values[mask] = frame.loc[mask, column].to_numpy()
        result_columns[column] = values
    result = pd.DataFrame(result_columns)
    for index, column in enumerate(model._component_label_cols()):
        result[column] = component_index == index
        result[f"{column}_var"] = weights[:, index]
    return result


def _sample_full_model_with_aligned_components(model, spin_model, hyper_samples, oversample, random_state):
    """Product-model sampler with an aligned conditional-mixture step.

    Fallback for models whose spin_model submodel raises
    "equal len keys and value" under the default .sample(...) path.
    """
    rng = coerce_random_state(random_state)
    accumulated = pd.concat([hyper_samples] * oversample, ignore_index=True)
    new_columns = []
    for submodel in topological_sort(model.models):
        if submodel is spin_model:
            sampled = _sample_conditional_mixture_aligned(submodel, accumulated, rng)
        else:
            sampled = submodel.sample(
                accumulated, oversample=1, random_state=rng
            ).reset_index(drop=True)
        for column in sampled.columns:
            accumulated[column] = sampled[column].values
            if column not in new_columns:
                new_columns.append(column)
    return accumulated[new_columns]


def sample_ppds(full_model, spin_model, df_raw, oversample, seed):
    """ppds = full_model.sample(df_raw, oversample=...), with aligned-component fallback."""
    try:
        return full_model.sample(df_raw, oversample=oversample, random_state=seed)
    except ValueError as error:
        if "equal len keys and value" not in str(error):
            raise
        return _sample_full_model_with_aligned_components(
            full_model, spin_model, df_raw, oversample, seed
        )


# -----------------------------------------------------------------------------
# Fraction (eta) vs mass_1
# -----------------------------------------------------------------------------


def evaluate_fraction_ppd(frac_model, df_raw, m1_grid):
    """w0s[hyper_draw, mass_grid_point, component] = P(component | mass, draw)."""
    data = {"mass_1_source": jnp.asarray(m1_grid)}
    get_fraction_ppd = jax.vmap(lambda x: frac_model.weights_normalized(data, x))
    params = {k: jnp.array(df_raw[k].values) for k in df_raw.columns}
    return np.asarray(get_fraction_ppd(params))


def fraction_bands_to_frame(m1_grid, w0s, letters, alpha=0.1):
    """w0s (n_hyper, n_grid, n_components) -> tidy DataFrame of HPD bands.

    Columns: mass_1_source, then <letter>_lower/_median/_upper per component,
    in the same order as `letters`.
    """
    frame = {"mass_1_source": np.asarray(m1_grid)}
    for component_index, letter in enumerate(letters):
        lower, median, upper = hpd_array(w0s[:, :, component_index], alpha=alpha, axis=0)
        frame[f"{letter}_lower"] = lower
        frame[f"{letter}_median"] = median
        frame[f"{letter}_upper"] = upper
    return pd.DataFrame(frame)


# -----------------------------------------------------------------------------
# Component-conditioned chi_eff / chi_p PPD bands
# -----------------------------------------------------------------------------


def component_derived_spin_ppd_draws(
    spin_model, posteriors, source_indices, seed, samples_per_hyperposterior=5000,
):
    """Evaluate conditional chi_eff/chi_p PPD densities on fixed 1D grids.

    Each hyperposterior draw's density is a raw histogram over
    samples_per_hyperposterior Monte Carlo samples spread over 200 bins, so
    samples_per_hyperposterior should be large enough that broad/flat
    components (e.g. high spin) still get enough samples per bin to avoid
    visible box/staircase artifacts (generating more samples, not smoothing
    bins, is the fix for that -- see conversation history / CLAUDE notes).

    Returns (grids, density_draws): grids maps name -> bin-edge array, and
    density_draws maps name -> list of (n_hyper, n_bins) arrays, one per
    requested component, in the same order as source_indices.
    """
    grids = {
        "chi_eff": np.linspace(-1.0, 1.0, 201),
        "chi_p": np.linspace(0.0, 1.0, 201),
    }
    density_draws = {name: [] for name in grids}
    n_hyper = len(posteriors)

    for source_index in source_indices:
        component_model = product(spin_model.component_models[source_index])
        samples = component_model.sample(
            posteriors, oversample=samples_per_hyperposterior, random_state=seed + source_index
        )
        q = samples["mass_ratio"].to_numpy().reshape(samples_per_hyperposterior, n_hyper).T
        chi_1 = samples["chi_1"].to_numpy().reshape(samples_per_hyperposterior, n_hyper).T
        chi_2 = samples["chi_2"].to_numpy().reshape(samples_per_hyperposterior, n_hyper).T
        cos_tilt_1 = samples["cos_tilt_1"].to_numpy().reshape(samples_per_hyperposterior, n_hyper).T
        cos_tilt_2 = samples["cos_tilt_2"].to_numpy().reshape(samples_per_hyperposterior, n_hyper).T
        values = {
            "chi_eff": (chi_1 * cos_tilt_1 + q * chi_2 * cos_tilt_2) / (1.0 + q),
            "chi_p": np.maximum(
                chi_1 * np.sqrt(np.clip(1.0 - cos_tilt_1**2, 0.0, None)),
                q * (4.0 * q + 3.0) / (4.0 + 3.0 * q) * chi_2
                * np.sqrt(np.clip(1.0 - cos_tilt_2**2, 0.0, None)),
            ),
        }
        for name, grid in grids.items():
            bin_width = grid[1] - grid[0]
            bin_index = np.floor((values[name] - grid[0]) / bin_width).astype(int)
            bin_index = np.clip(bin_index, 0, len(grid) - 2)
            counts = np.zeros((n_hyper, len(grid) - 1), dtype=float)
            np.add.at(
                counts,
                (np.repeat(np.arange(n_hyper), samples_per_hyperposterior), bin_index.ravel()),
                1,
            )
            density_draws[name].append(counts / (samples_per_hyperposterior * bin_width))

    return grids, density_draws


def component_derived_spin_ppd_bands_frame(grids, density_draws, letters):
    """Reduce component_derived_spin_ppd_draws output to a tidy bands DataFrame.

    One row per (grid_name, bin_center); columns <letter>_lower/_median/_upper.
    """
    rows = []
    for name, grid in grids.items():
        centers = 0.5 * (grid[:-1] + grid[1:])
        columns = {"grid_name": name, "bin_center": centers}
        for letter, draws in zip(letters, density_draws[name]):
            lower, median, upper = np.quantile(draws, [0.05, 0.5, 0.95], axis=0)
            columns[f"{letter}_lower"] = lower
            columns[f"{letter}_median"] = median
            columns[f"{letter}_upper"] = upper
        rows.append(pd.DataFrame(columns))
    return pd.concat(rows, ignore_index=True)


# -----------------------------------------------------------------------------
# Mass-ratio PPD (closed-form truncnorm; cheap, but centralized for uniformity)
# -----------------------------------------------------------------------------


def mass_ratio_ppd_bands_frame(posteriors, letters, q_grid=None):
    from scipy.stats import truncnorm

    q_grid = np.linspace(0.01, 1.0, 400) if q_grid is None else np.asarray(q_grid)
    columns = {"q": q_grid}
    for letter in letters:
        mu = posteriors[f"mu_mass_ratio_{letter}"].to_numpy()[:, None]
        sigma = posteriors[f"sigma_mass_ratio_{letter}"].to_numpy()[:, None]
        density_draws = truncnorm.pdf(
            q_grid[None, :], (0.01 - mu) / sigma, (1.0 - mu) / sigma, loc=mu, scale=sigma,
        )
        lower, median, upper = np.quantile(density_draws, [0.05, 0.5, 0.95], axis=0)
        columns[f"{letter}_lower"] = lower
        columns[f"{letter}_median"] = median
        columns[f"{letter}_upper"] = upper
    return pd.DataFrame(columns)


# -----------------------------------------------------------------------------
# Mass-contribution PPD (p(m1, component) and p(m1))
# -----------------------------------------------------------------------------


def component_mass_contribution_density_draws(mass_model, mixing_model, posteriors, mass_grid):
    """Evaluate p(m_1, component) and p(m_1) for every hyperposterior draw."""
    required_parameters = set(mass_model.hyper_var_names) | set(mixing_model.hyper_var_names)
    missing_parameters = required_parameters - set(posteriors.columns)
    if missing_parameters:
        raise KeyError(
            "Missing hyperparameters for component-conditioned mass PPD: "
            + ", ".join(sorted(missing_parameters))
        )

    posterior_params = {name: jnp.asarray(posteriors[name].to_numpy()) for name in required_parameters}
    mass_data = {"mass_1_source": jnp.asarray(mass_grid)}
    mass_density = jax.vmap(lambda params: mass_model(mass_data, params))(posterior_params)
    component_weights = jax.vmap(lambda params: mixing_model.weights_normalized(mass_data, params))(posterior_params)
    joint_density = mass_density[..., None] * component_weights
    return np.asarray(joint_density), np.asarray(mass_density)


def mass_contribution_bands_frame(
    mass_model, mixing_model, posteriors, letters, mass_limits=(3.0, 300.0), n_grid=500,
):
    mass_grid = np.geomspace(*mass_limits, n_grid)
    density_draws, total_mass_density_draws = component_mass_contribution_density_draws(
        mass_model, mixing_model, posteriors, mass_grid,
    )
    columns = {"mass_1_source": mass_grid}
    for component_index, letter in enumerate(letters):
        lower, median, upper = np.quantile(
            density_draws[:, :, component_index], [0.05, 0.5, 0.95], axis=0
        )
        columns[f"{letter}_lower"] = lower
        columns[f"{letter}_median"] = median
        columns[f"{letter}_upper"] = upper
    total_lower, total_median, total_upper = np.quantile(
        total_mass_density_draws, [0.05, 0.5, 0.95], axis=0
    )
    columns["total_lower"] = total_lower
    columns["total_median"] = total_median
    columns["total_upper"] = total_upper
    return pd.DataFrame(columns)


# -----------------------------------------------------------------------------
# High-spin cos(tilt) marginal bands
# -----------------------------------------------------------------------------


def tilt_marginal_posterior_band(posteriors, component_letter, tilt_index, x_grid, y_grid):
    from gravpop import compute_truncated_normal_pdf_on_samples

    gaussian_columns = [
        f"mu_cos_tilt_1_{component_letter}",
        f"mu_cos_tilt_2_{component_letter}",
        f"sigma_cos_tilt_1_{component_letter}",
        f"sigma_cos_tilt_2_{component_letter}",
    ]
    if any(column not in posteriors.columns for column in gaussian_columns):
        uniform_marginal = np.full(len(x_grid), 0.5)
        return uniform_marginal, uniform_marginal, uniform_marginal

    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid, indexing="ij")
    xy_points_1 = x_mesh.ravel()
    xy_points_2 = y_mesh.ravel()
    uniform_density_2d = 0.25
    density_draws = np.empty((len(posteriors), len(x_grid)))

    rho_column = f"rho_cos_tilt_{component_letter}"
    zeta_column = f"zeta_cos_tilt_{component_letter}"

    for draw_index, (_, row) in enumerate(posteriors.iterrows()):
        rho = row[rho_column] if rho_column in posteriors.columns else 1e-6
        gaussian_density_2d = np.asarray(
            compute_truncated_normal_pdf_on_samples(
                x_1=jnp.asarray(xy_points_1),
                x_2=jnp.asarray(xy_points_2),
                mu_1=row[f"mu_cos_tilt_1_{component_letter}"],
                mu_2=row[f"mu_cos_tilt_2_{component_letter}"],
                sigma_1=max(row[f"sigma_cos_tilt_1_{component_letter}"], 1e-6),
                sigma_2=max(row[f"sigma_cos_tilt_2_{component_letter}"], 1e-6),
                rho=rho,
                a=[-1.0, -1.0],
                b=[1.0, 1.0],
            )
        ).reshape(len(x_grid), len(y_grid))

        if zeta_column in posteriors.columns:
            zeta = row[zeta_column]
            total_density_2d = zeta * gaussian_density_2d + (1.0 - zeta) * uniform_density_2d
        else:
            total_density_2d = gaussian_density_2d

        if tilt_index == 1:
            density_draws[draw_index] = np.trapezoid(total_density_2d, y_grid, axis=1)
        else:
            density_draws[draw_index] = np.trapezoid(total_density_2d, x_grid, axis=0)

    lower, median, upper = np.quantile(density_draws, [0.05, 0.5, 0.95], axis=0)
    return lower, median, upper


def high_spin_tilt_bands_frame(posteriors, high_spin_letter, tilt_grid=None):
    tilt_grid = np.linspace(-1.0, 1.0, 200) if tilt_grid is None else np.asarray(tilt_grid)
    lower_1, median_1, upper_1 = tilt_marginal_posterior_band(
        posteriors, component_letter=high_spin_letter, tilt_index=1, x_grid=tilt_grid, y_grid=tilt_grid
    )
    lower_2, median_2, upper_2 = tilt_marginal_posterior_band(
        posteriors, component_letter=high_spin_letter, tilt_index=2, x_grid=tilt_grid, y_grid=tilt_grid
    )
    return pd.DataFrame({
        "cos_tilt": tilt_grid,
        "tilt1_lower": lower_1, "tilt1_median": median_1, "tilt1_upper": upper_1,
        "tilt2_lower": lower_2, "tilt2_median": median_2, "tilt2_upper": upper_2,
    })
