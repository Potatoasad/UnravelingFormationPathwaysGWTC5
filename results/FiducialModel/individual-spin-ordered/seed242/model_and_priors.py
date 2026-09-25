# =============================================================================
# Model-and-prior template for gravpop population analyses.
#
# Copy this file, rename it to match your run, then fill in the two sections
# marked CONFIGURE below. pop_analysis.py needs nothing else.
#
# Submit:
#   sbatch /path/to/bin/submit_run.sh /abs/path/to/this_file.py
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURE 1: Cluster infrastructure
# These are read by submit_run.sh via bootstrap_run.py WITHOUT executing this
# module, so they must be plain string/list literals (no expressions).
# -----------------------------------------------------------------------------
VENV_PATH    = "/mnt/home/ahussain/Explorations/o4b_multivariate_truncated_gaussians_in_chi_1_vs_chi_2/.venv"
MODULE_LOADS = []
REPO_DIR = "."
ANALYSIS_DIR     = "."
FIT_DIR = "/mnt/home/ahussain/Explorations/gravpop_analysis_template"

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from gravpop import *
import numpyro.distributions as dist

N_COMPONENTS = 3
N_REGIMES = 3
TILT_MIXTURE_OR_GAUSSIAN = "mix"
MIN_BREAK_MASS = 10
SIGMA_MASS_PEAK_LOW = 0.5

RUN_NAME = "Final-ThreeComponents-GWTC5-ThreeRegimes-mix-tilt-mmin10-sigpp_low0p5-unconstrained-labels-diagonal-all-mag-rho-fixed-ta090-depth8-1k"

# -----------------------------------------------------------------------------
# CONFIGURE 2: Run identity and optional sampler overrides
# -----------------------------------------------------------------------------
def get_run_config():
    return {
        "run_name":       RUN_NAME,
        ### CHANGE FOR O4A <--> O4B YOUR FILES WILL CHANGE
        #"observing_name": "O4a",
        #"event_data_file":    f"{FIT_DIR}/Fits_events/data_product/event_data.hdf5",
        #"selection_fit_file": f"{FIT_DIR}/Fits_selection/2026-03-30/selection_data_2026-03-31.h5",
        ### CHANGE FOR O4A <--> O4B YOUR FILES WILL CHANGE
        "observing_name": "O4b",
        "event_data_file":    f"{REPO_DIR}/Fits_events/event_data_2026-04-21.hdf5",
        "selection_fit_file":    f"{REPO_DIR}/Fits_selection/selection_data_2026-04-22_final_40_all_samples.h5",
        ####
        "results_base":       f"{ANALYSIS_DIR}/FinalResults",
        "num_samples":    1000,
        "num_warmup":     1000,
        "seed":           240,
        "use_forward_mode": False,
        "just_prior":     False,
        "max_tree_depth":     8,
        "target_accept_prob": 0.9,
        "mcmc_kwargs":    {'jit_model_args': False},
    }


# -----------------------------------------------------------------------------
# spin helper
# -----------------------------------------------------------------------------

### MIXTURE MODEL SETTINGS! #####
turn_off_rho_tilt = False
fixed_rho_chi = 1e-6
mixture_labels = ['a', 'b', 'c']
#mass_dependent_mixture = False

##############################

### Magnitude Models and Prior helpers
def make_2D_mag_model(position_name):
    mag = TruncatedGaussian2DAnalytic(
        var_names=['chi_1', 'chi_2'],
        hyper_var_names=[
            f'mu_chi_1_{position_name}',    f'sigma_chi_1_{position_name}',
            f'mu_chi_2_{position_name}',    f'sigma_chi_2_{position_name}',
            f'rho_chi_{position_name}',
        ],
        a=[0, 0], b=[1, 1],
    )
    return mag

def make_2D_mag_model_priors(position_name):
    prior = {
    f'mu_chi_1_{position_name}' : dist.Uniform(0,1),
    f'sigma_chi_1_{position_name}' 	: dist.Uniform(0,1),
    f'mu_chi_2_{position_name}' 		: dist.Uniform(0,1),
    f'sigma_chi_2_{position_name}' 	: dist.Uniform(0,1),
    }
    return prior


### Tilt Models and Prior helpers
def make_tilt_mixture_fraction_name(position_name):
    return f'zeta_cos_tilt_{position_name}'

def make_tilt_mixture_fraction_prior(position_name):
    return {make_tilt_mixture_fraction_name(position_name): dist.Uniform(0.0, 1.0)}
            
def make_2D_tilt_model(position_name):
    tilt0 = TruncatedGaussian2DAnalytic(
        var_names=['cos_tilt_1', 'cos_tilt_2'],
        hyper_var_names=[
            f'mu_cos_tilt_1_{position_name}', f'sigma_cos_tilt_1_{position_name}',
            f'mu_cos_tilt_2_{position_name}', f'sigma_cos_tilt_2_{position_name}',
            f'rho_cos_tilt_{position_name}',
        ],
        a=[-1, -1], b=[1, 1],
    )

    tilt1 = Uniform2DAnalytic(a=[-1, -1], b=[1, 1], var_names=['cos_tilt_1', 'cos_tilt_2'])
    zeta_name = make_tilt_mixture_fraction_name(position_name)
    return mixture([tilt0, tilt1], [zeta_name])


def make_2D_tilt_model_from_1D_product(position_name):
    tilt1_sel = TruncatedGaussian1DAnalytic(a=-1.0,b=1.0, 
                                     var_names=['cos_tilt_1'],
                                     hyper_var_names=[ f'mu_cos_tilt_1_{position_name}', 
                                                      f'sigma_cos_tilt_1_{position_name}',])
    tilt2_sel = TruncatedGaussian1DAnalytic(a=-1.0,b=1.0, 
                                     var_names=['cos_tilt_2'],
                                     hyper_var_names=[ f'mu_cos_tilt_2_{position_name}', 
                                                      f'sigma_cos_tilt_2_{position_name}',])
    model_tilt1_sel0 = product([tilt1_sel, tilt2_sel])
    model_tilt1_sel1 = Uniform2DAnalytic(a=[-1, -1], b=[1, 1], var_names=['cos_tilt_1', 'cos_tilt_2'])
    zeta_name = make_tilt_mixture_fraction_name(position_name)
    return mixture([model_tilt1_sel0, model_tilt1_sel1], [zeta_name])

def make_2D_tilt_model_priors(position_name, turn_off_rho_tilt=True):
    prior = {
    f'mu_cos_tilt_1_{position_name}' :  dist.Uniform(-1,1),
    f'sigma_cos_tilt_1_{position_name}' 	: dist.Uniform(0,2),
    f'mu_cos_tilt_2_{position_name}' 		: dist.Uniform(-1,1),
    f'sigma_cos_tilt_2_{position_name}' 	: dist.Uniform(0,2),
    **make_tilt_mixture_fraction_prior(position_name)
    }
    if not turn_off_rho_tilt:
        prior.update({f'rho_cos_tilt_{position_name}' : dist.Uniform(-0.9,0.9)})
    return prior


def make_mass_ratio_model(position_name):
    mass_ratio_model = TruncatedGaussian1D(a=0.01,
                                b=1.0,
                                var_names=['mass_ratio'],
                                hyper_var_names=[ f'mu_mass_ratio_{position_name}',
                                                  f'sigma_mass_ratio_{position_name}',])
    return mass_ratio_model

def make_mass_ratio_prior(position_name):
    mass_ratio_prior = { f'mu_mass_ratio_{position_name}' : dist.Uniform(0.4,1.2),
                         f'sigma_mass_ratio_{position_name}' : dist.Uniform(0.2, 0.8)}
    return mass_ratio_prior

### Full Spin Model helper
def _make_spin_models(position_name, turn_off_rho_tilt=turn_off_rho_tilt):
    """Return the list of component models for one sub-population."""
    position_name = position_name or str(int(position))
    mag = make_2D_mag_model(position_name)
    mag = fixed_parameters(mag, {f'rho_chi_{position_name}': fixed_rho_chi})
    tilt = make_2D_tilt_model(position_name)
    if turn_off_rho_tilt:
        tilt = fixed_parameters(tilt, {tilt.hyper_var_names[-1] : 1e-6})
        tilt_sel = make_2D_tilt_model_from_1D_product(position_name)
    else:
        tilt_sel = tilt
    return product([mag, tilt]),  product([mag, tilt_sel])

def _make_spin_models_with_mass_ratio(position_name, turn_off_rho_tilt=turn_off_rho_tilt):
    """Return the list of component models for one sub-population."""
    position_name = position_name or str(int(position))
    mag = make_2D_mag_model(position_name)
    mag = fixed_parameters(mag, {f'rho_chi_{position_name}': fixed_rho_chi})
    mass_ratio = make_mass_ratio_model(position_name)
    tilt = make_2D_tilt_model(position_name)
    if turn_off_rho_tilt:
        tilt = fixed_parameters(tilt, {tilt.hyper_var_names[-1] : 1e-6})
        tilt_sel = make_2D_tilt_model_from_1D_product(position_name)
    else:
        tilt_sel = tilt
    return product([mag, tilt, mass_ratio]),  product([mag, tilt_sel, mass_ratio])

### Fraction Model Helper ###
def mass_dependent_fraction_2d(data, params):
    f_start = params["eta_spin_f_start"]
    f_end = params["eta_spin_f_end"]
    mass_start = params["eta_spin_mass_start"]
    mass_end = params["eta_spin_mass_end"]

    m1 = jnp.asarray(data["mass_1_source"])
    clipped = jnp.clip(m1, mass_start, mass_end) - mass_start
    slope = (f_end - f_start) / (mass_end - mass_start)

    f = f_start + slope * clipped
    return jnp.stack([f, 1.0 - f], axis=-1) 


frac_model = make_piecewise_linear_mixing_fraction(
    var_names = ['mass_1_source'],
    n_components = N_COMPONENTS,
    n_flat = N_REGIMES,
    mass_cutoff_name='spin_fraction_break_mass',
    fraction_name='eta_spin',
    normalize=True,
)




# -----------------------------------------------------------------------------
# Population models
# -----------------------------------------------------------------------------
"""
def get_models():
    ### Mass Model
    non_smoothed_model = ThreeComponentPowerLaw(var_names=['mass_1_source'],  gaussian_mass_maximum=300);
    smoothed_mass_model = SmoothedPrimaryMassRatioSeparateMmin(primary_mass_model=non_smoothed_model, mass_ratio_name='mass_ratio', mmax_fixed=300, mmin_fixed=3)

    ### Spin Model

    spin_model_list = []
    spin_model_list_sel = []
    for position_name in mixture_labels:
        s,s_sel = _make_spin_models(position_name, turn_off_rho_tilt=turn_off_rho_tilt)
        spin_model_list.append(s)
        spin_model_list_sel.append(s_sel)

    frac_model = make_piecewise_linear_mixing_fraction(
                        var_names = ['mass_1_source'],
                        n_components = 4,
                        n_flat = 3,
                        mass_cutoff_name='spin_fraction_break_mass',
                        fraction_name='eta_spin',
                        normalize=True,
                    )
    spin_model_full = mixture(spin_model_list, frac_model)
    spin_model_full_sel = mixture(spin_model_list_sel, frac_model)

    ### Redshift Model
    redshift_model = PowerLawRedshift(var_names=['redshift'],hyper_var_names=['lamb'], z_max=3);

    model = [product([smoothed_mass_model, redshift_model, spin_model_full])]
    model_selection = [product([smoothed_mass_model, redshift_model, spin_model_full_sel])]

    return model, model_selection  # (event models, selection models)
"""

def get_models():
    ### Mass Model
    non_smoothed_model = ThreeComponentPowerLaw(var_names=['mass_1_source'],  gaussian_mass_maximum=300);
    #smoothed_mass_model = SmoothedPrimaryMassRatioSeparateMmin(primary_mass_model=non_smoothed_model, mass_ratio_name='mass_ratio', mmax_fixed=300, mmin_fixed=3)
    smoothed_mass_model = Smoothed1D(non_smoothed_model, mmin_fixed=3, mmax_fixed=300, normalization_shape=500)
    
    ### Spin Model

    spin_model_list = []
    spin_model_list_sel = []
    for position_name in mixture_labels:
        s,s_sel = _make_spin_models_with_mass_ratio(position_name, turn_off_rho_tilt=turn_off_rho_tilt)
        spin_model_list.append(s)
        spin_model_list_sel.append(s_sel)

    frac_model = make_piecewise_linear_mixing_fraction(
                        var_names = ['mass_1_source'],
                        n_components = N_COMPONENTS,
                        n_flat = N_REGIMES,
                        mass_cutoff_name='spin_fraction_break_mass',
                        fraction_name='eta_spin',
                        normalize=True,
                    )
    spin_model_full = mixture(spin_model_list, frac_model)
    spin_model_full_sel = mixture(spin_model_list_sel, frac_model)

    ### Redshift Model
    redshift_model = PowerLawRedshift(var_names=['redshift'],hyper_var_names=['lamb'], z_max=3);

    model = [product([smoothed_mass_model, redshift_model, spin_model_full])]
    model_selection = [product([smoothed_mass_model, redshift_model, spin_model_full_sel])]

    return model, model_selection  # (event models, selection models)

# -----------------------------------------------------------------------------
# Hyper-likelihood and priors — called by pop_analysis.py
# -----------------------------------------------------------------------------
def get_hyper_likelihood_and_priors(event_data, selection_data):
    ### EVENT DATA PREPROCESSING
    event_data['responsibilities'] = event_data['responsibilities_sampled']

    ### SELECTION DATA PREPROCESSING
    analysis_time   = selection_data.pop('analysis_time')
    total_generated = selection_data.pop('total_generated')
    total_detected  = selection_data.pop('total_detected')
    selection_data['responsibilities'] = selection_data['responsibilities_sampled']
    selection_data['prior'] = selection_data['prior'] * selection_data['spin_prior']

    selection_func = SelectionFunction(
        selection_data,
        analysis_time=analysis_time,
        total_generated=total_generated,
        total_detected=total_detected,
    )

    ### MODELS
    models, models_selection = get_models()

    ### PRIORS
    mass_redshift_priors = dict(
        alpha_1 = dist.Uniform(low = -4, high = 12),
        alpha_2 = dist.Uniform(low = -4, high = 12),
        #beta = dist.Uniform(low = -2, high = 7),
        #mlow = LowerTriangularUniform(var_names=['mlow_1', 'mlow_2'],
        #                              limits=[(3,5),(3,5)]),
        mmin = dist.Uniform(3,5),
        delta_m = dist.Uniform(low = 0, high = 10),
        delta_m_high = dist.Uniform(low = 0, high = 20),
        mmax = DiracDelta(300.0),#dist.Uniform(low = 299, high = 300),
        break_mass = dist.Uniform(low = 20, high = 50),
        mpp_1 = dist.Uniform(low = 5, high = 20),
        mpp_2 = dist.Uniform(low = 25, high = 60),
        sigpp_1 = dist.Uniform(low = SIGMA_MASS_PEAK_LOW, high = 10),
        sigpp_2 = dist.Uniform(low = 1, high = 10),
        lam = DirchletPrior(['lam_0', 'lam_1', 'lam_2']),
        lamb = dist.Uniform(low = -10, high = 10),
    )

    frac_model = models[0].models[-1].mixing_model
    
    fraction_priors = {}
    fraction_priors['breakmass_prior'] = ConditionalPriorTriangularND(
        var_names=frac_model.break_names[::-1],
        limits=[(MIN_BREAK_MASS, 200)] * len(frac_model.break_names),
        flat=True,
    )

    # Components remain exchangeable; labels are assigned only in post-processing.
    for r in range(frac_model.n_flat):
        fraction_priors[frac_model.fraction_name + f'_regime_{r}'] = DirchletPrior(
            var_names=[a[r] for a in frac_model.frac_names])

    ## Spin dimension priors
    spin_mag_priors = {}
    spin_tilt_priors = {}
    mass_ratio_priors = {}

    for position_name in mixture_labels:
        spin_mag_priors.update(make_2D_mag_model_priors(position_name))
        spin_tilt_priors.update(make_2D_tilt_model_priors(position_name, turn_off_rho_tilt=turn_off_rho_tilt))
        mass_ratio_priors.update(make_mass_ratio_prior(position_name))
        
    priors = {}
    priors.update(mass_redshift_priors)
    priors.update(mass_ratio_priors)
    priors.update(fraction_priors)
    priors.update(spin_mag_priors)
    priors.update(spin_tilt_priors)

    ### LIKELIHOOD
    HL = MarginalizedHybridLikelihood(
        event_data=event_data,
        selection_data=selection_func,
        models=models,
        models_selection=models_selection,
        fix_kernels_selection={},
        fix_kernels_events={},
        use_log_space=True,
        save_total_log_variance = True
    )

    return HL, priors
