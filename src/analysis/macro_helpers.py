from dataclasses import dataclass, field
from typing import Any, List, Dict
import os
from gravpop import *
from plot_helper import *
from .latex_processing import *

import numpy as np

SEED = 0
CI_level = 90


def print_ranges_precision(
    low,
    med,
    hi,
    precision=2,
    uncertainty_digits=None,
    use_sci=None,
):
    low = float(low)
    med = float(med)
    hi = float(hi)

    plus = hi - med
    minus = med - low

    abs_med = abs(med)

    # ----------------------------------------
    # Decide whether to use scientific notation
    # ----------------------------------------
    if use_sci is None:
        use_sci = (
            abs_med >= 1e3 or
            (abs_med > 0 and abs_med < 1e-2)
        )

    if use_sci:
        exponent = int(np.floor(np.log10(abs_med)))
        scale = 10.0 ** exponent

        med /= scale
        plus /= scale
        minus /= scale
    else:
        exponent = 0

    # ----------------------------------------
    # Formatter
    # ----------------------------------------
    if uncertainty_digits is not None:
        smallest_uncertainty = min(abs(plus), abs(minus))

        if smallest_uncertainty == 0:
            decimal_places = precision
        else:
            order = int(np.floor(np.log10(smallest_uncertainty)))

            # uncertainty_digits=1:
            #   54    -> nearest 10
            #   5.4   -> nearest 1
            #   0.54  -> nearest 0.1
            #
            # uncertainty_digits=2:
            #   54    -> nearest 1
            #   5.4   -> nearest 0.1
            #   0.54  -> nearest 0.01
            decimal_places = uncertainty_digits - 1 - order

        def fmt(x):
            if decimal_places >= 0:
                return f"{x:.{decimal_places}f}"
            else:
                return f"{round(x, decimal_places):.0f}"

    else:
        def fmt(x):
            return (
                f"{x:.{precision}f}"
                .rstrip("0")
                .rstrip(".")
            )

    thestring = (
        f"{fmt(med)}"
        f"^{{+{fmt(plus)}}}"
        f"_{{-{fmt(minus)}}}"
    )

    if use_sci:
        thestring += rf" (\times 10^{{{exponent}}})"

    return thestring

def format_range(x, precision=2, level=CI_level):
    return print_ranges_precision(*get_ranges(x, level=level), precision=precision)

def command(name, disp, run_name_latex):
    RUN_NAME_LATEX_safe = latex_safe_macro_name(run_name_latex)
    name_safe = latex_safe_macro_name(name)
    return f"\\newcommand{{\\{RUN_NAME_LATEX_safe}{name_safe}}}{{{disp}}}"

def make_macro_lines(df, params, outfile="macros.tex", default_precision=2, default_level=CI_level,
                     run_name_latex="DEFAULT"):
    lines = []

    for name, spec in params.items():
        # --- handle shorthand ---
        if isinstance(spec, str):
            col = spec
            precision = default_precision
            level = default_level
        else:
            col = spec["col"]
            precision = spec.get("digits", spec.get("precision", default_precision))
            level = spec.get("level", default_level)

        disp = format_range(df[col], precision=precision, level=level)
        lines.append(command(name, disp, run_name_latex=run_name_latex))

    return lines

def write_macro(lines, outfile):
    with open(outfile, "w") as f:
        f.write("\n".join(lines) + "\n")

    return lines

def extract_name(path):
    return re.search(r'\d{4}-\d{2}-\d{2}-(.+?)/?$', path).group(1).replace('-', '')


def compute_bayes_factors(post,
                          preprocess = None,
                          columns =['eta_spin_f_start','eta_spin_f_end'],
                          limits = [(0.5, 1), (0.0, 1.0)],
                          loc = (1.0, 1.0),
                          prior_value = 2.0,
                          in_favor = False,
                          bayes_factor_sample_size = 100,
                          subsampling = None, 
                          seed = None):
    if preprocess is not None:
        post = preprocess(post)
    from truncatedgaussianmixtures import fit_kde
    bootstrap_kde_estimate = np.zeros(bayes_factor_sample_size);
    bootstrap_log_BF_estimate = np.zeros(bayes_factor_sample_size);
    a = np.array([lim[0] for lim in limits]);
    b = np.array([lim[1] for lim in limits]);
    subsampling = subsampling or len(post)
    print(a, b, type(post[columns]))
    if seed is not None:
        seed_sequence = np.random.SeedSequence(seed)
        seeds = seed_sequence.generate_state(bayes_factor_sample_size)
    else:
        seeds = [None]*bayes_factor_sample_size
    
    for i in range(bayes_factor_sample_size):
        fraction_kde = fit_kde(post[columns].sample(subsampling, replace=True, random_state=seeds[i]), 
                               a=a, b=b);
        bootstrap_kde_estimate[i] = fraction_kde.pdf(np.array(loc)[None, :])[0]
        bootstrap_log_BF_estimate[i] = np.log10((bootstrap_kde_estimate[i]) / prior_value)
        if not in_favor:
            bootstrap_log_BF_estimate[i] = -bootstrap_log_BF_estimate[i]

    return bootstrap_log_BF_estimate


var = ["chi", "cos_tilt"]
label = ["a", "b", "c"]
hyper = ["mu", "sigma"]
bh = ["1", "2"]





COMMON_HYPERPARAM_MACROS = {
            "MuChi1A": "mu_chi_1_a",
            "MuChi2A": "mu_chi_2_a",
            "SigmaChi1A": "sigma_chi_1_a",
            "SigmaChi2A": "sigma_chi_2_a",
            "MuChi1B": "mu_chi_1_b",
            "MuChi2B": "mu_chi_2_b",
            "SigmaChi1B": "sigma_chi_1_b",
            "SigmaChi2B": "sigma_chi_2_b",
            "MuChi1C": "mu_chi_1_c",
            "MuChi2C": "mu_chi_2_c",
            "SigmaChi1C": "sigma_chi_1_c",
            "SigmaChi2C": "sigma_chi_2_c",
            
        }

INDEPENDENT_HYPERPARAM_MACROS = {
            "Fraction" : 'eta_spin',
            "FractionOneMinus" : 'one_minus_eta_spin'
            }

MASSDEPENDENT_HYPERPARAM_MACROS = {
            "MassStart" : {'col': 'eta_spin_mass_start', 'digits': 0},
            "MassEnd" : {'col': 'eta_spin_mass_end', 'digits': 0},
            "FractionLowMass" : 'eta_spin_f_start',
            "FractionHighMass" : 'eta_spin_f_end'
            }

COMMON_PPD_MACROS = {}

INDEPENDENT_PPD_MACROS = {}

MASSDEPENDENT_PPD_MACROS = {
                "Fraction" : 'eta_spin_var',
                "FractionOneMinus" : 'one_minus_eta_spin_var'
                }

COMMON_LOG_BF_CALCULATIONS = {
    "ZeroSpinSubpop" : dict(columns =['mu_chi_1_a', 'sigma_chi_1_a',
                                                  'mu_chi_2_a', 'sigma_chi_2_a'],
                          limits = [(0., 1.), (0., 1.), (0., 1.), (0., 1.)],
                          loc = (0.0, 0.0, 0.0, 0.0),
                          prior_value = 1.0,
                          in_favor = True)
}


@dataclass
class Run:
    folder : str
    mass_dependent : bool = True
    run_name_latex : str = None
    _seed : int = SEED
    CI_level : float = 90.0
    bayes_factor_sample_size : int = 1000
    oversampling : int = 1 ## sample `N=oversampling` posterior predictives for each hyper posterior sample
    precompute_fns : Any = field(default_factory=list)

    def __post_init__(self):
        self.model_file = os.path.join(self.folder, "model_and_priors.py");
        self.result_file = os.path.join(self.folder, "result.csv");
        self.run = load_run(self.model_file)
        self.result = pd.read_csv(self.result_file)
        self.result_extras = self.result.copy()
        self.result_bf_subsample = self.result_extras.sample(self.bayes_factor_sample_size, 
                                                             random_state=self.seed)
        self.likelihood = self.run['HL']
        self.models, self.models_selection = load_models(self.model_file)
        self.ppd = None
        
        self.macros_hyper = {}
        self.macros_ppds = {}
        self.macros_bfs = {}
        self.bayes_factor_calculations = {}

        # Fix the name for this run to be referenced in latex macros
        if not self.run_name_latex:
            self.run_name_latex = latex_safe_macro_name(extract_name(self.folder))
        else:
            self.run_name_latex = latex_safe_macro_name(self.run_name_latex)

        ### Hyper parameter macros
        self.macros_hyper.update(COMMON_HYPERPARAM_MACROS)

        if self.mass_dependent:
            self.macros_hyper.update(MASSDEPENDENT_HYPERPARAM_MACROS)
        else:
            if "one_minus_eta_spin" not in self.result_extras.columns:
                self.result_extras['one_minus_eta_spin'] = 1 - self.result_extras['eta_spin']
            if "one_minus_xi_spin" not in self.result_extras.columns:
                self.result_extras['one_minus_xi_spin'] = 1 - self.result_extras['xi_spin']
            self.macros_hyper.update(INDEPENDENT_HYPERPARAM_MACROS)

        ### PPD macros
        self.macros_ppds.update(COMMON_PPD_MACROS)
        if self.mass_dependent:
            self.macros_ppds.update(MASSDEPENDENT_PPD_MACROS)
        else:
            self.macros_ppds.update(INDEPENDENT_PPD_MACROS)

        ### BF Macros
        self.are_bayes_factors_computed = False
        self.bayes_factor_calculations.update(COMMON_LOG_BF_CALCULATIONS)
        if self.mass_dependent:
            self.bayes_factor_calculations.update(MASSDEPENDENT_LOG_BF_CALCULATIONS)
        else:
            self.bayes_factor_calculations.update(INDEPENDENT_LOG_BF_CALCULATIONS)

        self.precompute_columns()
                

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        """Setter: recomputes things dependent on seed."""
        if value != self._seed:
            self._seed = value
            self.compute_ppds(seed=value)
            self.result_bf_subsample = self.result_extras.sample(self.bayes_factor_sample_size, 
                                                                 random_state=value)
            self.compute_bfs()

    def precompute_columns(self):
        for func in self.precompute_fns:
            func(self.result_extras)

    def compute_ppds(self, seed=None, oversample=None):
        seed = seed or self.seed
        oversample = oversample or self.oversampling
        self.ppd = self.models[0].sample(self.result_extras, oversample=oversample, random_state=seed);

    def compute_bfs(self, seed=None):
        seed = seed or self.seed 
        for BF_name, BF_kwarg in self.bayes_factor_calculations.items():
            for_or_against = "For" if BF_kwarg['in_favor'] else "Against"
            final_BF_name = "logTenBF" + for_or_against + BF_name
            log10BF_values = compute_bayes_factors(post=self.result_bf_subsample, 
                                                   bayes_factor_sample_size=self.bayes_factor_sample_size,
                                                   seed=seed,
                                                   **BF_kwarg)
            self.result_bf_subsample[final_BF_name] = log10BF_values
            self.macros_bfs[final_BF_name] = final_BF_name
        self.are_bayes_factors_computed = True
        
    def output_macros(self, path=None, CI_level=90, default_digits=2):
        self.macro_path = path or os.path.join(self.folder, "macros.tex")
        self.CI_level = CI_level or self.CI_level
        self.default_digits = default_digits or self.default_digits

        if self.ppd is None:
            self.compute_ppds()

        if not self.are_bayes_factors_computed:
            self.compute_bfs()

        self.ppd_macro_lines = make_macro_lines(self.ppd, self.macros_ppds,
                                          default_precision=self.default_digits, 
                                          default_level=self.CI_level, 
                                          run_name_latex=self.run_name_latex)

        self.hyper_macro_lines = make_macro_lines(self.result_extras, self.macros_hyper, 
                                          default_precision=self.default_digits, 
                                          default_level=self.CI_level,
                                          run_name_latex=self.run_name_latex)

        self.BF_macro_lines = make_macro_lines(self.result_bf_subsample, self.macros_bfs, 
                                          default_precision=self.default_digits, 
                                          default_level=self.CI_level,
                                          run_name_latex=self.run_name_latex)

        self.all_macro_lines = self.hyper_macro_lines + self.ppd_macro_lines + self.BF_macro_lines
        macro_lines = write_macro(self.all_macro_lines, outfile=self.macro_path)
        print(f"Macros written to {self.macro_path}")
        return macro_lines