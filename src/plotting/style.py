"""Shared figure-style constants and cosmetic helpers.

Split out of the old makeThreeSubpopPlots.py / makeplotsThreeComponents.py
monoliths so both the compute and plot scripts (and any N-component variant)
share exactly one definition.
"""
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba

AASTEX_TEXT_WIDTH_IN = 7.1
AASTEX_COLUMN_SEP_IN = 0.3125
APJ_HALF_COLUMN_WIDTH_IN = (AASTEX_TEXT_WIDTH_IN - AASTEX_COLUMN_SEP_IN) / 2

LATEX_FONT_SIZES_PT = {
    "tiny": 5.0,
    "scriptsize": 7.0,
    "footnotesize": 8.0,
    "small": 9.0,
    "normalsize": 10.0,
    "large": 12.0,
    "Large": 14.4,
    "LARGE": 17.28,
    "huge": 20.74,
    "Huge": 24.88,
}

# Descriptive spin-role labels per established house style (see
# CODEX_HANDOFF/PLOTTING_REQUIREMENTS.md): three components are
# Low/Medium/High Spin; a fourth is "Higher Spin".
_SPIN_ROLE_NAMES = ["Low Spin", "Medium Spin", "High Spin", "Higher Spin"]


def component_letters(n_components):
    """('a','b','c'[,'d']) for n_components in {3,4}."""
    return tuple("abcd"[:n_components])


def default_component_colors(n_components):
    return sns.color_palette("colorblind", n_components)


def default_component_labels(n_components):
    """['A : Low Spin', 'B : Medium Spin', 'C : High Spin', ...]."""
    letters = component_letters(n_components)
    if n_components > len(_SPIN_ROLE_NAMES):
        raise ValueError(f"No default spin-role names for {n_components} components")
    return [
        f"{letter.upper()} : {role}"
        for letter, role in zip(letters, _SPIN_ROLE_NAMES[:n_components])
    ]


def apply_matplotlib_rcparams():
    """House LaTeX-matched font sizes; call once at script start."""
    import matplotlib.pyplot as plt

    plt.rcParams["font.size"] = LATEX_FONT_SIZES_PT["normalsize"]
    plt.rcParams["axes.labelsize"] = LATEX_FONT_SIZES_PT["normalsize"]
    plt.rcParams["xtick.labelsize"] = LATEX_FONT_SIZES_PT["normalsize"]
    plt.rcParams["ytick.labelsize"] = LATEX_FONT_SIZES_PT["normalsize"]
    plt.rcParams["legend.fontsize"] = LATEX_FONT_SIZES_PT["normalsize"]
    plt.rcParams["axes.titlesize"] = LATEX_FONT_SIZES_PT["normalsize"]


def format_figure_ticks(fig):
    """Limit every visible axis label in a figure to one decimal place."""
    for ax in fig.axes:
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    return fig


def fix_legend(fig, ax=None, model_labels=None, legend_location="upper left"):
    """Redraw a plot_helper KDE-contour legend as translucent-fill patches."""
    if ax is None:
        ax = fig.axes[0]
    leg = ax.get_legend()

    labels = [t.get_text() for t in leg.get_texts()]
    colors = [
        h.get_facecolor() if hasattr(h, "get_facecolor") else h.get_color()
        for h in leg.legend_handles
    ]

    handles = [
        Patch(facecolor=to_rgba(c, 0.7), edgecolor=c, linewidth=1.5)
        for c in colors
    ]

    leg = ax.legend(
        handles,
        labels,
        loc=legend_location,
        handlelength=0.7,
        handleheight=0.7,
        handletextpad=0.45,
        labelspacing=0.25,
        frameon=True,
    )

    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.7)
    frame.set_edgecolor("none")

    return fig


def keep_if_in_df(df, hyper_var_names):
    """Filter a hyperparameter-name list against df's actual columns."""
    return [var for var in hyper_var_names if var in df.columns]
