#!/usr/bin/env python
"""Create a paginated, ESS-sorted ArviZ trace PDF for one saved run."""

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_folder", type=Path)
    args = parser.parse_args()

    folder = args.result_folder.resolve()
    chain_file = folder / "chain_samples.nc"
    if not chain_file.is_file():
        raise FileNotFoundError(chain_file)

    data = az.from_netcdf(chain_file)
    posterior = data.posterior.dataset
    ess = az.ess(posterior, method="bulk")
    ordered = sorted(
        posterior.data_vars,
        key=lambda name: float(np.asarray(ess[name]).reshape(-1)[0]),
    )

    output = folder / "traceplot-all-hyperparameters.pdf"
    with PdfPages(output) as pdf:
        for name in ordered:
            plot = az.plot_trace(data, var_names=[name], backend="matplotlib")
            figure = plot.get_viz("figure")
            figure.set_size_inches(15, 5.5, forward=True)
            figure.suptitle(
                f"{name} | bulk ESS = {float(np.asarray(ess[name]).reshape(-1)[0]):.1f}"
                " | black rugs mark divergences",
                y=0.99,
                fontsize=12,
            )
            figure.subplots_adjust(top=0.84)
            pdf.savefig(figure)
            plt.close(figure)

    print(output)


if __name__ == "__main__":
    main()
