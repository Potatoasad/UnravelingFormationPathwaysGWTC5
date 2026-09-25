"""Vectorized HPD (highest-posterior-density) interval helpers.

Previously duplicated three times (src/analysis/macro_helpers.py used its own
copy implicitly via import, makeplotsThreeComponents.py had one at module
level, and eta_vs_m1_plot had a third private copy nested inside itself).
"""
import numpy as np


def calc_min_interval_sorted_array(x, alpha=0.1, axis=-1):
    """HPD/minimum-width interval along `axis`, assuming x is already sorted."""
    x = np.asarray(x)

    n = x.shape[axis]
    cred_mass = 1.0 - alpha

    interval_idx_inc = int(np.floor(cred_mass * n))
    if interval_idx_inc < 1:
        raise ValueError("Too few elements for interval calculation")

    n_intervals = n - interval_idx_inc
    if n_intervals <= 0:
        raise ValueError("Too few elements for interval calculation")

    xs = np.moveaxis(x, axis, -1)
    interval_width = xs[..., interval_idx_inc:] - xs[..., :n_intervals]
    min_idx = np.argmin(interval_width, axis=-1)

    low = np.take_along_axis(xs, min_idx[..., None], axis=-1)[..., 0]
    high = np.take_along_axis(xs, (min_idx + interval_idx_inc)[..., None], axis=-1)[..., 0]
    return low, high


def hpd_array(x, alpha=0.1, axis=None, median=np.median):
    """Vectorized HPD interval. Returns (low, med, high)."""
    x = np.asarray(x)

    if axis is None:
        xs = np.sort(x.ravel())
        low, high = calc_min_interval_sorted_array(xs, alpha=alpha, axis=-1)
        med = median(xs)
        return low, med, high

    axis = int(axis)
    if axis < 0:
        axis += x.ndim
    if axis < 0 or axis >= x.ndim:
        raise np.AxisError(axis, ndim=x.ndim)

    xs = np.sort(x, axis=axis)
    low, high = calc_min_interval_sorted_array(xs, alpha=alpha, axis=axis)
    med = median(x, axis=axis)
    return low, med, high
