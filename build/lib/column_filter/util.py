# -*- coding: utf-8 -*-
"""Utility functions."""

import numpy as np

__all__ = ['linear_scale', 'log_scale', 'normalize_array']


def linear_scale(xmin, xmax, nx):
    """Compute linear scale.

    Parameters
    ----------
    xmin : float
        Minimum value.
    xmax : float
        Maximum value.
    nx : int
        Number of values.

    Returns
    -------
    np.ndarray, shape=(N,)
        Array of sampled values.

    """

    return np.linspace(xmin, xmax, nx)


def log_scale(xmin, xmax, nx):
    """Compute logarithmic scale.

    Parameters
    ----------
    xmin : float
        Minimum value.
    xmax : float
        Maximum value.
    nx : int
        Number of values.

    Raises
    ------
    ValueError :
        If `xmin` or `xmax` are not greater than zero.

    Returns
    -------
    np.ndarray, shape=(N,)
        Array of sampled values.

    """

    if xmin <= 0:
        raise ValueError("xmin must be greater than zero for logarithmic "
                         "scale.")

    if xmax <= 0:
        raise ValueError("xmax must be greater than zero for logarithmic "
                         "scale.")

    return np.logspace(np.log(xmin),
                       np.log(xmax),
                       nx, base=np.e)


def normalize_array(arr):
    """Normalize a numpy array of shape=(n,3) along axis=1.

    Parameters
    ----------
    arr : np.ndarray, shape=(N,3)
        Data array

    Returns
    -------
    res : np.ndarray, shape=(N,3)
        Normalized data array.

    """

    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    lens[lens == 0] = np.nan
    res = np.zeros_like(arr)
    res[:, 0] = arr[:, 0] / lens
    res[:, 1] = arr[:, 1] / lens
    res[:, 2] = arr[:, 2] / lens
    res_sum = np.sum(res, axis=1)
    res[~np.isfinite(res_sum), :] = 0

    return res
