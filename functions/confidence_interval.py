import numpy as np
from scipy import stats

"""
Module to make the computation of confidence intervals easier
"""


def z_value(ci_level):
    """
    Compute the z-value that belongs to the confidence interval level

    Parameters
    ----------
    ci_level: float
        Confidence interval level, between 0 and 1

    Returns
    -------
    z: float
        Z-value of the confidence interval
    """
    return stats.norm.interval(ci_level, 0, 1)[1]


def std_n_to_ci(std, n, ci_level):
    """
    Compute the confidence interval value(s), given the standard deviation(s), population size(s), and confidence
    interval level.

    Parameters
    ----------
    std: float or pd.Series of float
        Standard deviation(s)
    n: int or pd.Series of int
        Population size(s)
    ci_level: float
        Desired confidence interval level

    Returns
    -------
    ci_values: float or pd.Series of float
        ci_values (what comes after the +-)

    Notes
    -----
    If 'n' or 'std' is a Series, 'ci_values' will be a series with the same index. If both 'std' and 'n'are series, they
    need to have the same index.
    """
    return z_value(ci_level) * std / (n ** 0.5)


def get_mean_and_ci(collection, ci_level):
    """
    Compute the mean and confidence interval of a given collection for a given ci_level.

    Parameters
    ----------
    collection: Iterable of Number
        The data for which to compute the mean and CI
    ci_level: float
        The confidence level to compute. Should be in [0,1]

    Returns
    -------
    mean: Number
        The mean of the collection
    ci: Number
        The confidence interval of the collection given the ci_level. i.e. We have mean +- ci as the ci_level confidence
        interval.
    """
    mean = np.mean(collection)
    std = np.std(collection)
    return mean, std_n_to_ci(std, len(collection), ci_level)


def latex_string(collection, ci_level, formatter='{:.2f}', as_percentage=False):
    m, s = get_mean_and_ci(collection, ci_level)

    if as_percentage:
        m *= 100
        s *= 100
    foo = rf'${formatter.format(m)}\pm{formatter.format(s)}'
    if as_percentage:
        foo += r"\%"
    return foo + '$'
