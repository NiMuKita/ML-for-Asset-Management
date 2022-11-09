import numpy as np
import scipy.stats as ss
import pandas as pd
from typing import Union, List
from sklearn.metrics import mutual_info_score

array_like = Union[np.ndarray, List[Union[int, float]], pd.Series]


def entropy(x: array_like, bins: int) -> float:

    """
    Function that, for a given random variable `x`, with a given partition of `bins`, calculates the Shannon entropy.
    The Shannon entropy is defined as follow: `H(x) = -sum{p(x)*log(p(x))}`

    Parameters
    ----------
    x: array_like
        Array of values.
    bins: int
        Number of partitions.

    Returns
    -------
    h: float
        Calculated value of the Shannon entropy.

    """

    h = ss.entropy(np.histogram(x, bins)[0])

    return h


def mutual_info(x: array_like, y: array_like, bins: int, normalised: bool = False) -> float:

    """
    Mutual information (MI) of two random variables is a measure of the mutual dependence between the two variables.
    For a given random variable `x`, with a given partition of `bins`, calculates the mutual information coefficient.

    Parameters
    ----------
    x: array_like
        Array of `x` values.
    y: array_like
        Array of `y` values.
    bins: int
        Number of partitions.
    normalised: bool
        If True, return normalized by `min(Hx, Hy)` MI coefficient.

    Returns
    -------
    i_xy: float
        Mutual information value.

    """

    c_xy = np.histogram2d(x, y, bins)[0]
    i_xy = mutual_info_score(None, None, contingency=c_xy)

    if normalised:
        h_x = entropy(x, bins)
        h_y = entropy(y, bins)
        i_xy = i_xy / min(h_x, h_y)

    return i_xy


def join_entropy(x: array_like, y: array_like, bins: int) -> float:
    
    """
    Function that, for a given random variables `x` and `y`,
    with a given partition of `bins`, calculates the Shannon joint entropy.

    The joint entropy is defined as follow: `H(x, y) = H(x) + H(y) - I(x,y)`, where Ixy is mutual information.

    Parameters
    ----------
    x: array_like
        Array of `x` values.
    y: array_like
        Array of `y` values.
    bins: int
        Number of partitions.

    Returns
    -------
    h_xy: float
        Joint entropy value.

    """

    h_x = entropy(x, bins)
    h_y = entropy(x, bins)
    i_xy = mutual_info(x, y, bins)

    # Joint entropy formula:
    h_xy = h_x + h_y - i_xy

    return h_xy


def conditional_entropy(x: array_like, y: array_like, bins: int) -> float:

    """
    Function that, for a given random variables `x` and `y`,
    with a given partition of `bins`, calculates the conditional entropy.

    Conditional entropy quantifies the amount of information needed to describe
    the outcome of a random variable X given that the value of another random variable Y is known.

    Chain rule for conditional entropy: `H(x|y) = H(x,y) - H(y)`.

    Parameters
    ----------
    x: array_like
        Array of `x` values.
    y: array_like
        Array of `y` values.
    bins: int
        Number of partitions.

    Returns
    -------
    h_cond: float
        Conditional entropy (X|Y) value.

    """

    h_xy = join_entropy(x, y, bins)
    h_y = entropy(y, bins)

    # Conditional entropy formula:
    h_cond = h_xy - h_y

    return h_cond


def variation_info(x: array_like, y: array_like, bins: int, normalised: bool = False) -> float:

    """
    Function that, for a given random variables `x` and `y`,
    with a given partition of `bins`, calculates the variation of information (VI).

    This measure can be interpreted as the uncertainty we expect in one variable if we are
    told the value of other. VI is a true metric.

    VI can be calculated as follow: `VI(x, y) = H(x) + H(y) - 2 * I(x, y)`

    Parameters
    ----------
    x: array_like
        Array of `x` values.
    y: array_like
        Array of `y` values.
    bins: int
        Number of partitions.
    normalised: bool
        If True, return normalized by joint_entropy H(x, y) value of VI.

    Returns
    -------
    v_xy: float
        Variation of information value.

    """

    i_xy = mutual_info(x, y, bins)
    h_x = entropy(x, bins)
    h_y = entropy(y, bins)

    # Variation of information formula
    v_xy = h_x + h_y - 2 * i_xy

    if normalised:
        h_xy = h_x + h_y - i_xy  # Joint entropy
        v_xy = v_xy / h_xy

    return v_xy


def n_bins(n_obs: int, corr: int = None) -> int:
    
    """
    Function for finding optimal number of bins for a given
    dataset of continuous random variables.

    Parameters
    ----------
    n_obs: int
        Number of observations.
    corr: int
        Correlation coefficient between `x` and `y` random values.

    Returns
    -------
    b: int
        Optimal number of bins.

    """

    if corr is None:  # Univariate case
        z = (8 + 324 * n_obs + 12 * (36 * n_obs + 729 * n_obs ** 2) ** .5) ** (1 / 3.)
        b = round(z / 6. + 2. / (3 * z) + 1. / 3)

    else:  # Bivariate case
        b = round(2 ** (-.5) * (1 + (1 + 24 * n_obs / (1. - corr ** 2)) ** .5) ** .5)

    return int(b)
