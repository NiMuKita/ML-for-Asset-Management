import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
from scipy.linalg import block_diag, inv
from sklearn.covariance import LedoitWolf
from typing import Union


# SOME MATH

def corr2cov(corr: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    :param corr: correlation matrix
    :param std: standart deviation
    :return: covariance matrix
    """
    cov = corr * np.outer(std, std)
    return cov


def cov2corr(cov: np.ndarray) -> np.ndarray:
    """
    :param cov: covariance
    :return: correlation
    """
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(cov))  # Takes square root of diagonal elements of covariance matrix
    corr = cov / np.outer(std, std)  # counts correlation
    corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error
    return corr


# ========================================================================


# MARCHENKO-PASTUR THEOREM
# 2.1
def mpPDF(var: float, q: float, pts: int) -> pd.Series:
    """
    :param var: volatility from Marchenko-Pastur distribution
    :param q: number of observations to number of assets (T/N)
    :param pts: number of bins in graphic
    :return: Marchenko-Pastur distribution as pd.Series
    """
    eMin, eMax = var * (1 - (1. / q) ** .5) ** 2, var * (
                1 + (1. / q) ** .5) ** 2  # Evaluating \lambda_{+} and \lambda_{-}
    eVal = np.linspace(eMin, eMax, pts)  # Count eigenvalues with bins
    pdf = q / (2 * np.pi * var * eVal) * (
                (eMax - eVal) * (eVal - eMin)) ** .5  # Count Marchenko-Pastur distribution
    pdf = pd.Series(pdf, index=eVal)  # creates result as pd.Series from np.array

    return pdf

# ------------------------------------------------------------------------


def getPCA(matrix: np.ndarray) -> tuple:
    """
    :param matrix:
    :return: eigenvalues and eigenvectors as np.array
    """
    # Get eVal, eVec from a Hermitian matrix
    eVal, eVec = np.linalg.eigh(matrix)  # Count eigenvalues and eigenvectors of initial matrix
    indices = eVal.argsort()[::-1]  # arguments for sorting eVal desc
    eVal, eVec = eVal[indices], eVec[:, indices]  # leave eigenvalues and eigenvectors with biggest eigenvalues
    eVal = np.diagflat(eVal)  # create diagonal matrix of eigenvalues
    return eVal, eVec

# ------------------------------------------------------------------------


def fitKDE(obs: np.array, bWidth: float = .25, kernel: str = 'gaussian', x=None) -> pd.Series:
    """
    :param obs: set of observations (one row = one observation)
    :param bWidth: bandwidth (smoothing parameter)
    :param kernel: what function choose as a kernel
    :param x:
    :return: Marchenko-Pastur distribution as pd.Series
    """
    # Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)  # transpose observations
    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)  # create kernel density estimator function
    if x is None:
        x = np.unique(obs).reshape(-1, 1)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    logProb = kde.score_samples(x)  # log(density)
    pdf = pd.Series(np.exp(logProb), index=x.flatten())  # exp(log(result))
    return pdf

# ========================================================================


# RANDOM COVARIANCE MATRIX BUILDING
# 2.3
def getRndCov(nCols: int, nFacts: int) -> np.array:
    """
    :param nCols: number of columns
    :param nFacts: number of rows
    :return: random covariance matrix as np.array
    """
    w = np.random.normal(size=(nCols, nFacts))  # creates random matrix of size nCols \times nFacts
    cov = np.dot(w, w.T)  # random cov matrix, however not full rank
    cov += np.diag(np.random.uniform(size=nCols))  # full rank cov
    return cov

# ========================================================================


# FITTING MARCHENKO-PASTUR DISTRIBUTION
# 2.4
def errPDFs(var: list, eVal: np.array, q: float, bWidth: float, pts: int = 1000) -> float:
    """
    :param var: volatility distribution
    :param eVal: np.array of eigenvalues
    :param q: number of observations to number of assets (T/N)
    :param bWidth: bandwidth (smoothing parameter)
    :param pts: number of bins in graphic
    :return: sum of the squared differences between the analytical PDF and the kernel
            density estimate (KDE) of the observed eigenvalues
    """
    var = var[0]
    pdf0 = mpPDF(var, q, pts)  # theoretical probability density function
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values)  # empirical pdf
    sse = np.sum((pdf1 - pdf0) ** 2)  # sum of the squared differences
    # print(sse)
    return sse

# ------------------------------------------------------------------------


def findMaxEval(eVal: np.array, q: float, bWidth: float) -> tuple:
    """
    :param eVal: np.array of eigenvalues
    :param q: number of observations to number of assets (T/N)
    :param bWidth: bandwidth (smoothing parameter)
    :return: max eigenvalue
    """
    # Find max random eVal by fitting Marcenko’s dist

    out = minimize(lambda *x: errPDFs(*x), x0=np.array(0.5), args=(eVal, q, bWidth),
                   bounds=((1E-5, 1 - 1E-5),))
    # row above minimize sum of the squared differences between the analytical PDF and the kernel density
    # estimate (KDE) of the observed eigenvalues

    if out['success']:
        # print('success')
        var = out['x'][0]
    else:
        var = 1
    eMax = var * (1 + (1. / q) ** .5) ** 2
    return eMax, var

# ========================================================================


# GENERATING A BLOCK-DIAGONAL COVARIANCE MATRIX AND A VECTOR OF MEANS


# 2.7
# Creates Block-diagonal matrix
def formBlockMatrix(nBlocks: int, bSize: int, bCorr: float) -> np.array:
    """
    :param nBlocks: number of blocks
    :param bSize: size of block
    :param bCorr: correlation of off-diagonal elements within each block
    :return: block-diagonal matrix
    """
    block = np.ones((bSize, bSize)) * bCorr
    block[range(bSize), range(bSize)] = 1
    corr = block_diag(*([block] * nBlocks))
    return corr

# ------------------------------------------------------------------------


def formTrueMatrix(nBlocks: int, bSize: int, bCorr: float) -> Union[np.array, np.array]:
    """
    :param nBlocks: number of blocks
    :param bSize: size of block
    :param bCorr: correlation of off-diagonal elements within each block
    :return: a vector of means and a covariance matrix as np.array
    """
    corr0 = formBlockMatrix(nBlocks, bSize, bCorr)  # Creates block matrix
    corr0 = pd.DataFrame(corr0)
    cols = corr0.columns.tolist()  # gets set of dataframe columns
    np.random.shuffle(cols)  # shuffle columns
    corr0 = corr0[cols].loc[cols].copy(deep=True)
    std0 = np.random.uniform(.05, .2, corr0.shape[0])
    cov0 = corr2cov(corr0, std0)  # count covariance matrix using correlation
    mu0 = np.random.normal(std0, std0, cov0.shape[0]).reshape(-1, 1)  # simulate means
    return mu0, cov0

# ========================================================================

# GENERATING THE EMPIRICAL COVARIANCE MATRIX


# 2.8
def simCovMu(mu0: np.array, cov0: np.array, nObs: int, shrink: bool = False) -> Union[np.array, np.array]:
    """
    :param mu0: true means
    :param cov0: true covariance matrix
    :param nObs: number of observations (T)
    :param shrink: True to apply Ledoit-Wolf shrinkage of the empirical covariance matrix, else False
    :return: simulated vector of means and covariance matrix as np.array
    """
    x = np.random.multivariate_normal(mu0.flatten(), cov0, size=nObs)
    mu1 = x.mean(axis=0).reshape(-1, 1)  # Count means
    if shrink:
        cov1 = LedoitWolf().fit(x).covariance_
    else:
        cov1 = np.cov(x, rowvar=False)
    return mu1, cov1


# ========================================================================


def optPort(cov: np.array, mu: np.array = None) -> np.array:
    """
    :param cov: covariance matrix
    :param mu: vector of means
    :return:
    """
    # to derive the minimum variance portfolio
    inv = np.linalg.inv(cov)  # inverse covariance matrix
    ones = np.ones(shape=(inv.shape[0], 1))
    if mu is None:
        mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)
    return w
