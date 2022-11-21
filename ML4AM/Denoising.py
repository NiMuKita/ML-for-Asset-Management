import pandas as pd
import numpy as np
from typing import Union
from matplotlib import pyplot as plt

from ML4AM.part_2_no_denoising import getPCA, findMaxEval, cov2corr, corr2cov


def denoise_data(data, method='res_eigen', visualize=False, alpha=0.5, bWidth=.01) -> Union[np.array, np.array]:
    '''
    Convert data into correlation matrix, denoise, and return denoised correlation matrix and covariance matrix

    Parameters
    ----------
    data - matrix of values in pd.Series format
    method - 'res_eigen' for  Constant Residual Eigenvalue Method ,'target_shrink' for Targeted Shrinkage Method
    visualize - Visualize eigenvalues of matrix before and after denoising
    alpha - Amount of shrinkage among the eigenvectors and eigenvalues associated with noise in Targeted Shrinkage
    bWidth - When parameter bWidth>0, the covariance matrix is denoised prior to estimating the minimum variance portfolio

    Returns
    -------
    Covariance matrix and Correlation matrix of denoised data
    '''
    corr_init = data.corr()
    cov_init = data.cov()

    eVal, eVec = getPCA(corr_init)

    q = data.shape[0] // data.shape[1]
    eMax, var = findMaxEval(np.diag(eVal), q, bWidth=bWidth)
    nFacts = eVal.shape[0] - np.diag(eVal)[::-1].searchsorted(eMax)

    if method == 'res_eigen':
        # Remove noise from corr by fixing random eigenvalues
        eVal_ = np.diag(eVal).copy()
        eVal_[nFacts:] = eVal_[nFacts:].sum() / float(eVal_.shape[0] - nFacts)
        eVal_ = np.diag(eVal_)
        cov = np.dot(eVec, eVal_).dot(eVec.T)
        corr = cov2corr(cov)
    elif method == 'target_shrink':
        # Remove noise from corr through targeted shrinkage
        eValL, eVecL = eVal[:nFacts, :nFacts], eVec[:, :nFacts]
        eValR, eVecR = eVal[nFacts:, nFacts:], eVec[:, nFacts:]
        corrL = np.dot(eVecL, eValL).dot(eVecL.T)
        corrR = np.dot(eVecR, eValR).dot(eVecR.T)
        corr = corrL + alpha * corrR + (1 - alpha) * np.diag(np.diag(corrR))
        cov = corr2cov(corr, np.diag(cov_init) ** .5)
    else:
        raise ValueError("method should be res_eigen or target_shrink")
    if visualize:
        visualize_denoised(corr,eVal)
    return corr, cov


def visualize_denoised(corr, eVal):
    '''
    Visualize eigenvalues of denoised and undenoised matrix
    Parameters
    ----------
    corr - Correlation matrix
    eVal - Eigenvalues of the matrix

    Returns
    -------

    '''
    val_denoised, vec_denoised = getPCA(corr)
    denoised_eigenvalue = np.diag(val_denoised)
    eigenvalue_prior = np.diag(eVal)
    plt.plot(range(0, len(denoised_eigenvalue)), np.log(denoised_eigenvalue), color='r', label="Denoised eigen-function")
    plt.plot(range(0, len(eigenvalue_prior)), np.log(eigenvalue_prior), color='g', label="Original eigen-function")
    plt.xlabel("Eigenvalue number")
    plt.ylabel("Eigenvalue (log-scale)")
    plt.legend(loc="upper right")
    plt.show()

