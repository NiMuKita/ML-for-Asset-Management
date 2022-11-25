#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.stats as ss
import itertools
#- - - - - - - -- - - - - - - - - - - - -- - - - - - - - - - - - - -- - - - - - - - - - - - -- - -
def getExpectedMaxSR(nTrials,meanSR,stdSR):
    """
    Function that, computes the theoretical E[SR_(n)] = expected Sharpe Ratio of the n'th order statistics (max)
    Parameters.
    ----------
    nTrials: int
    meanSR: float
    stdSR: float
    Returns
    -------
    sr0: float
    """
# Expected max SR, controlling for SBuMT
    emc=0.577215664901532860606512090082402431042159336
    sr0 = (1-emc)*norm.ppf(1-1./nTrials)+emc*norm.ppf(1-(nTrials*np.e)**-1)
    emc*norm.ppf(1-(nTrials*np.e)**-1)
    sr0=meanSR+stdSR*sr0
    return sr0
#- - - - - - - -- - - - - - - - - - - - -- - - - - - - - - - - - - -- - - - - - - - - - - - -- - -
def getDistMaxSR(nSims,nTrials,stdSR,meanSR):
    """
    Monte carlo of max{SR} on nTrials, from nSims simulations
    Parameters
    ----------
    nSims: int
        number of simulations
    nTrials: int
        number of trials
    stdSR: float
        standart deviation of sharpe ratio
    meanSR: float
        mean of sharpe ratio
    Returns
    -------
    out:PandasDataFrame
        list of ratio changings
    """
# Monte Carlo of max{SR} on nTrials, from nSims simulations
    rng=np.random.RandomState()
    out=pd.DataFrame()
    for nTrials_ in nTrials:
#1) Simulated Sharpe ratios
        sr=pd.DataFrame(rng.randn(nSims,nTrials_))
        sr=sr.sub(sr.mean(axis=1),axis=0) # center
        sr=sr.div(sr.std(axis=1),axis=0) # scale
        sr=meanSR+sr*stdSR
#2) Store output
        out_=sr.max(axis=1).to_frame('max{SR}')
        out_['nTrials']=nTrials_
        out=out.append(out_,ignore_index=True)
        return out
#- - - - - - - -- - - - - - - - - - - - -- - - - - - - - - - - - - -- - - - - - - - - - - - -- - -
# code snippet 8.2 - mean and standard deviation of the prediction errors
def getMeanStdError(nSims0, nSims1, nTrials, stdSR, meanSR):
    """
    Function that, computes standard deviation of errors per nTrials
    Parameters
    ----------
    nSims0: int
        number of max{SR} used to estimate E[max{SR}].
    nSims1: int
        number of errors on which std is computed.
    nTrials: int
        number of SR used to derive max{SR}
    stdSR: float
        standart deviation of sharpe ratio
    meanSR: float
        mean of sharpe ratio
    Returns
    -------
    out: PandasDataFrame
        mean and standart deviation errors
    """
    #compute standard deviation of errors per nTrials
    #nTrials: [number of SR used to derive max{SR}]
    #nSims0: number of max{SR} u{sed to estimate E[max{SR}]
    #nSims1: number of errors on which std is computed
    sr0=pd.Series({i:getExpectedMaxSR(i, meanSR, stdSR) for i in nTrials})
    sr0 = sr0.to_frame('E[max{SR}]') 
    sr0.index.name='nTrials'
    err=pd.DataFrame()
    for i in range(0, int(nSims1)):
        sr1 = getDistMaxSR(nSims=100, nTrials=nTrials, meanSR=0, stdSR=1)
        sr1=sr1.groupby('nTrials').mean()
        err_=sr0.join(sr1).reset_index()
        err_['err'] = err_['max{SR}']/err_['E[max{SR}]']-1.
        err = pd.concat([err, err_], ignore_index=True)
    out = {'meanErr':err.groupby('nTrials')['err'].mean()}
    out['stdErr'] = err.groupby('nTrials')['err'].std()
    out = pd.DataFrame.from_dict(out, orient='columns')
    return out
#- - - - - - - -- - - - - - - - - - - - -- - - - - - - - - - - - - -- - - - - - - - - - - - -- - -
# code snippet 8.3 - Type I (False positive), with numerical example (Type II False negative)
def getZStat(sr, t, sr_, skew, kurt):
    """
    Function that, computes Z-statistics
    Parameters
    ----------
    sr: float
        sharpe ratio estimation
    t:  float
        number of observations
    sr_: int
        threshold
    skew: float
        skewness of the returns
    kurt: float
        kurtosis of the returns
    Returns
    -------
    z: float
        z statistics
    """
    z = (sr-sr_)*(t-1)**.5
    z /= (1-skew*sr+(kurt-1)/4.*sr**2)**.5
    return z
#- - - - - - - -- - - - - - - - - - - - -- - - - - - - - - - - - - -- - - - - - - - - - - - -- - -
def type1Err(z, k):
    """
    Function that, computes type 1 error
    Parameters
    ----------
    z: float
        z statistic
    k: int
        number of observations
    Returns
    -------
    alpha_k: float
        type 1 error
    """
    #false positive rate
    alpha = ss.norm.cdf(-z)
    alpha_k = 1-(1-alpha)**k #multi-testing correction
    return alpha_k
#- - - - - - - -- - - - - - - - - - - - -- - - - - - - - - - - - - -- - - - - - - - - - - - -- - -       
# code snippet 8.4 - Type II error (false negative) - with numerical example
def getTheta(sr, t, sr_, skew, kurt):
    """
    Function that, computes theta statistics
    Parameters
    ----------
    sr: float
        sharpe ratio estimation
    t: float
        number of observations
    sr_: float
        threshold
    skew: float
        skewness of the returns
    kurt: float
        kurtosis of the returns
    Returns
    -------
    theta: float
        theta statistic
    """
    theta = sr_*(t-1)**.5
    theta /= (1-skew*sr+(kurt-1)/.4*sr**2)**.5
    return theta
#- - - - - - - -- - - - - - - - - - - - -- - - - - - - - - - - - - -- - - - - - - - - - - - -- - -    
def type2Err(alpha_k, k, theta):
    """
    Function that, computes type 2 error
    Parameters
    ----------
    alpha_k: float
        type 1 error
    k: int
        number of observations
    theta: float
        theta statistic
    Returns
    -------
    beta: float
        type 2 error
    """
    #false negative rate
    z = ss.norm.ppf((1-alpha_k)**(1./k)) #Sidak's correction
    beta = ss.norm.cdf(z-theta)
    return beta


# In[ ]:





# In[ ]:




