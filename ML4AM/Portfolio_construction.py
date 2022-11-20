#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Notes
#Line 78, 115, 205, 206 - must to use modules from other chapters (2,2,2,7)
#Deleted minVarPort function because optPort is the same function

import numpy as np
import scipy.stats as ss
import pandas as pd
from typing import Union, List, Tuple, TypeVar
from sklearn.metrics import mutual_info_score

#import chapters and which modules
from chapter_2 import optPort, cov2corr #chapter2
from Optimal_Clustering import clusterKMeansTop #chapter 7


array_like = Union[np.ndarray, List[Union[int, float]], pd.Series]
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
PandasSeries = TypeVar('pandas.core.series.Series')
pandas_like = Union[PandasDataFrame, PandasSeries]



class Portfolio_construction:
    '''
    
    Class that represent the algorithm of portfolio construction.
    
    Algorith involves 3 parts:
    
    1. Firstly, clusterization of corellation matrix and finding optimal number of clusters.
    2. Secondly, computation of optimal intracluster weights (preferably with denoised covariance matrix).
    3. Thirdly, computation optimal intercluster allocations with reduced matrix obtained from second step.
    
    '''

    @staticmethod 
    def intracluster_weights(cov1: array_like, clstrs: dict) -> Tuple[PandasDataFrame, PandasSeries]:
        
        
        '''
        Function that, for given covarince matrix (preferably denoised) and clusters of data 
        computes optimal intracluster weights.
        
        
        Parameters
        ----------
        
        cov1: array_like
            Denoised covariance matrix.
            
        clstrs: dict 
            List of best structured clusters of observations.
        
        
        Returns
        -------
        
        cov2: PandasDataFrame
            Reduced covariance matrix which reports corellation between clusters.
            
        w_intr_cluster: PandasSeries
            Estimated intracluster weights.
            
        '''
        
        # In case of absence of indexation
        if type(cov1)== np.ndarray or cov1.index is None:
            index = list(i for i in range(cov1.shape[0]))
        else:
            index = cov1.index
            
            
            
        w_intr_cluster = pd.DataFrame(0, index = index, columns = clstrs.keys())
        
        #Computation of intercluster weights
        for i in clstrs:
            w_intr_cluster.loc[clstrs[i], i] = optPort(cov1.loc[clstrs[i], clstrs[i]]).flatten() 

        cov2 = w_intr_cluster.T.dot(np.dot(cov1, w_intr_cluster)) # reduced covariance matrix

        return cov2, w_intr_cluster

    
    
    
    
    @staticmethod     
    def intracluster_allocations(cov2: array_like, w_intr_cluster: pandas_like) -> PandasSeries:
        
        '''
        Function that for given covariance matrix (reduced denoised matrix) and 
        intracluter weights compute optimal intercluster allocations and returns optimal allocation per security
        
        
        Parameters
        ----------
        cov2: array_like 
            Reduced covariance matrix computed by estimation of intracluster weights.
            
        w_intr_cluster: pandas_like
            Estimated intracluster weights.
        
        
        
        Returns
        -------
        w_intr_alloc -  PandasSeries
            Estimated allocation per security.
                
        
        '''
        
        
        w_intr_cluster = pd.DataFrame(optPort(cov2).flatten(), index = cov2.index) 
        w_intr_alloc = w_intr_cluster.mul(w_intr_cluster, axis = 1).sum(axis = 1).sort_index()
        
        return w_intr_alloc
    
    
    @staticmethod   
    def allocate_cvo(self, cov, mu_vec=None):
        
        """
        Function that receive covariation matrix and expected value (or identity matrix) to
        culculate Convex Optimization Solution (CVO).
        
        
        
        Parameters
        ----------
        cov: array_like 
            Covariation matrix.
        mu_vec: array_like 
            Expected value of the variables or identity matrix. 
            For former input result vector of weights will be maxization of Sharpe ratio and
            for latter result will be minimization of variation (risk).
            
        Returns
        -------
        
        w_cvo: array_like
            Optimal allocations
        """
            
        # Calculating the inverse covariance matrix
        inv_cov = np.linalg.inv(cov)

        # Generating a vector of size of the inverted covariance matrix
        ones = np.ones(shape=(inv_cov.shape[0], 1))

        if mu_vec is None:  # To output the minimum variance portfolio
            mu_vec = ones

        # Calculating the analytical solution using CVO - weights
        w_cvo = np.dot(inv_cov, mu_vec)
        w_cvo /= np.dot(mu_vec.T, w_cvo)

        return w_cvo   
    

    def optPort_nco(cov: array_like, mu: array_like = None, maxNumClusters: int = None) -> PandasDataFrame:
        '''
        Function that implement net clustering optimization (NCO) via 
        method of calculation Convex Optimization Solution (CVO)
        
        
        Parameters
        ----------
        
        cov: array_like
            Covariation of observations.
            
        mu: array_like
            Expected value of the variables or identity matrix. 
            For former input result vector of weights will be maxization of Sharpe ratio and
            for latter result will be minimization of variation (risk).
            
        maxNumClusters: int
            Maximum number of clusters. 
            (For no given maximum number recommended half of the number of columns in the correlation matrix)
        
        
        
        Returns
        -------
        nco: PandasDataFrame
            Allocation per security.
        
        
        '''
        
        # In case of absence of indexation
        if type(cov)== np.ndarray or cov1.index is None:
            index = list(i for i in range(cov.shape[0]))
        else:
            index = cov.index
        cov = pd.DataFrame(cov)

        if mu is not None:
            mu = pd.Series(mu[:, 0])
        
        
        # Optimal partition of clusters (step 1)
        corr1 = cov2corr(cov) 
        corr1, clstrs, _ = O_C.clusterKMeansTop(corr1, maxNumClusters, n_init = 10) 
        
        
        # Estimating the Convex Optimization Solution in a cluster (step 2)
        w_intra_cluster = pd.DataFrame(0, index = index, columns = clstrs.keys())
        
        
        #computation of optimal interclusteral weights
        for i in clstrs:
            cov_cluster = cov.loc[clstrs[i], clstrs[i]].values
            if mu is None:
                mu_cluster = None
            else: 
                mu_cluster = mu.loc[clstrs[i]].values.reshape(-1, 1)
                
            # Estimating the Convex Optimization Solution in a cluster (step 2)
            w_intra_cluster.loc[clstrs[i], i] = optPort(cov_, mu_).flatten()
            w_intra_clusters.loc[clstrs[i], i] =self.allocate_cvo(cov_cluster, mu_cluster).flatten()  
        
        
        #computing interclustering allocations
        cov_inter_cluster = w_intra_cluster.T.dot(np.dot(cov, w_intra_cluster))  #reduce covariance matrix
        mu_inter_cluster = (None if mu is None else wIntra.T.dot(mu))
        
        
        # Optimal allocations across the reduced covariance matrix (step 3)
        w_inter_clusters = pd.Series(self.allocate_cvo(cov_inter_cluster, mu_inter_cluster).flatten(), index=cov_inter_cluster.index)    
        
        
        # Final allocations - dot-product of the intra-cluster and inter-cluster allocations (step 4)
        nco = w_intra_cluster.mul(w_inter_clusters, axis = 1).sum(axis = 1).values.reshape(-1, 1)
        return nco

