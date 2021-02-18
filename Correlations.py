"""
Purpose: To compute all pairs Wasserstein on 1D distributions of periodicity
scores for each cell for each gray level and/or number of divisions
Also clustering on this
"""
import numpy as np
import seaborn as sns
from scipy import sparse
import scipy.io as sio
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams as plot_dgms
from SlidingWindow import *
from Periodicity import *
from scipy import integrate

def getAUROC(x, y, do_plot=False, ChunkSize = 1000, MaxLevels = 10000):
    """
    Parameters
    ----------
    x: ndarray(N1)
        True samples
    y: ndarray(N2)
        False samples
    ChunkSize: int
        Size of chunks of levels to do at a time
    
    Returns
    -------
    {
        'TP': ndarray(N)
            True positive thresholds,
        'FP': ndarray(N)
            False positive thresholds,
        'auroc': float
            The area under the ROC curve
    }
    """
    x = np.sort(x)
    y = np.sort(y)
    levels = np.sort(np.unique(np.concatenate((x, y))))
    if len(levels) > MaxLevels:
        levels = np.sort(levels[np.random.permutation(len(levels))[0:MaxLevels]])
    N = len(levels)
    FP = np.zeros(N)
    TP = np.zeros(N)
    i = 0
    while i < N:
        idxs = i + np.arange(ChunkSize)
        idxs = idxs[idxs < N]
        ls = levels[idxs]
        FP[idxs] = np.sum(ls[:, None] < y[None, :], 1) / y.size
        TP[idxs] = np.sum(ls[:, None] < x[None, :], 1) / x.size
        i += ChunkSize
    idx = np.argsort(FP)
    FP = FP[idx]
    TP = TP[idx]
    auroc = integrate.trapz(TP, FP)
    if do_plot:
        plt.plot(FP, TP)
        plt.xlabel("False Positives")
        plt.ylabel("True Positives")
    return {'FP':FP, 'TP':TP, 'auroc':auroc}


def getEuclideanAUROC(X, allndivs, divgroups = [[0, 1], [2, 3, 4], [5, 6]]):
    """
    Parameters
    ----------
    X: ndarray(N, k)
        The periodicity scores for each cell over all blocks
    allndivs: ndarray(N)
        The number of divisions in that cell
    divgroups: d-length list of lists
        Groups of number of divisions that should be compared to each other
    
    Returns
    -------
    aurocs: ndarray(d, d)
        Aurocs between all pairs of groups
    results: {string - > {'FP':ndarray(M), 'TP':ndarray(M), 'auroc':ndarray(M)}}
        All information needed to plot the ROC curves for each pair
    """
    allndivs = np.array(allndivs.flatten(), dtype=int)
    D = getSSM(X)
    N = len(divgroups)
    aurocs = np.zeros((N, N))
    results = {}
    for i, d1 in enumerate(divgroups):
        d1 = np.array(d1, dtype=int)
        idxs1 = np.zeros(D.shape[0])
        for d in d1:
            idxs1[allndivs == d] = 1
        dists11 = (D[idxs1 == 1, :])[:, idxs1 == 1]
        for j in range(i+1, len(divgroups)):
            idxs2 = np.zeros(D.shape[0])
            d2 = np.array(divgroups[j], dtype=int)
            for d in d2:
                idxs2[allndivs == d] = 1
            dists22 = (D[idxs2 == 1, :])[:, idxs2 == 1]
            distsother = ((D[idxs1 == 1, :])[:, idxs2 == 1]).flatten()
            distssame = np.concatenate((dists11.flatten(), dists22.flatten()))
            res = getAUROC(distsother, distssame)
            results["%i_%i"%(i, j)] = res
            aurocs[i, j] = res['auroc']
    aurocs += aurocs.T
    np.fill_diagonal(aurocs, np.nan)
    return aurocs, results