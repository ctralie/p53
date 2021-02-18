"""
Purpose: To compute all pairs Wasserstein on 1D distributions of periodicity
scores for each cell for each gray level.
Also clustering on this
"""
import numpy as np
from scipy import sparse
import scipy.io as sio
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams as plot_dgms
from SlidingWindow import *
from Periodicity import *
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import wasserstein_distance

def getWasserstein(X, dim = 30):
    """
    Compute all pairs of Wasserstein distances between 
    distributions of a list of values, assuming that
    all of the lists have the same number of values so that
    the CDF is easily invertible
    Parameters
    ----------
    X: ndarray(N, M)
        N lists, each with M unordered samples
    Returns
    -------
    D: ndarray(N, N)
        All pairs 1D Wasserstein distances between lists in X
    """
    bins = np.linspace(0, 1, dim)
    N = X.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        if i%250 == 0:
            print("%i of %i"%(i, N))
        histi, _ = np.histogram(X[i, :], bins=bins)
        for j in range(i+1, N):
            histj, _ = np.histogram(X[j, :], bins=bins)
            D[i, j] = wasserstein_distance(histi, histj)
    D += D.T
    return D

def getWassersteinSw1PerSBlocks(scoretype="pitch"):
    """
    Compute the Wasserstein distances between all pairs of
    distributions of block scores for each time series over 
    all gray levels
    Parameters
    ----------
    scoretype: string
        The type of scores we're looking at 
        "pitch" for autocorrelation or "tda" for Sw1PerS
    Returns
    -------
    D: ndarray(N, N)
        All pairs Wasserstein distances between time series
    grays: ndarray(N)
        The gray levels of each row/column of D
    X: ndarray(N, NBlocks)
        All scores for all blocks
    allndivs: ndarray(N)
        The number of divisions of each row/column of D
    """
    X = np.array([])
    grays = np.array([])
    allndivs = np.array([])
    for t in [0, 2, 4, 10]:
        scores = np.loadtxt("data/p53_%iGy_%s_scores.txt"%(t, scoretype))
        divs = np.loadtxt("data/div_%iGy.txt"%t)
        ndivs = np.sum(divs, 1)
        valid = np.sum(np.isnan(scores), 1) == 0
        scores = scores[valid == 1, :]
        ndivs = ndivs[valid == 1]
        idx = np.argsort(ndivs)
        ndivs = ndivs[idx]
        scores = scores[idx, :]
        if X.size == 0:
            X = scores
        else:
            X = np.concatenate((X, scores), 0)
        grays = np.concatenate((grays, np.array([t]*scores.shape[0])))
        allndivs = np.concatenate((allndivs, ndivs))
    D = getWasserstein(X)
    return D, X, grays, allndivs


def getWassersteinSw1PerSBlocks_July2019(scoretype="pitch"):
    """
    Compute the Wasserstein distances between all pairs of
    distributions of block scores for each time series over 
    all gray levels, for the July 2019 dataset
    Parameters
    ----------
    scoretype: string
        The type of scores we're looking at 
        "pitch" for autocorrelation or "tda" for Sw1PerS
    Returns
    -------
    D: ndarray(N, N)
        All pairs Wasserstein distances between time series
    grays: ndarray(N)
        The gray levels of each row/column of D
    X: ndarray(N, NBlocks)
        All scores for all blocks
    allndivs: ndarray(N)
        The number of divisions of each row/column of D
    """
    X = np.array([])
    grays = np.array([])
    allndivs = np.array([])
    X = np.loadtxt("data_july_2019/p53_%s_scores.txt"%scoretype)
    divs = np.loadtxt("data_july_2019/divisions.txt")
    divs = divs[:, 0:240]
    allndivs = np.sum(divs, 1)
    idx = np.argsort(allndivs)
    allndivs = allndivs[idx]
    X = X[idx, :]
    fin = open("data_july_2019/p53_indices.txt")
    grayidxs = [int(s) for s in fin.readlines()[0].split()]
    fin.close()
    grays = np.zeros(X.shape[0])
    for i, v in enumerate([0, 0.5, 1, 2, 4, 8]):
        grays[grayidxs[i]:grayidxs[i+1]] = v
    grays = grays[idx]
    D = getWasserstein(X)
    return D, X, grays, allndivs

def getEuclideanAutocorrelation():
    """
    Compute the Euclidean distances and autocorrelation
    between all pairs of time series
    Returns
    -------
    D: ndarray(N, N)
        All pairs Wasserstein distances between time series
    grays: ndarray(N)
        The gray levels of each row/column of D
    allndivs: ndarray(N)
        The number of divisions of each row/column of D
    """
    X = np.array([])
    grays = np.array([])
    allndivs = np.array([])
    for t in [0, 2, 4, 10]:
        Xt = np.loadtxt("data/p53_%iGy.txt"%t)
        divs = np.loadtxt("data/div_%iGy.txt"%t)
        ndivs = np.sum(divs, 1)
        valid = np.sum(Xt == -1, 1) == 0
        Xt = Xt[valid == 1, :]
        ndivs = ndivs[valid == 1]
        idx = np.argsort(ndivs)
        ndivs = ndivs[idx]
        Xt = Xt[idx, :]
        if X.size == 0:
            X = Xt
        else:
            X = np.concatenate((X, Xt), 0)
        grays = np.concatenate((grays, np.array([t]*Xt.shape[0])))
        allndivs = np.concatenate((allndivs, ndivs))
    N = X.shape[0]
    Win = 11
    XA = []
    for i in range(N):
        xk = X[i, :]
        xk = detrend_timeseries(xk, Win)
        X[i, :] = xk
        xk = xk - np.mean(xk)
        y = np.correlate(xk, xk, 'full') 
        y = y[y.size//2:]
        y /= y[0]
        XA.append(y)
    XA = np.array(XA)
    D = getSSM(X)
    DA = getSSM(XA)
    return D, DA, X, grays, allndivs

def make_clusterplots(D, X, grays, allndivs, nclusters = 4, euclidean=False, filename="Clusters.png"):
    """
    Make Caroline's cluster plots and save them to a file "Clusters.png"
    Parameters
    ----------
    D: ndarray(N, N)
        A distance matrix between all distributions
    X: ndarray(N, K)
        A set of K samples in each distribution for each time series
    grays: ndarray(N)
        Gray levels for each time series
    allndivs: ndarray(N)
        Number of divisions for each time series
    nclusters: int
        Number of clusters to perform
    eucildean: boolean
        Whether to treat the rows of the distance matrix as
        Euclidean coordinates or whether to use the raw distance matrix
    filename: string
        Name of file to which to save plot
    """
    N = D.shape[0]
    if euclidean:
        Z = linkage(D, 'ward')
    else:
        Z = linkage(D[np.triu_indices(N, 1)], 'ward')
    cluster_idxs = fcluster(Z, nclusters, criterion='maxclust')
    allcounts = []
    labels = []
    allidxs = [] # Indices of time series in each gray/divs group
    grays = grays.flatten()
    allndivs = allndivs.flatten()
    for gray in np.unique(grays):
        for ndivs in np.unique(allndivs):
            idxs = cluster_idxs[(grays == gray)*(allndivs == ndivs)]
            if len(idxs) == 0:
                continue
            allidxs.append(idxs)
            counts = [np.sum(idxs == c) for c in range(1, nclusters+1)]
            allcounts.append(counts)
            labels.append("%g/%g"%(gray, ndivs))
    allcounts = np.array(allcounts, dtype=float).T
    idx = np.argmax(allcounts, 0)
    allcountsmax = np.zeros(allcounts.shape, dtype=int)
    allcountsmax[idx, np.arange(allcounts.shape[1])] = 1
    allcounts /= np.sum(allcounts, 0)[None, :]
    # Compute means for each type
    means_per_type = []
    for i, l in enumerate(labels):
        gray, ndivs = [float(s) for s in l.split("/")]
        x = X[(grays == gray)*(allndivs == ndivs), :]
        means_per_type.append(np.mean(x))
    # Compute means for each cluster
    means_per_cluster = np.zeros(allcountsmax.shape[0])
    for i in range(allcountsmax.shape[0]):
        x = np.array([])
        for j in range(allcountsmax.shape[1]):
            if allcountsmax[i, j] > 0:
                gray, ndivs = [float(s) for s in labels[j].split("/")]
                xj = X[(grays == gray)*(allndivs == ndivs)]
                if x.size == 0:
                    x = xj
                else:
                    x = np.concatenate((x, xj), 0)
        means_per_cluster[i] = np.mean(x.flatten())
    # Sort clusters in ascending order of mean
    idx = np.argsort(means_per_cluster)
    means_per_cluster = means_per_cluster[idx]
    allcounts = allcounts[idx, :]
    allcountsmax = allcountsmax[idx, :]
    # Plot the results
    plt.figure(figsize=(20, 6))
    plt.subplot(211)
    plt.imshow(allcounts, vmin=0, vmax=1)
    plt.xticks(np.arange(allcounts.shape[1]), labels)
    plt.subplot(212)
    plt.imshow(allcountsmax)
    plt.xticks(np.arange(allcounts.shape[1]), ["%s\n%.3g"%(l, m) for (l, m) in zip(labels, means_per_type)])
    plt.yticks(np.arange(len(means_per_cluster)), ["  %.3g  "%m for m in means_per_cluster])
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')

def exclude_gray_div_types(D, X, grays, allndivs, types):
    """
    Exclude some types of gray/division combinations from the data
    Parameters
    ----------
    D: ndarray(N, N)
        A distance matrix between all distributions
    X: ndarray(N, K)
        A set of K samples in each distribution for each time series
    grays: ndarray(N)
        Gray levels for each time series
    allndivs: ndarray(N)
        Number of divisions for each time series
    types: list of tuple(gray, div)
        The types to exlude
    Returns
    -------
    D, X, grays, allndivs
        The above variables without gray/div types
    """
    grays = grays.flatten()
    allndivs = allndivs.flatten()
    for (gray, div) in types:
        idx = np.arange(D.shape[0])
        idx = idx[(grays == gray)*(allndivs == div) == 0]
        D = D[idx, :]
        D = D[:, idx]
        X = X[idx, :]
        grays = grays[idx]
        allndivs = allndivs[idx]
    return D, X, grays, allndivs

def testEuclideanAutocorrelation():
    D, DA, X, grays, allndivs = getEuclideanAutocorrelation()
    make_clusterplots(D, X, grays, allndivs, nclusters=3, filename="EuclideanClusters.png")
    make_clusterplots(DA, X, grays, allndivs, nclusters=3, filename="AutocorrelationClusters.png")

if __name__ == '__main__':
    D, X, grays, allndivs = getWassersteinSw1PerSBlocks("tda")
    sio.savemat("tda.mat", {"D":D, "X":X, "grays":grays, "allndivs":allndivs})
    """
    res = sio.loadmat("pitch.mat")
    D, X, grays, allndivs = res["D"], res["X"], res["grays"], res["allndivs"]
    make_clusterplots(D, X, grays, allndivs, nclusters=3, filename="ClustersPitch.png")
    """

if __name__ == '__main__2':
    #D, X, grays, allndivs = getWassersteinSw1PerSBlocks_July2019("tda")
    #sio.savemat("tda_july_2019.mat", {"D":D, "X":X, "grays":grays, "allndivs":allndivs})
    res = sio.loadmat("tda_july_2019.mat")
    D, X, grays, allndivs = res["D"], res["X"], res["grays"], res["allndivs"]
    #D, X, grays, allndivs = exclude_gray_div_types(D, X, grays, allndivs, [(4, 3), (2, 7)])
    nclusters = 4
    make_clusterplots(D, X, grays, allndivs, nclusters=nclusters, filename="ClustersPitch_July_2019_%iClusters.png"%nclusters)
    nclusters = 6
    make_clusterplots(D, X, grays, allndivs, nclusters=nclusters, filename="ClustersPitch_July_2019_%iClusters.png"%nclusters)