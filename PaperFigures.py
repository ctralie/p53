"""
Code to generate figures used in our conference paper
"""
import numpy as np
from scipy import sparse
import scipy.io as sio
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams as plot_dgms
from SlidingWindow import *
from Periodicity import *
from Correlations import *
import seaborn as sns

timeticks = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120]

def get0GrayData():
    divs = np.loadtxt("data/div_0Gy.txt")
    allndivs = np.sum(divs, 1)
    X = np.loadtxt("data/p53_0Gy.txt")
    idx = np.arange(X.shape[0])
    idx = idx[np.sum(X, 1) > 0]
    X = X[idx, :]
    allndivs = allndivs[idx]
    return X, allndivs

def getExamples():
    """
    Return an example of p53 time series with 0 divisions
    and an example with 5 divisions.  These examples will be
    used throughout all figures
    Returns
    -------
    {
        'x0': ndarray(N)
            A time series with 0 divisions,
        'x5': ndarray(N)
            A time series with 5 divisions,
        'dividx': ndarray(5)
            Indices of the 5 division events in x5,
        't': ndarray(N),
            Time in hours of each sample in x0 and x5
    }
    """
    X, ndivs = get0GrayData()
    x0 = X[ndivs == 0, :]
    x0 = x0[0, :]
    x5 = X[ndivs == 5, :]
    x5 = x5[0, :]
    divs5 = divs[ndivs == 5, :]
    divs5 = divs5[0, :]
    dividx = np.arange(divs5.size)
    dividx = dividx[divs5 == 1]
    t = 0.5*np.arange(x0.size)
    return {'x0':x0, 'x5':x5, 'dividx':dividx, 't':t}



def PeriodicityScoringExamples():
    """
    Show examples of detrending time series and scoring their
    periodicity
    """
    res = getExamples()
    x0, x5, dividx, t = res['x0'], res['x5'], res['dividx'], res['t']
    ymax = max(np.max(x0), np.max(x5))*1.1

    Win = 11
    for x, s in zip([x0, x5], ["0", "5"]):
        plt.figure(figsize=(8, 6))
        idx, score, y, x_detrended = pitch_detection(x, detrend_win=Win)
        plt.clf()
        plt.subplot(311)
        plt.plot(t, x)
        plt.xlim([0, np.max(t)])
        plt.ylim([0, ymax])
        if s == "5":
            plt.scatter(t[dividx], x[dividx], c='C1')
        plt.xticks(timeticks)
        plt.ylabel("p53 Nuclear\nFluorescence")
        plt.xlabel("Elapsed Time (Hours)")
        plt.title("%s Divisions"%s)

        plt.subplot(312)
        plt.plot(t, x_detrended)
        plt.xlim([0, np.max(t)])
        plt.ylim([-1, 1])
        if s == "5":
            plt.scatter(t[dividx], x_detrended[dividx], c='C1')
        plt.xticks(timeticks)
        plt.ylabel("p53 Nuclear\nFluorescence (Detrended)")
        plt.xlabel("Elapsed Time (Hours)")
        plt.title("%s Divisions Detrended"%s)

        plt.subplot(313)
        plt.plot(t[np.arange(y.size)], y)
        plt.stem([t[idx]], [score])
        plt.xlabel("Shift (hours)")
        plt.xticks(timeticks)
        plt.xlim([0, np.max(t)])
        plt.title("Normalized Autocorrelation (Score %.3g)"%score)
        plt.tight_layout()
        plt.savefig("Figures/Autocorrelation_%s.svg"%s, bbox_inches='tight')



def BlockConcept():
    """
    Make a figure showing what it means to extract blocks
    from a detrended time series and score them with DAPs
    """
    res = getExamples()
    x0, x5, dividx, t = res['x0'], res['x5'], res['dividx'], res['t']
    ymax = max(np.max(x0), np.max(x5))*1.1
    block_len = 48
    Win = 11
    idx, score, y, xd = pitch_detection(x0, detrend_win=Win)

    starts = [0, 40, 77]

    plt.figure(figsize=(8, 4))
    plt.subplot2grid((2, 3), (0, 0), colspan=3)
    plt.plot(t, xd)
    plt.xlim([-1, np.max(t)])
    plt.ylim([-1, 1])
    plt.xticks(timeticks)
    plt.ylabel("p53 Nuclear Fluorescence\n(Detrended)")
    plt.xlabel("Elapsed Time (Hours)")

    Win = int(block_len*0.5) # Block length 24 hours
    c = 'C1'
    AW = 0.1
    AXW = 0.005
    y1, y2 = np.min(xd), np.max(xd)
    pad = 0.3*(y2-y1)
    #c = np.array([1.0, 0.737, 0.667])
    ax = plt.gca()
    for s in starts:
        ax.arrow(s+Win, y2+0.3*pad, 4, 0, head_width = AW, head_length = 2, fc = c, ec = c, width = AXW)
        ax.arrow(s+Win, y1-0.3*pad, 4, 0, head_width = AW, head_length = 2, fc = c, ec = c, width = AXW)
        plt.plot([s, s+Win], [y1-pad, y1-pad], c=c)
        plt.plot([s, s+Win], [y2+pad, y2+pad], c=c)
        plt.plot([s, s], [y1-pad, y2+pad], c=c)
        plt.plot([s+Win, s+Win], [y1-pad, y2+pad], c=c)
    plt.title("0 Divisions Detrended Block Extraction And Scoring")
    
    starts = [s*2 for s in starts]
    for i, s in enumerate(starts):
        plt.subplot(2, 3, 4+i)
        xdi = xd[s:s+block_len]
        _, score, _, _ = pitch_detection(xdi)
        plt.plot(t[s:s+block_len], xd[s:s+block_len])
        plt.title("Score = %.3g"%score)
        plt.ylim([-1, 1])
        if i == 0:
            plt.ylabel("p53 Nuclear Fluorescence\n(Detrended)")
        plt.xlabel("Elapsed Time (Hours)")
    plt.tight_layout()
    plt.savefig("Figures/BlockConcept.svg", bbox_inches='tight')


def BlockDistributions():
    """
    Show the histograms of the pitch scores for all blocks in the
    0 division and 5 division example next to each other
    """
    res = getExamples()
    x0, x5, dividx, t = res['x0'], res['x5'], res['dividx'], res['t']
    ymax = max(np.max(x0), np.max(x5))*1.1
    block_len = 48
    plt.figure(figsize=(12, 6))
    Win = 11
    allscores = []
    for x, s in zip([x0, x5], ["0", "5"]):
        plt.clf()
        # Get all block scores
        k = 0
        scores = []
        idx, score, y, xd = pitch_detection(x, detrend_win=Win)
        while k+block_len < xd.size:
            xk = xd[k:k+block_len]
            k += 1
            _, score, _, _ = pitch_detection(xk)
            scores.append(score)
        allscores.append(np.array(scores))
    
    plt.figure(figsize=(6, 3))
    sns.distplot(allscores[0])
    sns.distplot(allscores[1])
    plt.legend(["0 Divisions", "5 Divisions"])
    plt.xlabel("Pitch Score")
    plt.ylabel("Density")
    plt.title("Pitch Score Distributions")
    plt.savefig("Figures/Distributions.svg", bbox_inches='tight')


def ROCFigures():
    plt.figure(figsize=(4, 4))
    path = "tda.mat"
    name = "RawSignal"
    k = 0
    #for k, (path, name) in enumerate(zip(["tda.mat"],
    #                                     ["TDA"])):
    res = sio.loadmat(path)
    #X, grays, allndivs = res["X"], res["grays"], res["allndivs"]
    #X = np.reshape(X, (X.size, 1))
    divs = np.loadtxt("data/div_0Gy.txt")
    allndivs = np.sum(divs, 1)
    X = np.loadtxt("data/p53_0Gy.txt")
    idx = np.arange(X.shape[0])
    idx = idx[np.sum(X, 1) > 0]
    X = X[idx, :]
    allndivs = allndivs[idx]
    
    #grays = grays.flatten()
    #allndivs = allndivs.flatten()
    #X = X[grays == 0, :]
    #allndivs = allndivs[grays == 0]
    #grays = grays[grays == 0]
    aurocs, results = getEuclideanAUROC(X, allndivs, divgroups = [[0], [1], [2], [3], [4], [5]])
    
    print("k = ", k)
    plt.subplot(1, 1, k+1)
    sns.heatmap(aurocs, annot=True, fmt='.2f')
    plt.xlabel("Number of Divisions")
    plt.ylabel("Number of Divisions")
    plt.title(name)
    plt.savefig("ROCResults.svg", bbox_inches='tight')

def makeClusterPlot(D, X, allndivs, nclusters):
    """
    Compute and plot clusters for all cells that are 0 gray,
    given a set of distances computed between all cells
    D: ndarray(N, N)
        A distance matrix between all cells
    X: ndarray(N, K)
        A set of K samples in each distribution for each time series
    allndivs: ndarray(N)
        Number of divisions for each time series
    nclusters: int
        Number of clusters to perform
    """
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    N = D.shape[0]
    Z = linkage(D[np.triu_indices(N, 1)], 'ward')
    cluster_idxs = fcluster(Z, nclusters, criterion='maxclust')
    allcounts = []
    labels = []
    allidxs = [] # Indices of time series in each gray/divs group
    allndivs = allndivs.flatten()
    for ndivs in np.unique(allndivs):
        idxs = cluster_idxs[allndivs == ndivs]
        if len(idxs) == 0:
            continue
        allidxs.append(idxs)
        counts = [np.sum(idxs == c) for c in range(1, nclusters+1)]
        allcounts.append(counts)
        labels.append("%g"%ndivs)
    allcounts = np.array(allcounts, dtype=float).T
    idx = np.argmax(allcounts, 0)
    allcounts /= np.sum(allcounts, 0)[None, :]

    # Compute mean periodicity scores for each cluster
    means_per_cluster = np.zeros(nclusters)
    for i in range(nclusters):
        scores = X[allndivs == i, :]
        means_per_cluster[i] = np.mean(scores)

    # Compute mean periodicity scores for each number of divisions
    means_per_type = []
    for ndivs in np.unique(allndivs):
        x = X[allndivs == ndivs, :]
        means_per_type.append(np.mean(x))

    # Sort clusters in ascending order of mean
    idx = np.argsort(means_per_cluster)
    means_per_cluster = means_per_cluster[idx]
    allcounts = allcounts[idx, :]
    # Plot the results
    plt.imshow(allcounts, vmin=0, vmax=1)
    plt.xticks(np.arange(allcounts.shape[1]), ["%s\n%.2g"%(l, m) for (l, m) in zip(labels, means_per_type)])
    plt.yticks(np.arange(len(means_per_cluster)), ["  %.2g  "%m for m in means_per_cluster])
    for x in range(allcounts.shape[1]):
        for y in range(allcounts.shape[0]):
            if allcounts[y][x] < 0.01:
                allcounts[y][x] = 0
            plt.text(x-0.3, y, "%.2g"%allcounts[y][x], color='white')
    plt.xlabel("Number of Divisions")
    plt.ylabel("Cluster")

def ClusterPlots():
    """
    Plot the results of single linkage clustering using different types
    of distances
    """
    from Wasserstein import getWasserstein
    plt.figure(figsize=(12, 10))
    nclusters=4

    # First make a plot with Euclidean on the raw data
    X, allndivs = get0GrayData()
    plt.subplot(337)
    makeClusterPlot(getSSM(X), X, allndivs, nclusters)
    plt.title("Euclidean Raw Data")

    for row, (filename, score) in enumerate(zip(["pitch.mat", "tda.mat"], ["DAPs", "Sw1PerS"])):
        res = sio.loadmat(filename)
        X, grays, allndivs = res["X"], res["grays"].flatten(), res["allndivs"].flatten()
        X = X[grays == 0, :]
        allndivs = allndivs[grays == 0]
        for col, dist in enumerate(["Euclidean", "Euclidean Sorted", "Wasserstein"]):
            plt.subplot(3, 3, row*3+col+1)
            if col == 0:
                D = getSSM(X)
            elif col == 1:
                D = getSSM(np.sort(X, 1))
            else:
                D = getWasserstein(X)
            makeClusterPlot(D, X, allndivs, nclusters)
            plt.title("%s %s"%(score, dist))
    plt.tight_layout()
    plt.savefig("Clusters.svg", bbox_inches='tight')

def DistributionPlot():
    plt.figure(figsize=(7, 6))
    for row, (filename, score) in enumerate(zip(["pitch.mat", "tda.mat"], ["DAPS", "Sw1PerS"])):
        res = sio.loadmat(filename)
        X, allndivs = res["X"], res["allndivs"].flatten()
        divgroups = [[0], [1], [2], [3], [4], [5]]
        legend = ["0", "1", "2", "3", "4", "5"]
        # Step 1: Plot periodicity distributions
        plt.subplot(2, 2, row*2 + 1)
        xs = []
        for ndivs in divgroups:
            idx = np.zeros(X.shape[0])
            for div in ndivs:
                idx[allndivs == div] = 1
            scores = X[idx == 1, :]
            sns.distplot(scores.flatten())
            xs.append(scores.flatten())
        plt.legend(legend)
        plt.title("%s Block Distributions"%score)
        plt.xlabel("Periodicity Score")
        plt.ylabel("Distribution Density")
        # Step 2: Compute and plot ROC scores
        D = np.zeros((len(divgroups), len(divgroups)))
        for i, divi in enumerate(legend):
            x = xs[i]
            for j in range(i+1, len(divgroups)):
                divj = legend[j]
                y = xs[j]
                D[i, j] = getAUROC(x, y)['auroc']
                print("%s %s divisions vs %s divisions: %.3g"%(score, divi, divj, D[i, j]))
        D += D.T
        np.fill_diagonal(D, 0.5)
        plt.subplot(2, 2, row*2+2)
        plt.imshow(D, cmap='magma', vmin=0.5, vmax=1.0)
        plt.xticks(np.arange(D.shape[0]), legend)
        plt.yticks(np.arange(D.shape[0]), legend)
        plt.xlabel("Number of Divisions")
        plt.ylabel("Number of Divisions")
        plt.title("%s AUROC Scores"%score)
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if i == j:
                    continue
                color = 'black'
                if D[i, j] < 0.75:
                    color = 'white'
                plt.text(j-0.45, i+0.2, "%.2g"%D[i, j], color=color)
    plt.tight_layout()
    plt.savefig("DistributionPlots.svg", bbox_inches='tight')

def ROCExample():
    res = sio.loadmat("pitch.mat")
    X, allndivs = res["X"], res["allndivs"].flatten()
    div1 = 1
    div2 = 4
    legend = ["%i Division"%div1, "%i Divisions"%div2]
    X1 = X[allndivs == div1, :]
    X2 = X[allndivs == div2, :]

    plt.figure(figsize=(10, 3))
    plt.subplot(121)
    sns.distplot(X1.flatten())
    sns.distplot(X2.flatten())
    plt.legend(legend)
    plt.title("DAPS Block Distributions")
    plt.xlabel("Periodicity Score")
    plt.ylabel("Distribution Density")
    # Step 2: Compute and plot ROC scores
    plt.subplot(122)
    auroc = getAUROC(X1.flatten(), X2.flatten(), do_plot=True)['auroc']
    plt.plot([0, 1], [0, 1], linestyle=':', linewidth=1, color='k')
    plt.title("ROC Curve, AUROC = %.3g"%auroc)
    plt.savefig("ROCExample.svg", bbox_inches='tight')


def DetrendingExample():
    """
    To help Mahdi with a block diagram of our procedure
    """
    X, ndivs = get0GrayData()
    X = X[ndivs == 0, :]
    t = np.arange(X.shape[1])*0.5
    
    i = 6
    Win = 11
    block_hop = 1 # Maximum data augmentation is to take all blocks
    block_len = 48

    plt.clf()
    x = X[i, :]
    y = detrend_timeseries(x, Win)
    score_fn = lambda xk: pitch_detection(xk)[1]

    scores = []
    i1 = 0
    if not block_len:
        block_len = x.size
    while i1+block_len <= y.size: # Loop over all blocks
        print("%i %i"%(i, i1))
        # Pull out block and do sliding window
        yk = np.array(y[i1:i1+block_len])
        score = score_fn(yk)
        scores.append(score)
        i1 += block_hop
    scores = np.array(scores)

    plt.figure(figsize=(12, 8))
    plt.subplot(311)
    plt.plot(t, x)
    plt.xlim([0, 120])
    plt.xlabel("Time (Hours)")
    plt.subplot(312)
    plt.plot(t, y)
    plt.xlim([0, 120])
    plt.title("Detrended")
    plt.xlabel("Time (Hours)")
    plt.subplot(313)
    plt.title("Scores")
    plt.plot(t[0:scores.size], scores)
    plt.xlim([0, 120])
    plt.tight_layout()

    plt.savefig("Detrending%i.svg"%i)

def DAPSConceptFigure():
    """
    An example of a linearly drifting and linearly amplitude modulated
    signal with oscillations hidden inside of it, which allows us
    to show all of the steps of detrending in DAPS
    """
    N = 50
    Win = 5
    x = np.arange(N)*((-1)**(np.arange(N)) + np.arange(N))
    X = getSlidingWindowNoInterp(x, Win)
    XPC = X - np.mean(X, 1)[:, None]
    Norm = np.sqrt(np.sum(XPC**2, 1))
    Norm[Norm == 0] = 1
    XPCSN = XPC/Norm[:, None]
    y = getSlidingWindowL2Inverse(XPCSN)
    Y = getSlidingWindowNoInterp(y, Win)
    
    plt.figure(figsize=(8, 8))
    plt.subplot(511)
    plt.plot(x)
    plt.title("Original Time Series $x$")
    plt.subplot(512)
    plt.imshow(X.T, aspect='auto', interpolation='nearest')
    plt.axis('off')
    plt.colorbar()
    plt.title("Sliding Window Matrix $X$")
    plt.subplot(513)
    plt.imshow(XPCSN.T, aspect='auto', interpolation='nearest')
    plt.axis('off')
    plt.colorbar()
    plt.title("Point-Centered/Sphere Normalized Matrix $\hat{X}$")
    plt.subplot(514)
    plt.imshow(Y.T, aspect='auto', interpolation='nearest')
    plt.axis('off')
    plt.colorbar()
    plt.title("Projected Hankel Matrix $Y$")
    plt.subplot(515)
    plt.plot(y)
    plt.tight_layout()
    plt.title("Detrended Time Series $y$")
    plt.savefig("DAPSConcept.svg", bbox_inches='tight')

if __name__ == '__main__':
    #PeriodicityScoringExamples()
    #BlockConcept()
    #BlockDistributions()
    #ROCFigures()
    #ClusterPlots()
    #DistributionPlot()
    #ROCExample()
    #DetrendingExample()
    DAPSConceptFigure()