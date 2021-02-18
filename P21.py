import numpy as np
import matplotlib.pyplot as plt


def getAllP21Means():
    X = np.array([])
    grays = np.array([])
    allndivs = np.array([])
    for t in [0, 2, 4, 10]:
        scores = np.loadtxt("data/p53_%iGy_scores.txt"%t)
        divs = np.loadtxt("data/div_%iGy.txt"%t)
        ndivs = np.sum(divs, 1)
        valid = np.sum(np.isnan(scores), 1) == 0
        p21vals = np.loadtxt("data/p21_%iGy.txt"%t)
        p21vals = p21vals[valid == 1]
        ndivs = ndivs[valid == 1]
        p21vals = np.sort(p21vals, 1)
        idx = np.argsort(ndivs)
        ndivs = ndivs[idx]
        p21vals = p21vals[idx, :]
        if X.size == 0:
            X = p21vals
        else:
            X = np.concatenate((X, p21vals), 0)
        grays = np.concatenate((grays, np.array([t]*p21vals.shape[0])))
        allndivs = np.concatenate((allndivs, ndivs))
    # Take the area under the curve (also the mean since they're all the same length)
    return np.mean(X, 1)
    # Do the Wasserstein distance
    #return getWasserstein(X)

if __name__ == '__main__':
    x = getAllP21Means()
    plt.plot(x)
    plt.show()