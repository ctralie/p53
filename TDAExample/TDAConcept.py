import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import scipy.io as sio
from sklearn.decomposition import PCA
from ripser import ripser
from persim import plot_diagrams as plot_dgms
import scipy
from matplotlib.patches import Polygon
import sys 
import time

def TDAConceptFigure():
    X = scipy.misc.imread("ExampleShape.png")
    X = X[:, :, 0:3]
    X = np.sum(X, 2)
    N = X.shape[0]
    pix = np.arange(N)
    I, J = np.meshgrid(pix, pix)
    I = I[X == 0]
    J = J[X == 0]
    X = np.array([I.flatten(), J.flatten()], dtype=float).T
    X /= float(N)
    I = scipy.misc.imread("ExampleShape_Filled.png")

    XSqr = np.sum(X**2, 1)
    DSqr = XSqr[None, :] + XSqr[:, None] - 2*X.dot(X.T)

    N = X.shape[0]
    dgms = ripser(X)['dgms']
    print(dgms[1])
    scales = [0.072, 0.115, 0.14, 0.26]
    scalesplot = scales 
    NS = len(scales)+2
    res = 3.5
    plt.figure(figsize=(NS*res/2, res*2))

    plt.subplot(2, NS/2, 1)
    plt.imshow(I)
    plt.scatter(I.shape[0]*X[:, 0], I.shape[0]*X[:, 1])
    plt.xlim([0, 160])
    plt.ylim([-10, 210])
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')
    plt.title("Original Shape / Samples")


    for s, scale in enumerate(scales):
        plt.subplot(2, NS/2, s+2)
        for i in range(N):
            for j in range(i+1, N):
                if DSqr[i, j] <= scale**2:
                    plt.plot(X[[i, j], 0], X[[i, j], 1], 'k')
        patches = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    if DSqr[i, j] <= scale**2 and DSqr[i, k] <= scale**2 and DSqr[j, k] <= scale**2:
                        patches.append(Polygon(X[[i, j, k], :]))
        ax = plt.gca()
        p = PatchCollection(patches, alpha=0.2, facecolors='C1')
        ax.add_collection(p)
        plt.scatter(X[:, 0], X[:, 1], c='C0', zorder=10)
        plt.axis('equal')
        plt.title("$\\alpha$ = %.3g"%scale)
        ax.invert_yaxis()
    plt.subplot(2, NS/2, NS)
    for scale in scalesplot:
        plt.plot([-0.01, scale], [scale, scale], 'gray', linestyle='--', linewidth=1, zorder=0)
        plt.plot([scale, scale], [scale, 0.46], 'gray', linestyle='--', linewidth=1, zorder=0)
        plt.text(scale+0.01, scale-0.01, "%.3g"%scale)
    plot_dgms(dgms, size=30)
    I = dgms[1]
    I = I[I[:,1]-I[:, 0] > 0.05, :]
    plt.scatter(I[:, 0], I[:, 1], 100, marker='x', c='C1')
    I = dgms[0]
    I = I[I[:, 1] - I[:, 0] > 0.2, :]
    I = I[np.isfinite(I[:, 1]), :]
    plt.scatter(I[:, 0], I[:, 1], 100, marker='x', c='C0')
    plt.xlim([-0.01, 0.3])
    plt.ylim([0, 0.3])
    plt.title("Persistence Diagram")
    plt.savefig("TDAExample.svg", bbox_inches='tight')

if __name__ == '__main__':
    TDAConceptFigure()