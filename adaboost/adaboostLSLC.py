import numpy as np
from numpy.random import choice
from leastSquares import leastSquares

def adaboostLSLC(X, Y, K, nSamples):
    # Adaboost with least squares linear classifier as weak classifier
    # for a D-dim dataset
    #
    # INPUT:
    # X         : the dataset (numSamples x numDim)
    # Y         : labeling    (numSamples x 1)
    # K         : number of weak classifiers (iteration number of Adaboost) (scalar)
    # nSamples  : number of data which are weighted sampled (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (K x 1)
    # para      : parameters of least square classifier (K x 3) 
    #             For a D-dim dataset each least square classifier has D+1 parameters
    #             w0, w1, w2........wD

    #####Start Subtask 1e#####
    N, D = X.shape
    alphaK = np.zeros(K)
    para = np.ndarray((K, D+1))
    W = (1/N*np.ones(N)).reshape(N,1)

    for k in range(K):
        # Train classifier
        sIdx = choice(N, nSamples, True, W.ravel())

        weight, bias = leastSquares(X[sIdx, :], Y[sIdx])

        para[k, :] = [bias, weight[0], weight[1]]

        # Calculate labeled classification vector
        C = np.sign(X.dot(weight) + bias).reshape(N,1)

        C = C*Y
        mask = [item[0] for sublist in [C < 0] for item in sublist]
        # Compute weighted error of classifier
        I = np.zeros(N)
        I[mask] = 1
        epsilon = max(I.dot(W), 0.001)

        # Calculate voting weight
        alpha = 0.5*np.log((1-epsilon)/epsilon)
        alphaK[k] = alpha

        # Update weights and normalize
        W = W*np.exp((-alpha)*C)
        W = W/sum(W)
    #####End Subtask#####

    return [alphaK, para]
