import numpy as np
from numpy.random import choice
from simpleClassifier import simpleClassifier
from eval_adaBoost_simpleClassifier import eval_adaBoost_simpleClassifier

def adaboostCross(X, Y, K, nSamples, percent):
    # Adaboost with an additional cross validation routine
    #
    # INPUT:
    # X         : training examples (numSamples x numDims )
    # Y         : training lables (numSamples x 1)
    # K         : number of weak classifiers to select (scalar)
    #             (the _maximal_ iteration count - possibly abort earlier)
    # nSamples  : number of training examples which are selected in each round. (scalar)
    #             The sampling needs to be weighted!
    # percent   : percentage of the data set that is used as test data set (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (K x 1)
    # para      : parameters of simple classifier (K x 2)
    # testX     : test dataset (numTestSamples x numDim)
    # testY     : test labels  (numTestSamples x 1)
    # error	    : error rate on validation set after each of the K iterations (K x 1)

    #####Start Subtask 1d#####
    # Randomly sample a percentage of the data as test data set
    N = len(X)
    numb = round(N * percent)
    pos = choice(N, numb, False)
    allpos = range(N)
    restpos = np.setdiff1d(allpos, pos)

    testX = X[pos]
    testY = Y[pos]
    X = X[restpos]
    Y = Y[restpos]

    # Initialization
    n = N - numb

    # Initialize the classifier models
    j = np.zeros(K) * (-1)
    theta = np.zeros(K) * (-1)

    alphaK = np.zeros(K)
    w = (np.ones(n) / n).reshape(n, 1)

    error = np.zeros(K)

    for k in range(K): # Iterate over all classifiers

        # Sample data with weights
        index = choice(n, nSamples, True, w.ravel())
        X_sampled = X[index, :]
        Y_sampled = Y[index]

        # Train the weak classifier C_k
        j[k], theta[k] = simpleClassifier(X_sampled, Y_sampled)

        cY = (np.ones(n) * (-1)).reshape(n,1)  # placeholder for class predictions
        cY[X[:, int(j[k]-1)] > theta[k]] = 1  # classify

        # Calculate weighted error for given classifier
        temp = np.where([Y[i] != cY[i] for i in range(n)], 1, 0).reshape(n, 1)
        ek = np.sum(w * temp)

        # If the error is zero, the data set is correct classified - break the loop
        if ek < 1.0e-01:
            alphaK[k] = 1
            break

        # Compute the voting weight for the weak classifier alpha_k
        alphaK[k] = 0.5 * np.log((1 - ek) / ek)

        # Update the weights
        w = w * np.exp((-alphaK[k] * (Y * cY)))
        w = w / sum(w)

        para = np.stack((j[:k+1], theta[:k+1]), axis=1)

        # Compute error for boosted classifier
        classlabels, _ = eval_adaBoost_simpleClassifier(testX, alphaK[:k+1], para[:k+1])
        classlabels = classlabels.reshape(len(classlabels), 1)

        error[k] = len(classlabels[classlabels != testY]) / len(testY)

    para = np.stack((j, theta), axis=1)

    #####End Subtask#####
    return alphaK, para, testX, testY, error

