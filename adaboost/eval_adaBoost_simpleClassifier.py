import numpy as np


def eval_adaBoost_simpleClassifier(X, alphaK, para):
    # INPUT:
    # para	: parameters of simple classifier (K x 2) - for each round
    #           : dimension 1 is j
    #           : dimension 2 is theta
    # alphaK    : classifier voting weights (K x 1)
    # X         : test data points (numSamples x numDim)
    #
    # OUTPUT:
    # classLabels: labels for data points (numSamples x 1)
    # result     : weighted sum of all the K classifier (numSamples x 1)

    #####Start Subtask 1c#####
    K = para.shape[0]  # number of classifiers
    N = X.shape[0]  # number of test points
    result = np.zeros(N)  # prediction for each test point

    for k in range(K):
        # Initialize temporary labels for given j and theta
        cY = np.ones(N) * (-1)
        # Classify
        cY[X[:, int(para[k, 0]-1)] > para[k, 1]] = 1

        result += alphaK[k] * cY  # update results with weighted prediction

    classLabels = np.sign(result)  # class-predictions for each test point

    #####End Subtask#####
    return classLabels, result
