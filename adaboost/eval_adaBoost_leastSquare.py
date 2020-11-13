import numpy as np

def eval_adaBoost_leastSquare(X, alphaK, para):
    # INPUT:
    # para		: parameters of simple classifier (K x (D +1)) 
    #           : dimension 1 is w0
    #           : dimension 2 is w1
    #           : dimension 3 is w2
    #             and so on
    # alphaK    : classifier voting weights (K x 1)
    # X         : test data points (numSamples x numDim)
    #
    # OUTPUT:
    # classLabels: labels for data points (numSamples x 1)
    # result     : weighted sum of all the K classifier (scalar)

    #####Start Subtask 1e#####
    K = para.shape[0]
    N = X.shape[0]
    result = np.zeros(N)

    for k in range(K):
        cY = np.sign(np.append(np.ones(N).reshape(N,1), X, axis = 1).dot(para[k])).T
        result = result + cY.dot(alphaK[k])

    classLabels = np.sign(result)
    #####End Subtask#####

    return [classLabels, result]

