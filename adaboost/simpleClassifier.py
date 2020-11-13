import numpy as np


def simpleClassifier(X, Y):
    # Select a simple classifier
    #
    # INPUT:
    # X         : training examples (numSamples x numDim)
    # Y         : training lables (numSamples x 1)
    #
    # OUTPUT:
    # theta 	: threshold value for the decision (scalar)
    # j 		: the dimension to "look at" (scalar)

    #####Start Subtask 1b#####
    N, D = X.shape

    # Initialize least error
    le = 1
    j = 1  # dimension
    theta = 0  # decision value
    
    # Iterate over dimensions, which j to choose
    for jj in range(D):

        # Find interval to choose theta 
        val = X[:, jj]  # shape: (100 x 1)

        sVal = np.sort(val) # TODO: returns unique sorted values, shape: (100 x 1)
        idx = np.argsort(val)
        change = np.where(np.roll(Y[idx], -1) + Y[idx] == 0)[0]  # shape: (36 x 1)

        # Calculate thresholds for which we want to check classifier error.
        # Candidates for theta are always between two points of different classes.

        th = (sVal[change[change < len(X)-1]] + sVal[change[change < len(X)-1]+1])/2  # shape: (35 x 1)

        error = np.zeros(len(th))  # error-placeholder for each value of threshold th

        # Iterate over canidates forfor theta
        for t in range(len(th)):
            # Initialize temporary labels for given j and theta
            cY = np.ones(N)*(-1)  # shape: (100 x 1)
            
            # Classify
            cY[X[:, jj] > th[t]] = 1  # Set all values to one, which are bigger then current threshold

            # Calculate error for given classifier
            error[t] = sum([Y[i] != cY[i] for i in range(N)])
            
            # Visualize potential threshold values
            print('J = {0} \t Theta = {1} \t Error = {2}\n'.format(jj, th[t], error[t]))

        le1 = min(error/N)
        ind1 = np.argmin(error/N)
        le2 = min(1-error/N)
        ind2 = np.argmin(1-error/N)
        le0 = min([le1,le2,le])

        if le == le0:
            continue
        else:
            le = le0
            j=jj+1  # Set theta to current value of threshold
            # Choose theta and parity for minimum error
            if le1 == le:
                theta = th[ind1]
            else:
                theta = th[ind2]

    #####End Subtask#####
    return j, theta
