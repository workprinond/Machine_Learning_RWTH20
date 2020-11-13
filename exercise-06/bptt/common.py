import numpy as np


def softmax(data):
    """
    compute the safe softmax of a given numpy matrix in a row wise manner

    :param data: dims = (nSamples, nVocab) input tensor
    :return: dims = (nSamples, nVocab) output tensor
    """
    output = np.zeros_like(data)

    for i, col in enumerate(data):
        max_elem = np.max(col)
        col = np.exp(col - max_elem)
        output[i] = col / np.sum(col)
    return output


def cross_entropy_loss(output, target):
    """
    Cross entropy loss function with output interpreted as probability and target as correct class labels

    :param output: dims = (nVocab x 1)
    :param target: dims = (nVocab x 1)
    :return: cross entropy loss
    """
    return np.squeeze(-np.log(output[target == 1]))

def clip_gradient(grad, threshold):
    """
    Clip gradient grad according to threshold

    :param grad: parameter gradients
    :param threshold: scalar threshold
    :return:
    """
    #####Start Subtask 1c#####
    norm = np.linalg.norm(grad)
    if norm > threshold:
        grad /= norm/5
    #####End Subtask#####
    pass
