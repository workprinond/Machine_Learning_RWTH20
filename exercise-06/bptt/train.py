from test import *
import data_generator as dg
import common as cm
from BasicRNNCell import *
from LinearLayer import *

import numpy as np


def run_training(net, data, labels, clip_gradients=True, update_weights=True, learning_rate=0.001):
    """
    Run the training routine on the network specified as a list of layers in net on data and labels

    :param clip_gradients:
    :param learning_rate:
    :param net: list of layers in the network
    :param data: dims = (nSamples, nVocab)
    :param labels: dims = (nSamples, nVocab)
    :param update_weights: bool to indicate if the gradients should be clipped and updated
    :return: total loss of the training step
    """

    # Iterate over all layers of the RNN
    output = data.copy()
    for layer in net:
        output = layer.fprop(output)

    # Compute the column-wise softmax of the output
    logits = cm.softmax(output)

    # Compute cross entropy loss
    loss = np.sum(cm.cross_entropy_loss(logits, labels))

    # Compute gradients from cross entropy loss and softmax
    dys = logits - labels
    dys = [np.expand_dims(dy, axis=1) for dy in dys]

    # Backprop into network
    for layer in reversed(net):
        dys = layer.bprop(dys)
        if clip_gradients:
            layer.clip_gradients()
        if update_weights:
            layer.update_weights(eta=learning_rate)

    return loss


def run_train_memory(net, to_remember_len, n_vocab, n_iters, delim_idx, blank_idx, learn_rate):
    """
    Run the memory training scheme described in exercise 1f). The data is generated in this method, while the network
    is passed as a parameter. After the training is complete, the average accuracy on a test set is printed out

    :param net: network as a list of layers
    :param to_remember_len: number of characters to remember
    :param n_vocab: number of vocabulary elements from which the characters are chosen
    :param n_iters: number of iterations to run the training procedure
    :param delim_idx: index for the delimiter character
    :param blank_idx: index for the blank character
    :param learn_rate: learning rate to apply to the training procedure, e.g. gradient descent
    :return: print out the loss every 1000 iterations and the accuracy at the end of the program
    """

    for i in range(n_iters):
        # Create random training data
        mem_data, mem_labels = dg.generate_data_memory(to_remember_len, n_vocab, delim_idx, blank_idx)

        # One training iteration over a random training data set
        loss = run_training(net, mem_data, mem_labels, learning_rate=learn_rate)

        # Print progress every 1000 iterations
        if i % 1000 == 0:
            print("Training iteration {}. Loss: {}".format(i, loss))

    # Run the accuracy test for the trained model
    n_acc_iters = int(n_iters/100)
    run_memory_test(net, to_remember_len, n_vocab, n_acc_iters, delim_idx, blank_idx)
