import data_generator as dg
import train as tr
import common as cm
from BasicRNNCell import *
from LinearLayer import *

import numpy as np
import matplotlib.pyplot as plt


def run_exploding_grads_test(net, data, labels, max_seq_len, n_pretrain):
    """
    Visualize the increase in the norm of the gradient when gradient clipping or similar mitigation schemes are not
    applied. This will only be the case when gradient clipping is not yet implemented. This function will use gradient
    clipping as soon as its implemented and may be used for verification.

    :param net: list of layers in the network
    :param data: dims = (maxSeqLen, nVocab) input data
    :param labels: dims = (maxSeqLen, nVocab) labels corresponding to inputs
    :param max_seq_len: length of the longest sequence for which the gradient should be computed
    :param n_pretrain: number of pretraining iterations on network, so that the exploding gradient becomes more
    realistic
    :return: show the plot of the gradient's magnitude against the sequence length
    """

    # Run pretraining
    n_seq_pretrain = 3
    print(f"Pre-training for {n_pretrain} iterations...")
    for i in range(n_pretrain):
        pre_data, pre_labels = dg.generate_random_data(n_seq_pretrain, data.shape[1])
        tr.run_training(net, pre_data, pre_labels, clip_gradients=False)

    # Empty array that is used for storing the magnitudes of the computed analytical derivatives
    grads = []

    # Testing exploding gradients
    print(f"Computing plot...")
    for i in range(1, max_seq_len):
        _ = tr.run_training(net, data[0:i + 1], labels[0:i + 1], clip_gradients=True, update_weights=False)
        grads_tmp = []
        for layer in net:
            grads_tmp.append(np.concatenate([grad.flatten() for _, grad, _ in layer.get_params()]))
            layer.clear_stored_states()
            layer.clear_stored_derivs()
        grads_tmp = np.abs(np.concatenate(grads_tmp))
        grads.append(np.mean(grads_tmp))

    grads = np.array(grads)
    plt.figure()
    plt.title("Visualization of exploding gradients")
    plt.loglog(range(1, max_seq_len), grads)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Mean Absolute Derivative")
    plt.show()


def run_memory_test(net, to_remember_len, n_vocab, n_acc_iters, delim_idx, blank_idx):
    """
    Test the network for accuracy on previously unseen data.

    :param net: The RNN as a list of layers
    :param to_remember_len: length of sequence to remember
    :param n_vocab: number of possible vocabulary elements
    :param n_acc_iters: Number of iterations for which to average the accuracy measure
    :param delim_idx: index of the delimiter
    :param blank_idx: index of the blank sign
    :return: average loss value
    """
    acc = 0
    for i in range(n_acc_iters):
        #####Start Subtask 1e#####
        data, labels = dg.generate_data_memory(to_remember_len, n_vocab, delim_idx, blank_idx)
        loss, logits = run_forward_pass(net, data, labels, clear_after=True)
        output_cls = np.argmax(logits, axis=1)[-to_remember_len:]
        label_cls = np.argmax(labels, axis=1)[-to_remember_len:]
        acc += np.sum(output_cls == label_cls) / to_remember_len
        #####End Subtask 1e#####
        pass
    acc /= n_acc_iters
    print(f"Validation set size {n_acc_iters}, Avg. acc. on validation set {acc}")


def run_forward_pass(net, data, labels, clear_after=True):
    """
    Do a single forward pass without backpropagation

    :param data: dims = (nSamples, nVocab) input data
    :param labels: dims = (nSamples, nVocab) labels corresponding to inputs
    :param clear_after: bool indicator whether or not to clear the stored states in the layers after finishing run
    :return: loss, logits
    """
    # Iterate over all layers of the RNN
    output = data.copy()
    for layer in net:
        output = layer.fprop(output)

    # Compute the column-wise softmax of the output
    logits = cm.softmax(output)

    # Compute cross entropy loss
    loss = np.sum(cm.cross_entropy_loss(logits, labels))

    # Clear any generated variables stored for backpropagation if the flag is set
    if clear_after:
        for layer in net:
            layer.clear_stored_states()

    # Return total loss and softly classified samples
    return loss, logits


def run_check_grads(net, data, labels, num_checks):
    """
    Check the gradient computed by the backpropagation algorithm using finite differences

    :param net: list of layers in the network
    :param data: dims = (nSamples, nVocab) input data
    :param labels: dims = (nSamples, nVocab) labels corresponding to inputs
    :param num_checks: number of checks to perform per each parameter tensor
    :return: prints out the parameter name and the error associated with randomly selected values
    """
    # Delta for numerical difference quotient
    delta = 1e-7

    # One complete pass across the data to obtain the analytical derivatives without updating the weights, so that
    # the network is not changed
    tr.run_training(net, data, labels, clip_gradients=False, update_weights=False)

    grad_err_found = False
    # Compute num_checks numerical derivatives for each layer in the RNN
    for layer in net:
        for param, grad, name in layer.get_params():
            print(f"Working on {layer} and parameter: {name}")
            size_param = param.size
            n = 0
            while n < num_checks:
                # Select a random parameter for which to calculate the derivative
                select = int(np.random.uniform(0, size_param))

                # Select the analytical derivative
                d_analytical = grad.flat[select].copy()

                # Compute the numerical derivative
                orig_val = param.flat[select]
                param.flat[select] = orig_val + delta
                L1, _ = run_forward_pass(net, data, labels, clear_after=True)  # loss with positive disturbance
                param.flat[select] = orig_val - delta
                L2, _ = run_forward_pass(net, data, labels, clear_after=True)  # loss with negative disturbance
                param.flat[select] = orig_val  # reset the parameter to the original value
                d_numerical = (L1 - L2) / (2 * delta)  # compute the numerical derivative
                if d_numerical == 0:
                    continue

                # Compute and print absolute and relative errors
                abs_error = abs(d_analytical - d_numerical)
                rel_error = abs_error / abs(d_numerical)
                # print(f"Numerical Grad {d_numerical}, Analytic Grad {d_analytical}")
                print(f"Absolute error {abs_error}, Relative Error {rel_error}")
                if abs_error > 1e-7 or rel_error > 0.005:
                    grad_err_found = True
                    print("WARNING: Gradient seems to be wrong")
                n += 1
    print("------------------")
    print("------------------")
    if grad_err_found:
        print("WARNING: Gradient seems to be wrong!")
    else:
        print("Gradient seems to be right! Enjoy!")
    print("------------------")
    print("------------------")
