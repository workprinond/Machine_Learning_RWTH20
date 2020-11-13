from train import *
from test import *
from BasicRNNCell import *
from LinearLayer import *

import argparse

if __name__ == "__main__":
    # Construct command line argument parser
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--task", type=int, default=1, help="Task number")
    parser.add_argument("--nbr_iters", type=int, default=70000, help="Number of training iterations for task 1f)")
    parser.add_argument("--lr", type=float, default=0.001, help="(Initial) learning rate")
    parser.add_argument("--sz_rnn", type=int, default=50, help="RNN size")
    parser.add_argument("--sz_vocab", type=int, default=10, help="Vocabulary size")
    FLAGS, unparsed = parser.parse_known_args()

    # Case for Questions 1a) and 1b): Forward propagation and backward propagation through time
    if FLAGS.task == 1:
        # Assign variables
        n_hidden = FLAGS.sz_rnn
        n_vocab = FLAGS.sz_vocab
        n_samples = 3
        n_checks = 5

        # Construct an RNN consisting of one BasicRNNCell and one Linear layer
        # Dimensions between layers must match
        net = [BasicRNNCell(n_vocab, n_hidden), LinearLayer(n_hidden, n_vocab)]

        # Generate random data which is passed through the network for gradient checking
        data, labels = dg.generate_random_data(n_samples, n_vocab)

        # Run gradient checking routine
        run_check_grads(net, data, labels, n_checks)

    # Case for Question 1c)d): Exploding gradients and gradient clipping
    elif FLAGS.task == 2:
        # Assign variables
        n_hidden = 10
        max_seq_len = 1000
        n_vocab = FLAGS.sz_vocab
        n_pretrain = 10

        # Construct an RNN consisting of one BasicRNNCell and one Linear layer
        net = [BasicRNNCell(n_vocab, n_hidden), LinearLayer(n_hidden, n_vocab)]
        data, labels = dg.generate_random_data(max_seq_len, n_vocab)

        run_exploding_grads_test(net, data, labels, max_seq_len, n_pretrain)

    # Case for Question 1d): Memorization task
    elif FLAGS.task == 3:
        # Assign variables
        learn_rate = FLAGS.lr
        n_hidden = FLAGS.sz_rnn
        n_vocab = FLAGS.sz_vocab
        n_iters = FLAGS.nbr_iters
        to_remember_len = 13
        delim_idx = n_vocab-2
        blank_idx = n_vocab-1

        # Construct an RNN consisting of two BasicRNNCells and one Linear layer (Remember to match dimensions)
        net = [BasicRNNCell(n_vocab, n_hidden), LinearLayer(n_hidden, n_vocab)]

        # Run training
        run_train_memory(net, to_remember_len, n_vocab, n_iters, delim_idx, blank_idx, learn_rate)
