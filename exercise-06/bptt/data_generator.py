import numpy as np


def generate_data_memory(to_remember_len, vocab_size, delim_idx, blank_idx):
    """
    Generate the data for the memory test

    :param to_remember_len: length of the sequence
    :param vocab_size: number of different vocabulary items excluding the delimiter
    :param delim_idx: index of the delimiter that indicates the end of the sequence
    :param blank_idx: index of the blank that indicates an empty character
    :return: data numpy array
    """
    # Construct an index array for all the admissible characters in the vocabulary
    idx = np.random.randint(0, vocab_size - 2, to_remember_len)

    data_idx = np.concatenate((idx, [delim_idx], blank_idx*np.ones_like(idx)))
    label_idx = np.concatenate((blank_idx*np.ones_like(idx), [delim_idx], idx))
    # Construct an empty data array of the correct dimensions D = (nLetters x nUniqueCharacters)
    data = np.zeros(shape=(2*to_remember_len+1, vocab_size))
    labels = np.zeros(shape=(2*to_remember_len+1, vocab_size))
    # Convert into one-of-k/ one-hot-encoding in order to be readable by the RNN
    data[range(2*to_remember_len+1), data_idx] = 1
    labels[range(2*to_remember_len+1), label_idx] = 1

    return data, labels


def generate_random_data(n_samples, n_vocab):
    """
    Generate a random dataset for usage in visualizing exploding gradients

    :param n_samples: number of characters in the dataset
    :param n_vocab: number of possible characters for the one hot encoding
    :return: data and labels as two separate numpy arrays
    data
    """

    # Construct an index array for all the admissible characters in the vocabulary
    idx = np.random.randint(0, n_vocab, n_samples)

    # Construct an empty data array of the correct dimensions D = (nLetters x nUniqueCharacters)
    data = np.zeros(shape=(n_samples - 1, n_vocab))
    labels = np.zeros(shape=(n_samples - 1, n_vocab))

    # Construct data and labels in one-hot-encoding, where the task is to predict the next character
    data[range(n_samples - 1), idx[:-1]] = 1
    labels[range(n_samples - 1), idx[1:]] = 1

    return data, labels
