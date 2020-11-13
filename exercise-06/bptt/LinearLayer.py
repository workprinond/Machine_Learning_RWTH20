import numpy as np
import common as cm

class LinearLayer:
    """
    Simple linear fully connected layer without any fancy stuff
    """
    def __init__(self, n_in, n_out):
        """
        Initilize a linear layer with all necessary variables

        :param n_in: length of the input vectors (usually nHidden)
        :param n_out: length of the output vectors (usually nVocab)
        """
        self.n_in = n_in
        self.n_out = n_out

        # Glorot initialization of weights and zero initialization for derivatives
        he_init = np.sqrt(2/(n_in + n_out))
        self.W = np.random.normal(0, he_init, size=(n_out, n_in))
        self.dW = np.zeros_like(self.W)
        self.b = np.zeros(shape=(n_out, 1))
        self.db = np.zeros_like(self.b)
        self.dxs = []
        self.xs = []

    def __str__(self):
        """
        Function for basic string representation of the class

        :return: class name as string
        """
        return "LinearLayer"

    def fprop(self, data):
        """
        Simple forward propagation of the layer

        :param data: dims = (nSamples, nHidden) input vector from the RNN cell
        :return: dims = nSamples x (nVocab, 1) output vector containing logits for the characters of the vocab
        """
        output = []
        # Extract data dimensions
        n_samples, n_in = data.shape
        for i in range(n_samples):
            # Append the input vector for later reusability
            self.xs.append(data[i:i + 1].T)

            # Append the output
            output.append(np.dot(self.W, self.xs[-1]) + self.b)

        return np.squeeze(np.array(output))

    def bprop(self, dys):
        """
        Simple backward propagation of a linear layer

        :param dys: dims = nSamples x (nVocab, 1) output gradient
        :return: dxs with dims = nSamples x (nHidden, 1) gradient of output wrt to input
        """

        for dy in reversed(dys):
            self.dW += np.dot(dy, self.xs[-1].T)
            self.db += dy
            self.dxs.insert(0, np.dot(self.W.T, dy))

            # Remove stored x values one by one
            self.xs.pop()

        return self.dxs

    def clip_gradients(self):
        """
        Clip the gradients so that the norm remains in the order of [-1,1] and cannot explode

        :return: None
        """
        for grad in [self.dW, self.db]:
            threshold = 5
            cm.clip_gradient(grad, threshold)

    def update_weights(self, update_fun=None, *, eta=0.001):
        """
        Update the weights according to some update rule. If no update rule is supplied, simple gradient descent is used

        :param update_fun: function which takes two parameters (weight, derivative) and computes the updated weight
        as return value.
        :param eta: learning rate
        :return: None
        """

        if update_fun is None:
            def update_fun(x, y):
                return -eta*y
        for weight, der in zip([self.W, self.b], [self.dW, self.db]):
            weight += update_fun(weight, der)
            der *= 0
        self.dxs = []

    def get_params(self):
        """
        Generator for returning parameter and parameter derivative pair

        :return: tuple in the form (param, dparam)
        """
        for weight, der, name in zip([self.W, self.b], [self.dW, self.db], ["Linear W", "Linear b"]):
            yield (weight, der, name)

    def clear_stored_states(self):
        """
        Helper function that removes all previously stored hidden states

        :return:
        """
        self.xs = []
        self.dxs = []

    def clear_stored_derivs(self):
        """
        Function that removes all previously stored derivatives

        :return: None
        """
        for _, der, _ in self.get_params():
            der *= 0
