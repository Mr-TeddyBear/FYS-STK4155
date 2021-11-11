import numpy as np

from sklearn.util import resample


class layer():
    def __init__(self, input_nodes, weigths, bias, activation):
        self.input_nodes = input_nodes
        self.weights = weigths
        self.bias = bias
        self.activation = activation

    def __call__(self, X):
        """
        Calcualtes dot product for this layer
        """
        ldot = np.dot(X, self.weights) + self.bias
        return self.activation(ldot)


class FFNnetwork():
    """
    Need:
        input layer
        Hidden layer(s)
        Output layer

    Inputlayer has same ammout of neurons as pixel in image

    ActivationFunction, input and output


    """

    def __init__(self, dataX, dataY, n_hidden_nodes, n_cat, epochs, batch_size, eta, lmbd):
        self.dataX, self.dataY = dataX, dataY

        self.n_inpt, self.n_feat = dataX.shape
        self.n_hidden_nodes = n_hidden_nodes
        self.n_cat = n_cat

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inpt // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.generate_biases_and_weights()

    def sigmoid(self, X):
        """
        Sigmoid activation function
        """
        return 1/(1 + np.exp(-X))

    def generate_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_feat, self.n_hidden_nodes)
        self.hidden_bias = np.zeros(self.n_hidden_nodes) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_nodes, self.n_cat)
        self.output_bias = np.zeros(self.n_cat) + 0.01

    def feed_forward_loop(self, X):
        # input to hidden layers
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        # calculate activation in hidden layers
        a_h = self.sigmoid(z_h)
        # calculate for outputlayer
        z_o = np.matmul(a_h, self.output_weights) + self.output_bias

        # softmax
        exp_term = np.exp(z_o)

        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_end(self, X):
        # input to hidden layers
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        # calculate activation in hidden layers
        self.a_h = self.sigmoid(z_h)
        # calculate for outputlayer
        z_o = np.matmul(a_h, self.output_weights) + self.output_bias

        # softmax
        exp_term = np.exp(z_o)

        return exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def backpropagation(self):
        error_output = self.probabilities - self.dataY
        error_hidden = self.np.matmul(
            error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.dataX.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd + self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    @property
    def get_prob(self):
        return self.probabilities

    @property
    def get_predictor(self):
        return np.argmax(self.probabilities, axis=1)

    def train(self):
        data_indices = np.arrange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                self.batch_dataX, self.batch_dataY = resample(
                    self.dataX, self.dataY, replace=False)
