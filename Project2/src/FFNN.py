import numpy as np

from sklearn.utils import resample

from src.utilFunctions import MSE


class layer():
    def __init__(self, input_nodes, weigths_bias, activation):
        self.input_nodes = input_nodes
        self.weights, self.bias = weigths_bias
        self.activation = activation

    def __call__(self, X):
        """
        Calcualtes dot product for this layer
        """
        ldot = np.matmul(X, self.weights) + self.bias
        self.layer_data = self.activation(ldot)
        return self.layer_data

    @property
    def output(self):
        return self.layer_data


class FFNNetwork():
    """
    Need:
        input layer
        Hidden layer(s)
        Output layer

    Inputlayer has same ammout of neurons as pixel in image

    ActivationFunction, input and output


    """

    def __init__(self, dataX, dataY, testX, testY, layers: list, activation_function="sigmoid"):
        self.dataX, self.dataY = dataX, dataY
        self.testX, self.testY = testX, testY

        self.n_inpt, self.n_feat = dataX.shape

        self._layers = []

        activation = getattr(self, activation_function)

        input_prev_layer = None

        for i, nodes in enumerate(layers):
            if i == 0:
                # Create the input layer
                self._layers.append(
                    layer(nodes,
                          self.generate_biases_and_weights(
                              self.n_feat, nodes),
                          activation))
            else:
                # Create all the hidden layers
                self._layers.append(layer(
                                    input_prev_layer,
                                    self.generate_biases_and_weights(
                                        input_prev_layer, nodes),
                                    activation))
            input_prev_layer = nodes

        print(dataY.shape)
        self._layers.append(
            layer(input_prev_layer, self.generate_biases_and_weights(input_prev_layer, dataY.shape[0]), activation))

    def sigmoid(self, X):
        """
        Sigmoid activation function
        """
        return 1/(1 + np.exp(-X))

    def generate_biases_and_weights(self, n_input, n_nodes):
        hidden_weights = np.random.randn(n_input, n_nodes)
        hidden_bias = np.zeros(n_nodes) + 0.01

        return hidden_weights, hidden_bias

    def feed_forward_loop(self, X):
        for layer in self._layers:
            X = layer(X)
        return X

    def backpropagation(self, X, Y, lrate=0.01, lamb=0):

        # calcualte error of output layer
        error_output_layer = self._layers[-1].layer_data - Y

        # create array to hold all errors
        errors = np.empty(len(self._layers)-1, dtype=np.ndarray)

        # set last elemtment of errors array to output error
        errors[-1] = error_output_layer

        # Loop to calculate error in all layers expect input
        for i in reversed(range(len(self._layers)-1)):
            # Get output from current layer
            layerD = self._layers[i].layer_data

            # calculate layer error and set in errors array
            layer_error = errors[-1] @ (self._layers[i +
                                        1].weights).T * layerD*(1-layerD)
            errors[i] = layer_error

        layer_input = X

        # mBatch size, all input from input layer is the batchsize, this will
        # also depend on batchsize given in train function
        m = X.shape[0]  # ?

        # Update weigths using SGD
        for error, (i, layer) in zip(errors, enumerate(self._layers)):
            weights_grad = layer_input.T @ error
            bias_grad = np.sum(error, axis=0)

            reg = lamb * layer.weights

            # Update layer weights
            layer.weights = layer.weights - lrate*(weights_grad + reg) / m
            layer.bias = layer.bias - lrate*bias_grad / m

            layer_input = layer.layer_data

    @property
    def get_prob(self):
        return self.probabilities

    @property
    def get_predictor(self):
        return np.argmax(self.probabilities, axis=1)

    def train(self, n_epochs=10, t: tuple = (5, 50), lrate=0.01, lamb=1, n_batches=1):
        batch_size = self.dataX.shape[0]//n_batches

        prev_accuracy = np.empty([n_epochs, n_batches])

        for i in range(n_epochs):
            for j in range(n_batches):
                batch_dataX, batch_dataY = resample(
                    self.dataX, self.dataY, replace=False, n_samples=batch_size)

                self.feed_forward_loop(batch_dataX)
                self.backpropagation(batch_dataX, batch_dataY, lrate, lamb)

                prev_accuracy[i, j] = self.accuracy(self.testX, self.testY)

        return prev_accuracy

    def pred(self, X):
        self.feed_forward_loop(X)
        return np.argmax(self._layers[-1].output, axis=1)

    def accuracy(self, X, y):
        y_pred = self.pred(X)
        print(y_pred.shape)

        return self._onehot_pred(y, y_pred)

    def _onehot_pred(self, trueY, predY):
        p = predY  # np.argmax(predY, axis=0)
        t = trueY  # np.argmax(trueY, axis=0)

        return np.sum(p == t)/len(predY)
