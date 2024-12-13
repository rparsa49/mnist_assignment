import dataclasses

import numpy as np

import warnings
warnings.filterwarnings("error")

def sigmoid(Z):
    """Perform the sigmoid function on each element of a vector."""
    Z = np.clip(Z, -500, 500)  # prevent floating-point overflows by clipping the input.
    return 1/(1+np.exp(-Z))

def sigmoid_prime(Z):
    """Derivative of the sigmoid function"""
    return sigmoid(Z)*(1-sigmoid(Z))

class NeuralNet:
    """
    A neural network that can be:
        - trained, modifying its weights, or
        - can be applied to get an output vector.
    """

    def __init__(self, nn_architecture):
        """
        Initialize the neural net with some initial weights based on the given topology.

        The architecture should be a list of layer specifications, where each layer spec is a dict
        with the following keys:

            - input_dim: (integer) the dimension of vectors input to this layer
            - output_dim: (integer) the dimension of vectors output by this layer
            - activation: (string) the name of the activation function to be used by nodes in this layer

        Each layer's input dimension should be the same as the preceding layer's output dimension.

        The input dimension of the first layer is the dimension of vectors input to the neural net.

        The output dimension of the last layer is the dimension of vectors output by the neural net.

        You should support at least "sigmoid" as an activation function, but you may find it worthwhile
         to support and try out other functions such as "relu", "softmax", etc.

        If input and output dimensions of consecutive layers don't match up, or if an unsupported
        activation function is requested for any layer, initialization should fail.
        """
        self.layers = []
        self.weights = []
        self.biases = []
        self.activations = {"sigmoid": sigmoid}
        

        for i, layer in enumerate(nn_architecture):
            if "input_dim" not in layer or "output_dim" not in layer or "activation" not in layer:
                raise ValueError(f"Invalid layer specification: {layer}")
            if layer["activation"] not in self.activations:
                raise ValueError(f"Unsupported activation function: {layer['activation']}")
            if i > 0 and nn_architecture[i-1]["output_dim"] != layer["input_dim"]:
                raise ValueError("Input and output dimensions of consecutive layers do not match.")

            # Initialize weights and biases
            input_dim = layer["input_dim"]
            output_dim = layer["output_dim"]
            self.weights.append(np.random.randn(output_dim, input_dim) * 0.01)
            self.biases.append(np.zeros((output_dim, 1)))
            self.layers.append(layer)

    def apply(self, x):
        """
        Given a single column vector, x, as input, run a forward propagation of x through the net.

        The dimension of x should be the same as the input dimension of the first layer.

        Return the output vector of the last layer at the end of the forward propagation.
        """
        if x.shape[0] != self.layers[0]["input_dim"]:
            raise ValueError("Input dimension does not match the first layer.")
        current_activation = x
        for i, layer in enumerate(self.layers):
            Z = np.dot(self.weights[i], current_activation) + self.biases[i]
            activation_func = self.activations[layer["activation"]]
            current_activation = activation_func(Z)
        return current_activation

    def train_batch(self, X, Y, learning_rate):
        """
        Execute one forward-and-backward propagation to train the network using the given
        input and label vectors.

        X is a matrix created by horizontally concatenating one or more input (column) vectors for
        the net.

        Y is a matrix created by horizontally concatenating one or more label (column) vectors for
        the net.

        This method updates the weights of the network, but does not return anything.
        """
        # Forward propagation
        A = [X] # Store activations for each layer
        Z = [] # Store pre-activation values
        for i, layer in enumerate(self.layers):
            Z_current = np.dot(self.weights[i], A[-1]) + self.biases[i]
            activation_func = self.activations[layer["activation"]]
            A_current = activation_func(Z_current)
            Z.append(Z_current)
            A.append(A_current)
            
        # Backward propagation
        m = X.shape[1]
        dA = -(Y / A[-1]) + ((1 - Y) / (1 - A[-1])) # Initial gradient (loss function derivative)
        for i in reversed(range(len(self.layers))):
            activation_func = self.activations[self.layers[i]["activation"]]
            # Derivative of activation
            dZ = dA * activation_func(Z[i]) * (1 - activation_func(Z[i]))
            dW = np.dot(dZ, A[i].T) / m
            dB = np.sum(dZ, axis=1, keepdims=True) / m
            dA = np.dot(self.weights[i].T, dZ)

            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * dB
