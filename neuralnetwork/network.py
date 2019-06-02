"""
network.py
---------

An implementation of a neural network.
Stores a list of layers in the network and
a list of the weights connecting between the layers.

At the moment it also is able to preform the learning operation and
holds the cost_function - this will be changed later.
"""
# Standart library imports
from typing import Tuple

# Third party imports
import numpy as np

# Package imports
from neuralnetwork.layer import Layer

# Define type aliases


class Network(object):
    """Network class for constructing the neural network.

    To use:
    net = Network()
    layer = Layer(10, Sigmoid())
    net.add_layer()

    Attributes:
        layer_list: A list of the layer objects in the network,
            order is first layer = layer_list[0]
        weights: A list of weights connecting layers
    """

    def __init__(self,):
        """Inits a blank network.
        """
        self.layer_list = []
        self.weights = []

    def add_layer(self, layer: Layer):
        """Add a layer object to the network.
        
        Layers must be added in sequence - therefore first layer = layer_list[0].
        The first layers is also defined as the input layer,
        so its bias will not be used.

        Args:
            layer: A Layer
        """
        if not self.layer_list:
            # First layer is the input
            self.layer_list.append(layer)

        else:
            left_layer_size = self.layer_list[-1].size
            self.layer_list.append(layer)
            right_layer_size = self.layer_list[-1].size

            # Random uniform init of weights for now
            self.weights.append(np.random.randn(right_layer_size, left_layer_size))

    def feedforward(self, a: np.ndarray) -> Tuple:
        """Feed an input a forward through the network.

        Feedforward the network using the input a 

        Args:
            a: input to the network - should maybe be renamed as x.

        Returns:
            A tuple of the inputs at each layer and the output of each.
            (array[size-1], array[size])
        """
        zs = []
        activations = [a]

        for connection_num, layer in enumerate(self.layer_list[1:], start=0):
            z, a = layer.calc_weighted_input_and_activation(
                a, self.weights[connection_num]
            )

            zs.append(z)
            activations.append(a)

        return (zs, activations)

    def calc_cost_driv(self, activation: np.ndarray, target: np.ndarray) -> np.ndarray:
        """The partial derivative of the cost function.

        RMS partial derivative - should be replaced by a cost_function class.
        """
        assert activation.shape == target.shape, "not matching size"
        return activation - target

    def output_error(
        self, z_final: np.ndarray, activation_final: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """Calculate the end error of the network.

        The final error of the network is used as a start for backpropagation.

        Args:
            z_final:
            activation_final:
            target:

        Return:
            A numpy array of the error after the final layer compared the training data.
        """
        return self.calc_cost_driv(activation_final, target) * self.layer_list[
            -1
        ].neuron_prime(z_final)

    def backpropagate_error(self, zs, last_step_error):
        """Backpropagates the final error through the layer.

        Args:
            zs:
            last_step_error:

        Returns:
            A list of the error at each layer.
        """
        errors = [last_step_error]
        # Skiping over the last and first layer - since this is the input layer.
        for layer_l, w_l, z_l in zip(
            reversed(self.layer_list[1:-1]),
            reversed(self.weights[1:]),
            reversed(zs[:-1]),
        ):
            primed = layer_l.neuron_prime(z_l)
            error_l = np.dot(w_l.T, errors[-1]) * primed
            errors.append(error_l)

        return errors[::-1]

    def SGD(self, train_x, train_y, epochs, mini_batch_size, eta):
        """Stochastic gradient descent.

        Should be replaced by an optimizer class instead.
        """
        n = train_y.shape[1]
        num_of_batches = n / mini_batch_size
        for epoch in range(epochs):
            # train_x = np.random.shuffle(train_x)

            indices = np.random.permutation(n)
            train_x = train_x[:, indices]
            train_y = train_y[:, indices]

            mini_batches_x = np.array_split(train_x, num_of_batches, axis=1)
            mini_batches_y = np.array_split(train_y, num_of_batches, axis=1)

            for mini_batch_x, mini_batch_y in zip(mini_batches_x, mini_batches_y):

                # Handeling not even batch sizes
                cur_batch_size = len(mini_batch_y)

                # step one feedforward
                zs, activations = self.feedforward(mini_batch_x)
                # step two calc output error
                delta_L = self.output_error(zs[-1], activations[-1], mini_batch_y)
                # #backprop
                errors = self.backpropagate_error(zs, delta_L)
                # Gradient_decent
                for l, layer_l in enumerate(reversed(self.layer_list[1:]), start=1):

                    delta_b = -(eta / cur_batch_size) * np.sum(
                        errors[-l], axis=1
                    ).reshape(-1, 1)

                    layer_l.add_bias(delta_b)

                    self.weights[-l] = self.weights[-l] - (
                        eta / cur_batch_size
                    ) * np.dot(errors[-l], activations[-l - 1].T)
