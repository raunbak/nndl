"""
layer.py
---------

Implements a layer class used in a Neural network
Responsability:
Contrains a type of neuron and how many
Contrains the bias on the neurons in the layer

Calculate the output of the layer based on the incoming a_l and weights

"""
# Third party imports
import numpy as np

# Package imports
from neuralnetwork.neuron import NeuronBase


class Layer(object):
    """Layer class for storing neurons in the network, and their bias

    To use:
    # Make a layer of sigmoid neurons of size 10
    from neuralnetwork.neuron import Sigmoid
    layer = Layer(10, Sigmoid())
    """

    neuron_type: NeuronBase
    num_of_neuron: int
    biases_on_neurons: np.ndarray

    def __init__(self, size: int, neuron_type: NeuronBase):

        if not isinstance(neuron_type, NeuronBase):
            raise TypeError("Neuron needs to subclass NeuronBase")

        if size < 1:
            raise ValueError("Layer size must be positive")
        self.neuron_type = neuron_type
        self.num_of_neuron = size

    def get_size(self,) -> int:
        """
        """
        return self.num_of_neuron

    def init_biases(self, biases: np.array):
        """
        ADD CHECK SO SIZE FITS
        """

        self.biases_on_neurons = biases

    def update_biases(self, delta: np.array):
        self.biases_on_neurons = self.biases_on_neurons + delta

    def get_biases(self,) -> np.array:
        """
        return the weight of the network linking into this layer
        """
        return self.biases_on_neurons

    def calc_weigthed_input_and_activation(self, a, incoming_weights):
        """
        Calculate the output of the layer - given the incoming weights and 
        """
        # np.dot(w, a)+b
        m, _ = incoming_weights.shape
        _, n = a.shape

        z_l = (
            np.dot(incoming_weights, a)
            + self.biases_on_neurons
        )

        return (z_l, self.neuron_type.func(z_l))

    def neuron_prime(self, z):
        """
        """
        return self.neuron_type.func_prime(z)
