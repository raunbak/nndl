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
    """Layer class for storing neurons in the network, and their bias.

    To use:
    # Make a layer of sigmoid neurons of size 10
    from neuralnetwork.neuron import Sigmoid
    layer = Layer(10, Sigmoid())

    Attributes:
        size: An int of how many neurons are in this layer.
        bias: A numpy array of the bias on the neuron in this layer.
        neuron_type: A NeuronBase subclass object,
            defines which activation function is used in the layer.
    """
    def __init__(self, size: int, neuron_type: NeuronBase):
        """Inits a Layer with a number of neurons and a neuron type.

        """

        if not isinstance(neuron_type, NeuronBase):
            raise TypeError("Neuron needs to subclass NeuronBase")

        if size < 1:
            raise ValueError("Layer size must be positive")

        self.neuron_type = neuron_type
        self.num_of_neuron = size
        self.biases_on_neurons = np.zeros([size, 1])

    @property
    def size(self):
        return self.__get_size()

    def __get_size(self) -> int:
        return self.num_of_neuron

    @property
    def bias(self):
        return self.__get_bias()

    @bias.setter
    def bias(self, bias: np.ndarray):
        return self.__set_bias(bias)

    def __set_bias(self, bias: np.ndarray):
        self.biases_on_neurons = bias

    def __get_bias(self) -> np.ndarray:
        return self.biases_on_neurons

    def add_bias(self, delta: np.ndarray):
        """ Adds delta to the bias of the neuron in the layer.

        Adds a delta array elementswise to the existing bias in the layer.

        Args:
            delta: a numpy array of the change to be done to the bias in the layer.

        Return:
            None

        Raises:
            ValueError: delta shape not matching the bias' shape.
        """
        if not delta.shape == self.biases_on_neurons.shape:
            raise ValueError("")
        self.biases_on_neurons += delta

    def calc_weighted_input_and_activation(self, a, incoming_weights):
        """Calculate output of layer, and its weighted input.

        Using z = dot(w,a)+b find the weighted input and pass it to the neuron 
        to calc the output of the layer.

        Args:
            a: a numpy array of the output from the previous layer.
            incoming_weights: A numpy matrix of the weights connecting
                this layer and the previous one.

        Returns:
            A tuple of the weighted input (z_l) and the output of the layer a_l,
            shaped as such (z_l,a_l).
        """
        if not incoming_weights.shape[1] == a.shape[0]:
            raise ValueError("Mismatch in dimensions")

        z_l = np.dot(incoming_weights, a) + self.biases_on_neurons

        return (z_l, self.neuron_type.func(z_l))

    def neuron_prime(self, z) -> np.ndarray:
        """Calculates the prime of weighted input z at this layer.

        Uses the input weighted z and calculates the prime of
        the activation function.

        Args:
            z: weighted input z=np.dot(w,a)+b

        Returns:
            A (size,1) numpy array of the layers neuron type. 
        """
        return self.neuron_type.func_prime(z)
