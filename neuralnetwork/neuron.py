"""
neuron.py
---------

Implements a neuron base class for creating activation-functions, used in
layers of a neural network.

The class is implemented as an ABC, as such a new type should subclass this.

Several neuron types are added as a standart.

"""
# Standart library import
from abc import ABC, abstractmethod

# Third party imports
import numpy as np


class NeuronBase(ABC):
    """ABC class for all neurons types.
    A neuron must implement its activation function (func)
    and the prime of it (func_prime)

    To use:
    from neuralnetwork.neuron import NeuronBase

    class Sigmoid(NeuronBase):
        def func(self, z):
            return 1.0/(1.0+np.exp(-z))
        def func_prime(self, z):
            return self.func(z)*(1-self.func(z))

    """

    @abstractmethod
    def func(self, z: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def func_prime(self, z: np.ndarray) -> np.ndarray:
        pass


class Sigmoid(NeuronBase):
    """
    Neuron using logistic function as the activation function.
    
    To use:
    sig = Sigmoid()
    z1 = sig.func(z0)
    """

    def func(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def func_prime(self, z):
        return self.func(z) * (1 - self.func(z))


class ReLU(NeuronBase):
    """
    Neuron using ReLU (retified linear unit)
    
    Note the prime (f'(x)) of ReLU is undefined at x=0,
    Tensorflow appear to use f'(0) = 0, as will it here. 

    To use:
    relu = ReLU()
    z1 = relu.func(z0)
    """

    def func(self, z):
        return np.fmax(0, z)

    def func_prime(self, z):
        return (z > 0).astype(z.dtype)
