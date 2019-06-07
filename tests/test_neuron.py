# import standart libaries

# import third party libaries
import numpy as np

# import package modules
from neuralnetwork.neuron import Sigmoid, ReLU, Tanh, NeuronBase


# Notes for improvement:
# Maybe convert to a test class for each neuron type 
# in order to package them together.
# Use a fixture to create a single neuron object of each type to be tested
# and not one en each function.

# Create parametrized test, for testing multipole outputs of same function.

# Perhaps use a mock object to test the NeuronBase creation,
#  or just create a dummy example.


# In general more tests needed.

def test_Sigmoid_class_type():
    assert isinstance(
        Sigmoid(), NeuronBase
    ), "Sigmoid must be an instance of NeuronBase"


def test_Sigmoid_func_zero_value_output():
    z = np.array([[0]])
    output = Sigmoid().func(z=z)
    # Sigmoid().func(z=z) == np.array([[0.5]])
    assert (
        output == np.array([[0.5]])
    ), "Sigmoid must equal 0.5 at z = 0.0"


def test_Sigmoid_func_20_value_output():
    z = np.array([[1]])
    assert (
        Sigmoid().func(z=z) > 0.5
    ), "Sigmoid with positive input must be greater than 0.5 "


def test_Sigmoid_func_neg_20_value_output():
    z = np.array([[-1]])
    assert (
        Sigmoid().func(z=z) == (1.0 / (1.0 + np.exp(-z)))
    ), "Sigmoid with negative input must be less than 0.5"


def test_Sigmoid_func_1_length_shape():
    z = np.array([[0]])
    assert (
        Sigmoid().func(z=z).shape == (1, 1)
    ), "Sigmoid with input vector (1,1) shape must output vector of shape (1,1)"


def test_Sigmoid_func_3_length_shape():
    z = np.array([[0, 1, 1]]).T
    assert (
        Sigmoid().func(z=z).shape == (3, 1)
    ), "Sigmoid with input vector of shape (3,1) must output a vector of shape (3,1)"


def test_Sigmoid_func_prime_output():
    z = np.array([[0, 0, 0]]).T
    assert (
        np.array_equal(Sigmoid().func_prime(z=z), np.array([[0.25, 0.25, 0.25]]).T)
    ), "Sigmoid prim_func with input vector [0,0,0].T must equal [0.25, 0.25, 0.25]"


def test_ReLU_class_type():
    assert isinstance(
        ReLU(), NeuronBase
    ), "ReLU must be an instance of NeuronBase"


def test_Tanh_class_type():
    assert isinstance(
        Tanh(), NeuronBase
    ), "Tanh must be an instance of NeuronBase"
