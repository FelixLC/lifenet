"""
A NeuralNet is just a collection of layers stacked in a particular way
It behaves a lot like a layer itself, although
we're not going to make it one.
"""
from typing import Sequence, Iterator, Tuple

from lifenet.tensor import Tensor
from lifenet.layers import Layer


class NeuralNet:
    """
    Just as our layers, the neural net will be able to pass the input forward,
    and propagate gradients backward
    """
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        """Recursively compute outputs from inputs for each layer"""


    def backward(self, grad: Tensor) -> Tensor:
        """Recursively compute gradients from inputs for each layer"""

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:

