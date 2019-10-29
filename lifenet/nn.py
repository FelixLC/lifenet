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
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        """Recursively compute gradients from inputs for each layer"""

        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad
