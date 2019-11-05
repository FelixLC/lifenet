"""
Our neural nets will be made up of layers.
Each layer needs to pass its inputs forward
and propagate gradients backward.

For example, a neural net might look like
inputs -> Linear Layer -> Tanh Layer -> Linear Layer -> output
"""

from typing import Callable, Dict

import numpy as np

from lifenet.tensor import Tensor


class Layer:
    """
    A layer has some parameters that will be used to predict, and can store some gradients
    regarding those parameters.
    It can make a forward pass - predict outputs with respect to inputs
    And a backward pass - compute gradients for each parameter
    """
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce the outputs corresponding to these inputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagate this gradient through the layer
        """
        raise NotImplementedError


class Linear(Layer):
    """
    computes output = inputs @ w + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        outputs = inputs @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = g(z(x)) and z(x) = w @ x + b

        with:
        - g our loss function, or the gradient from next layer
        - z(x) our inputs
        - w & b our weights for current layer

        then:
        - dy/dw = g'(x) * x
        - dy/dx = g'(x) * w
        - dy/db = g'(x)
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T


F = Callable[[Tensor], Tensor]


class Activation(Layer):
    """
    An activation layer just applies a function
    element-wise to its inputs.
    It has no params
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = g(a(x))

        with:
         - g our loss function, or the gradient from next layer
         - a our activation function, which depends on our inputs

        then:
        - dy/dx = g'(a(x)) * a'(x)
        """
        return self.f_prime(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:
    """
    Hyperbolic Tan takes any number and ouputs a number between -1 & 1
    [-∞; +∞] => [-1; 1]
    """
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    """
    Hyperbolic Tan Derived is 1 - tan²(x)
    """
    y = tanh(x)
    return 1 - y ** 2


class Tanh(Activation):
    """
    Tanh activation is a common activation function
    """
    def __init__(self):
        super().__init__(tanh, tanh_prime)
