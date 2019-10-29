"""
We use an optimizer to adjust the parameters
of our network based on the gradients computed
during backpropagation.
"""
from lifenet.nn import NeuralNet


class Optimizer:
    """
    The optimizer adjusts the parameters of our network step by step, according
    to a meta parameter called learning rate.s
    """
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent is a classic optimizer for finding a minimum in a (convex)
    function. It searches for the the strongest downhill way to the minimum, taking a step
    of a learning rate length in the downhill direction.
    """
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        """Gradient has been propagated back in the neural net. For each param in the net,
        take a `step` in the direction of the gradient to update the param"""
        for param, grad in net.params_and_grads():
            param -= self.lr * grad
