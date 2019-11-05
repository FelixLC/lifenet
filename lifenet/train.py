"""
Here's a function that can train a neural net
"""

from lifenet.tensor import Tensor
from lifenet.nn import NeuralNet
from lifenet.loss import Loss, MSE
from lifenet.optim import Optimizer, SGD
from lifenet.data import BatchIterator


def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: BatchIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None:
    """
    A training function is just n loop on our data with:
     - prediction step
     - loss calculation
     - gradient backpropagation
     - parameter update

    Each loop, called an epoch means that our net has seen our entire dataset.
    We usually loop many times over the data to let the model learn the right amount
    of knowledge.
    """

