"""
We'll feed inputs into our network in batches of examples.
During training, we'll iterate over batches of examples to make predictions
and estimate our loss, in order to make changes in our model
"""
from typing import Iterator, NamedTuple

import numpy as np

from lifenet.tensor import Tensor

# A batch is a tuple of inputs and targets
# ([input_1, ... input_n], [target_1, ... target_n])
Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])


class BatchIterator:
    """Our batch iterator will be a callable object, initiated with the batch size.
    Once initiated with its batch size and called with inputs & targets for our dataset, we can iterate
    batch by batch
    """
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        """Create an iterator which will slice our inputs and targets by batch.
        For each iteration, yield a batch (the next slice in the origin table (next_inputs, next_targets)
        """
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)
