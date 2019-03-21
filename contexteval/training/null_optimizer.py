from collections import defaultdict
import logging

import torch

from allennlp.training.optimizers import Optimizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Optimizer.register('null')
class NullOptimizer(torch.optim.Optimizer):
    """
    A dummy optimizer that does not update anything. This is necessary
    for models that have no parameters to optimize, since PyTorch optimizers
    error out when the input list of parameters is empty.
    """
    def __init__(self, params=[], defaults={}):
        self.defaults = defaults
        if isinstance(params, torch.Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(params))

        self.state = defaultdict(dict)
        self.param_groups = []

    def step(self, closure=None):
        """
        Performs a single optimization step. This is a no-op.

        Parameters
        ----------
        closure : ``callable``, optional.
            A closure that reevaluates the model and returns the loss.
        """
        pass

    def zero_grad(self):
        pass

    def state_dict(self):
        return {}
