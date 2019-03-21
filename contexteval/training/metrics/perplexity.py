from overrides import overrides

from allennlp.training.metrics.metric import Metric

import math


@Metric.register("perplexity")
class Perplexity(Metric):
    """
    The perplexity metric stores the average negative log-likelihood over instances as
    well as how many instances the loss is taken from. It then calculates the perplexity
    by taking exp(average instance-wise loss)
    """
    def __init__(self) -> None:
        self._total_loss = 0.0
        self._total_num_instances = 0

    @overrides
    def __call__(self, loss, num_instances):
        """
        Parameters
        ----------
        loss: ``float``
            The average loss value of all the instances in the batch.
        num_instances: ``int``
            The number of instances that the loss is averaged over.
        """
        if hasattr(loss, "item"):
            loss = loss.item()
        if hasattr(num_instances, "item"):
            num_instances = num_instances.item()
        self._total_loss += (loss * num_instances)
        self._total_num_instances += num_instances

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        perplexity: float
            Returns exp(average instance loss), where exp is e^(x)
        """
        average_loss = self._total_loss / self._total_num_instances if self._total_num_instances > 0 else 0
        if reset:
            self.reset()
        return math.exp(average_loss)

    @overrides
    def reset(self):
        self._total_loss = 0.0
        self._total_num_instances = 0
