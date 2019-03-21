from typing import List
import logging

from allennlp.common.registrable import Registrable
import torch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Contextualizer(torch.nn.Module, Registrable):
    """
    A ``Contextualizer`` knows how to turn a batch of sentences (the latter of which is a
    sequence of strings) into a List of PyTorch Tensors, which are the
    contextualized representations of each input sentence.
    """
    def __init__(self):
        super().__init__()

    def forward(self, sentences: List[List[str]]) -> torch.FloatTensor:
        """
        Parameters
        ----------
        sentences: List[List[str]]
            A batch of sentences. len(sentences) is the batch size, and each sentence
            itself is a list of strings (the constituent words).

        Returns
        -------
        representations: List[FloatTensor]
            A list with the contextualized representations of all words in an input
            sentence. Each inner FloatTensor is of shape (num_layers, seq_len, repr_dim)
            indicating the number of layers of represntations to evaluate, the length
            of the sentence, and the dimensionality of each contextualized word representation,
            respectively.
        """
        raise NotImplementedError
