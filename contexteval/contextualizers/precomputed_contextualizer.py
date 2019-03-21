import json
import logging
from typing import List, Optional

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
import h5py
import torch

from contexteval.contextualizers import Contextualizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Contextualizer.register("precomputed_contextualizer")
class PrecomputedContextualizer(Contextualizer):
    """
    This "contextualizer" simply reads representations from an hdf5 file provided
    upon construction. This is useful if you want to evaluate representations
    from some CLI tool (that might not even have a Python interface and can't
    be added directly).

    Parameters
    ----------
    representations_path: str
        Path to an HDF5 file with the representations.
    layer_num: int, optional (default=``None``)
        The (0-indexed) representation layer to use, when
        multiple are returned. -1 indicates that the last
        layer is used. If None, we use the last layer (-1).
    scalar_weights: List[float], optional (default=``None``)
        Compute a weighted average of the layers in the
        representation instead of taking the representation
        at layer_num. Also requires that a ``gamma`` value
        is provided. These weights should be normalized.
    gamma: float, optional (default=``None``)
        Compute a weighted average of the layers in the
        representation instead of taking the representation
        at layer_num. Also requires that a ``scalar_weights`` value
        is provided.
    first_half_only: bool, optional (default=``False``)
        Whether to return only the first half of the word representation.
        For example, for the original ELMo embeddings, this would return
        only the first 512 elements of each word vectors (vector[:512]).
    second_half_only: bool, optional (default=``False``)
        Whether to return only the second half of the word representation.
        For example, for the original ELMo embeddings, this would return
        only the last 512 elements of each word vectors (vector[512:]).
    """
    def __init__(self,
                 representations_path: str,
                 layer_num: Optional[int] = None,
                 scalar_weights: Optional[List[float]] = None,
                 first_half_only: bool = False,
                 second_half_only: bool = False,
                 gamma: Optional[float] = None) -> None:
        super(PrecomputedContextualizer, self).__init__()
        # if `file_path` is a URL, redirect to the cache
        self._representations_path = cached_path(representations_path)
        # Read the HDF5 file.
        self._representations = h5py.File(self._representations_path, "r")
        # Get the sentences to index mapping
        self._sentence_to_index = json.loads(self._representations.get("sentence_to_index")[0])
        # The layer to use.
        self._layer_num = layer_num
        # The scalar weights.
        self._scalar_weights = torch.FloatTensor(scalar_weights) if scalar_weights else None
        # The gamma value
        self._gamma = torch.FloatTensor([gamma]) if gamma else None

        self._first_half_only = first_half_only
        self._second_half_only = second_half_only
        if self._first_half_only is True and self._second_half_only is True:
            raise ConfigurationError("first_half_only and second_half_only "
                                     "cannot both be true.")

        # If scalar_weights are provided, gamma must be as well (and vice versa)
        if ((self._gamma is None and self._scalar_weights is not None) or
                (self._scalar_weights is None and self._gamma is not None)):
            raise ConfigurationError("scalar_weights and gamma must either both be None "
                                     "(the default) or both have input values. Got scalar_weights "
                                     "{} , gamma {}".format(self._scalar_weights, self._gamma))

        # Everything was provided, raise an error.
        if self._layer_num and (self._scalar_weights and self._gamma):
            raise ConfigurationError("Either provide a layer_num or values for scalar_weights "
                                     "and gamma, but do not use both sets.")

        # Nothing was provided, so default to using the last layer
        if self._layer_num is None and (self._scalar_weights is None and self._gamma is None):
            self._layer_num = -1

        # The number of layers in the input HDF5 file
        self._num_layers = None

    def forward(self, sentences: List[List[str]]) -> torch.FloatTensor:
        """
        Parameters
        ----------
        sentences: List[List[str]]
            A batch of sentences. len(sentences) is the batch size, and each sentence
            itself is a list of strings (the constituent words). If the batch is padded,
            the expected padding token in the Python ``None``.

        Returns
        -------
        representations: List[FloatTensor]
            A list with the contextualized representations of all words in an input sentence.
            Each inner FloatTensor is of shape (seq_len, repr_dim), and an outer List
            is used to store the representations for each input sentence.
        """
        batch_representations = []

        for sentence in sentences:
            representation = torch.FloatTensor(self._representations[
                self._sentence_to_index[" ".join([x for x in sentence if x is not None])]])
            num_layers = 1 if representation.dim() == 2 else representation.size(0)
            if self._num_layers is None:
                self._num_layers = num_layers
            elif self._num_layers != num_layers:
                raise ValueError("Inconsistent number of layers in the input representations. "
                                 "Saw representations with {} and {} layers".format(
                                     self._num_layers, num_layers))
            if self._scalar_weights is not None and (len(self._scalar_weights) != self._num_layers):
                raise ValueError("Must have the same number of scalar weights as "
                                 "representation layers, but got {} and {} respectively".format(
                                     self._scalar_weights, self._num_layers))

            if representation.dim() == 3:
                # We should either take a particular layer or do a scalar mix.
                if self._layer_num is not None:
                    representation = representation[self._layer_num]
                else:
                    scalar_weights_to_broadcast = self._scalar_weights.unsqueeze(-1).unsqueeze(-1)
                    # Take the scalar weighting
                    weighted_representation = representation * scalar_weights_to_broadcast
                    representation = self._gamma * torch.sum(weighted_representation, dim=0)
            if self._first_half_only:
                representation = torch.chunk(representation, chunks=2, dim=-1)[0]
            elif self._second_half_only:
                representation = torch.chunk(representation, chunks=2, dim=-1)[1]
            batch_representations.append(representation)
        return batch_representations
