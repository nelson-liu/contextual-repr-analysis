import json
import logging
from typing import List

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.modules.scalar_mix import ScalarMix
import h5py
import torch

from contexteval.contextualizers import Contextualizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Contextualizer.register("scalar_mixed_precomputed_contextualizer")
class ScalarMixedPrecomputedContextualizer(Contextualizer):
    """
    This contextualizer simply reads N layers of representations from an
    hdf5 file provided upon construction, and learns a scalar weighting of
    them as the final output representation.

    Specifically, we computes a parameterized scalar mixture of N tensors,
    ``mixture = gamma * sum(s_k * tensor_k)`` where ``s = softmax(w)``,
    with ``w`` and ``gamma`` scalar parameters to be learnt.

    Parameters
    ----------
    representations_path: str
        Path to an HDF5 file with the representations.
    num_layers: int
        The number of layers of representations each key of
        ``representations_path``.
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
                 num_layers: int,
                 first_half_only: bool = False,
                 second_half_only: bool = False) -> None:
        super(ScalarMixedPrecomputedContextualizer, self).__init__()
        # if `file_path` is a URL, redirect to the cache
        self._representations_path = cached_path(representations_path)
        # Read the HDF5 file.
        self._representations = h5py.File(representations_path, "r")
        # Get the sentences to index mapping
        self._sentence_to_index = json.loads(self._representations.get("sentence_to_index")[0])

        # The number of layers in the input HDF5 file
        self._num_layers = num_layers

        # Computes a paramterized scalar mixture.
        self._scalar_mix = ScalarMix(self._num_layers)

        self._first_half_only = first_half_only
        self._second_half_only = second_half_only
        if self._first_half_only is True and self._second_half_only is True:
            raise ConfigurationError("first_half_only and second_half_only "
                                     "cannot both be true.")

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
            representation = self._scalar_mix.gamma.new_tensor(
                self._representations[self._sentence_to_index[
                    " ".join([x for x in sentence if x is not None])]])
            if representation.dim() != 3:
                raise ValueError("Got representations of shape {}, expected 3 "
                                 "dimensions of shape (num_layers, sequence_length, "
                                 "embedding_dim)".format(representation.size()))

            # Take a scalar mix of the representations.
            mixed_representation = self._scalar_mix(representation)
            if self._first_half_only:
                mixed_representation = torch.chunk(representation, chunks=2, dim=-1)[0]
            elif self._second_half_only:
                mixed_representation = torch.chunk(representation, chunks=2, dim=-1)[1]
            batch_representations.append(mixed_representation)

        return batch_representations
