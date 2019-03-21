from typing import Dict, List, Optional, Tuple, Union
import logging
import random

from allennlp.common.checks import ConfigurationError
from allennlp.data.fields import ArrayField, Field, ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
import numpy as np
from torch import FloatTensor

from contexteval.contextualizers import Contextualizer
from contexteval.data.dataset_readers import TruncatableDatasetReader
from contexteval.data.fields import SequenceArrayField

logger = logging.getLogger(__name__)
NEGATIVE_SAMPLING_METHODS = ["balanced", "all"]


class DependencyArcPredictionDatasetReader(TruncatableDatasetReader):
    """
    Base dataset readers for tasks that involve predicting whether two tokens
    in a sequence have a dependency relation and doing some dataset augmentation to add
    negative examples as well. This is used in the SyntacticDependencyArcPredictionDatasetReader
    and the SemanticDependencyArcPredictionDatasetReader.

    Parameters
    ----------
    negative_sampling_method: str
        How to generate negative examples for use in the dependency arc
        prediction task. "balanced" indicates that for each positive (child, parent)
        example, we will produce a new example with (child, random_word), where
        random_word is a random token in the sequence that is not a parent of the child.
        "all" indicates that we will take all available negative examples
        (i.e., all pairs of words a, b where a is not the child of b).
    contextualizer: Contextualizer, optional (default=``None``)
        If provided, it is used to produce contextualized representations of the text.
    include_raw_tokens: bool, optional (default=``False``)
        Whether to include the raw tokens in the generated instances. This is
        False by default since it's slow, but it is necessary if you want to use
        your ``Contextualizer`` as part of a model (e.g., for finetuning).
    max_instances: int or float, optional (default=``None``)
        The number of instances to use during training. If int, this value is taken
        to be the absolute amount of instances to use. If float, this value indicates
        that we should use that proportion of the total training data. If ``None``,
        all instances are used.
    seed: int, optional (default=``0``)
        The random seed to use.
    lazy : ``bool``, optional (default=``False``)
        If this is true, ``instances()`` will return an object whose ``__iter__`` method
        reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.
    """
    def __init__(self,
                 negative_sampling_method: str,
                 contextualizer: Optional[Contextualizer] = None,
                 include_raw_tokens: bool = True,
                 max_instances: Optional[Union[int, float]] = None,
                 seed: int = 0,
                 lazy: bool = False) -> None:
        super().__init__(max_instances=max_instances,
                         lazy=lazy)
        if negative_sampling_method not in NEGATIVE_SAMPLING_METHODS:
            raise ConfigurationError(
                "Received negative_sampling_method {}, but allowed negative "
                "sampling methods are {}".format(negative_sampling_method,
                                                 NEGATIVE_SAMPLING_METHODS))
        self._negative_sampling_method = negative_sampling_method
        self._contextualizer = contextualizer
        self._include_raw_tokens = include_raw_tokens
        self._rng = random.Random(seed)
        self._seed = seed

    def _reseed(self, seed: int = 0):
        """
        Reseed the Random instance underlying the dataset reader.

        Parameters
        ----------
        seed: int, optional (default=``0``)
            The random seed to use.
        """
        self._rng = random.Random(seed)

    def text_to_instance(self,  # type: ignore
                         tokens: List[str],
                         arc_indices: List[Tuple[int, int]],
                         token_representations: FloatTensor = None,
                         labels: List[str] = None):
        """
        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in the sentence to be encoded.
        arc_indices: ``List[Tuple[int, int]]``, required.
            A List of tuples, where each item denotes an arc. An arc is a
            tuple of (child index, parent index). Indices are 0 indexed.
        token_representations: ``FloatTensor``, optional (default=``None``)
            Precomputed token representations to use in the instance. If ``None``,
            we use a ``Contextualizer`` provided to the dataset reader to calculate
            the token representations. Shape is (seq_len, representation_dim).
        labels: ``List[str]``, optional (default=``None``)
            The labels of the arcs. ``None`` indicates that labels are not
            provided.

        Returns
        -------
        An ``Instance`` containing the following fields:
            raw_tokens : ListField[MetadataField]
                The raw str tokens in the sequence. Each MetadataField stores the raw string
                of a single token.
            arc_indices : ``SequenceArrayField``
                Array of shape (num_arc_indices, 2) corresponding to the arc indices.
                The first column holds the child indices, and the 2nd column holds
                their respective parent indices.
            token_representations: ``ArrayField``
                Contains the representation of the tokens.
            labels: ``SequenceLabelField``
                The labels corresponding each arc represented in token_indices.
        """
        fields: Dict[str, Field] = {}

        # Add tokens to the field
        if self._include_raw_tokens:
            fields["raw_tokens"] = ListField([MetadataField(token) for token in tokens])
        # Add arc indices to the field
        arc_indices_field = SequenceArrayField(
            np.array(arc_indices, dtype="int64"))
        fields["arc_indices"] = arc_indices_field

        if token_representations is None and self._contextualizer:
            # Contextualize the tokens
            token_representations = self._contextualizer([tokens])[0]

        # Add representations of the tokens at the arc indices to the field
        # If we don't have representations, use an empty numpy array.
        if token_representations is not None:
            fields["token_representations"] = ArrayField(
                token_representations.numpy())
        if labels:
            fields["labels"] = SequenceLabelField(labels, arc_indices_field)
        return Instance(fields)

    def _sample_negative_indices(self,  # type: ignore
                                 child_index: int,
                                 all_arc_indices: List[Tuple[int, int]],
                                 seq_len: int):
        """
        Given a child index, generate a new (child, index) pair where
        ``index`` refers to an index in the sequence that is not a parent
        of the provided child.

        Parameters
        ----------
        child_index: ``int``, required
            The index of the child to sample a negative example for.
        all_arc_indices: ``List[Tuple[int, int]]``, required
            A list of (child index, parent index) pairs that correspond to each
            dependency arc in the sentence. Indices are 0 indexed.
        seq_len: ``int``, required
            The length of the sequence.
        """
        # All possible parents
        all_possible_parents = set(range(seq_len))
        # Find all the parents of the child.
        parents = {arc_indices[1] for arc_indices in all_arc_indices if
                   arc_indices[0] == child_index}
        # Get the indices that are not parents of the child or the child itself
        non_parents = all_possible_parents - parents.union({child_index})
        # If there are no indices that are not a parent.
        if not non_parents:
            return None
        return (child_index, self._rng.sample(non_parents, 1)[0])
