from typing import Dict, List, Optional, Set, Union
import json
import logging
import numpy as np
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, Field, ListField, MetadataField
from allennlp.data.instance import Instance
from allennlp.common.file_utils import cached_path
from torch import FloatTensor

from contexteval.contextualizers import Contextualizer
from contexteval.data.dataset_readers import TruncatableDatasetReader
from contexteval.data.fields import SequenceArrayField

logger = logging.getLogger(__name__)


@DatasetReader.register("event_factuality")
class EventFactualityDatasetReader(TruncatableDatasetReader):
    """
    A dataset reader for the processed ItHappened event factuality dataset.

    Parameters
    ----------
    include_raw_tokens: bool, optional (default=``False``)
        Whether to include the raw tokens in the generated instances. This is
        False by default since it's slow, but it is necessary if you want to use
        your ``Contextualizer`` as part of a model (e.g., for finetuning).
    contextualizer: Contextualizer, optional (default=``None``)
        If provided, it is used to produce contextualized representations of the text.
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
                 include_raw_tokens: bool = False,
                 contextualizer: Optional[Contextualizer] = None,
                 max_instances: Optional[Union[int, float]] = None,
                 seed: int = 0,
                 lazy: bool = False) -> None:
        super().__init__(
            max_instances=max_instances,
            seed=seed,
            lazy=lazy)
        self._include_raw_tokens = include_raw_tokens
        self._contextualizer = contextualizer

    @overrides
    def _read_dataset(self,
                      file_path: str,
                      count_only: bool = False,
                      keep_idx: Optional[Set[int]] = None):
        """
        Yield instances from the file_path.

        Parameters
        ----------
        file_path: str, required
            The path to the data file.
        count_only: bool, optional (default=``False``)
            If True, no instances are returned and instead a dummy object is
            returned. This is useful for quickly counting the number of instances
            in the data file, since creating instances is relatively expensive.
        keep_idx: Set[int], optional (default=``None``)
            If not None, only yield instances whose index is in this set.
        """
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        if count_only:
            logger.info("Counting instances in ItHappened file at: %s",
                        file_path)
        else:
            logger.info("Reading instances from ItHappened file at: %s",
                        file_path)

        index = 0
        # Read json data file
        with open(file_path) as event_factuality_data_file:
            event_factuality_data = json.load(event_factuality_data_file)
        for sentence_id, sentence_data in event_factuality_data.items():
            tokens = sentence_data["sentence"]
            predicate_indices = sentence_data["predicate_indices"]
            labels = sentence_data["labels"]
            # Skip the example if there are no predicated indices or labels
            if not predicate_indices or not labels:
                continue
            if keep_idx is not None and index not in keep_idx:
                index += 1
                continue
            if count_only:
                yield 1
                continue

            # Contextualize the tokens if a Contextualizer was provided.
            if self._contextualizer:
                token_representations = self._contextualizer([tokens])[0]
            else:
                token_representations = None

            yield self.text_to_instance(tokens=tokens,
                                        predicate_indices=predicate_indices,
                                        token_representations=token_representations,
                                        labels=labels)
            index += 1

    @overrides
    def text_to_instance(self,  # type: ignore
                         tokens: List[str],
                         predicate_indices: List[int],
                         token_representations: FloatTensor = None,
                         labels: List[float] = None):
        """
        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in the sentence to be encoded.
       predicate_indices: ``List[int]``, required.
            A List of int, where each item denotes the index of a
            token to predict a value for.
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
            label_indices : ``SequenceArrayField``
                Array of shape (num_labels,) corresponding to the indices of tokens
                to predict a value for.
            token_representations: ``ArrayField``
                Contains the representation of the tokens.
            labels: ``SequenceArrayField``
                The labels corresponding each arc represented in token_indices.
        """
        fields: Dict[str, Field] = {}

        # Add raw_tokens to the field
        if self._include_raw_tokens:
            fields["raw_tokens"] = ListField([MetadataField(token) for token in tokens])

        # Add label_indices to the field
        label_indices_field = SequenceArrayField(
            # Subtract 1 since original data is 1-indexed
            # Pad with -1 since 0 (usually mask token) is a valid label index
            np.array(predicate_indices, dtype="int64") - 1, padding_value=-1)
        fields["label_indices"] = label_indices_field

        if token_representations is None and self._contextualizer:
            # Contextualize the tokens
            token_representations = self._contextualizer([tokens])[0]

        # Add representations of the tokens at the arc indices to the field
        # If we don't have representations, use an empty numpy array.
        if token_representations is not None:
            fields["token_representations"] = ArrayField(
                token_representations.numpy())
        if labels:
            fields["labels"] = SequenceArrayField(np.array(labels, dtype="float32"))
        return Instance(fields)
