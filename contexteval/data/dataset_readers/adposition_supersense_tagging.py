from typing import Dict, List, Optional, Set, Union
import json
import logging
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, Field, ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.common.file_utils import cached_path
import numpy as np
from torch import FloatTensor

from contexteval.data.dataset_readers import TaggingDatasetReader
from contexteval.contextualizers import Contextualizer
from contexteval.data.fields import SequenceArrayField

logger = logging.getLogger(__name__)

mode_to_key = {
    "role": "ss",
    "function": "ss2"
}


@DatasetReader.register("adposition_supersense_tagging")
class AdpositionSupersenseTaggingDatasetReader(TaggingDatasetReader):
    """
    Reads a JSON file with data from STREUSLE 4.1 and returns instances suitable
    for tagging of adposition supersenses.

    Parameters
    ----------
    mode: str
        The supersense to predict (one of "role" or "function").
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
                 mode: str,
                 include_raw_tokens: bool = False,
                 contextualizer: Optional[Contextualizer] = None,
                 max_instances: Optional[Union[int, float]] = None,
                 seed: int = 0,
                 lazy: bool = False) -> None:
        super().__init__(
            contextualizer=contextualizer,
            max_instances=max_instances,
            seed=seed,
            lazy=lazy)
        if mode not in ["role", "function"]:
            raise ConfigurationError("Invalid mode {}, must be one of "
                                     "\"role\" or \"function\".".format(mode))
        self._mode = mode
        self._include_raw_tokens = include_raw_tokens

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
            logger.info("Counting instances in STREUSLE file at: %s",
                        file_path)
        else:
            logger.info("Reading instances from STREUSLE file at: %s",
                        file_path)
        index = 0
        with open(file_path) as input_file:
            data = json.load(input_file)
            for instance in data:
                # Get the tokens
                tokens = [x["word"] for x in instance["toks"]]

                # Get the indices and supersenses if the item to predict starts with p.
                swes = [(int(index) - 1, swe[mode_to_key[self._mode]]) for index, swe in
                        instance["swes"].items() if swe[mode_to_key[self._mode]] and
                        swe[mode_to_key[self._mode]].startswith("p.")]

                # Skip the instance if there are no swe indices to classify
                if not swes:
                    continue

                if keep_idx is not None and index not in keep_idx:
                    index += 1
                    continue
                if count_only:
                    yield 1
                    continue

                # Contextualize the tokens if a Contextualizer was provided.
                # TODO (nfliu): How can we make this batched?
                # Would make contextualizers that use the GPU much faster.
                if self._contextualizer:
                    token_representations = self._contextualizer([tokens])[0]
                else:
                    token_representations = None

                label_indices = [x[0] for x in swes]
                labels = [x[1] for x in swes]
                # Create an instance from the label indices and the labels
                yield self.text_to_instance(
                    tokens=tokens,
                    label_indices=label_indices,
                    token_representations=token_representations,
                    labels=labels)
                index += 1

    def text_to_instance(self,  # type: ignore
                         tokens: List[str],
                         label_indices: List[int],
                         token_representations: FloatTensor = None,
                         labels: List[str] = None):
        """
        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in the sentence to be encoded.
       label_indices: ``List[int]``, required.
            A List of int, where each item denotes the index of a
            token to predict a label for.
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
                to predict a label for.
            token_representations: ``ArrayField``
                Contains the representation of the tokens.
            labels: ``SequenceLabelField``
                The labels corresponding each arc represented in token_indices.
        """
        fields: Dict[str, Field] = {}

        # Add raw_tokens to the field
        if self._include_raw_tokens:
            fields["raw_tokens"] = ListField([MetadataField(token) for token in tokens])

        # Add label_indices to the field
        label_indices_field = SequenceArrayField(
            np.array(label_indices, dtype="int64"), padding_value=-1)
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
            fields["labels"] = SequenceLabelField(labels, label_indices_field)
        return Instance(fields)
