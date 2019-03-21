from typing import Dict, List, Optional, Set, Union
import logging
from overrides import overrides

from allennlp.data.fields import ArrayField, Field, ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from torch import FloatTensor

from contexteval.contextualizers import Contextualizer
from contexteval.data.dataset_readers import TruncatableDatasetReader

logger = logging.getLogger(__name__)


class TaggingDatasetReader(TruncatableDatasetReader):
    """
    A base DatasetReader for tagging tasks.

    Parameters
    ----------
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
                 contextualizer: Optional[Contextualizer] = None,
                 max_instances: Optional[Union[int, float]] = None,
                 seed: int = 0,
                 lazy: bool = False) -> None:
        super().__init__(
            max_instances=max_instances,
            seed=seed,
            lazy=lazy)
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
        raise NotImplementedError

    @overrides
    def text_to_instance(self,  # type: ignore
                         tokens: List[str],
                         token_representations: FloatTensor = None,
                         labels: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in a given sentence.
        token_representations: ``FloatTensor``, optional (default=``None``)
            Precomputed token representations to use in the instance. If ``None``,
            we use a ``Contextualizer`` provided to the dataset reader to calculate
            the token representations. Shape is (seq_len, representation_dim).
        labels : ``List[str]``, optional, (default = None).
            The labels for the words in the sentence.

        Returns
        -------
        An ``Instance`` containing the following fields:
            raw_tokens : ListField[MetadataField]
                The raw string tokens in the sentence. Each MetadataField stores the raw string
                of a single token.
            token_representations : ``ArrayField``
                Stores the representations for each token. Shape is (seq_len, represenatation_dim)
            labels : ``SequenceLabelField``
                The labels (only if supplied)
        """
        # pylint: disable=arguments-differ
        raw_text_field = ListField([MetadataField(token) for token in tokens])
        fields: Dict[str, Field] = {"raw_tokens": raw_text_field}

        if token_representations is None and self._contextualizer:
            # Contextualize the tokens
            token_representations = self._contextualizer(tokens)
        # Add representations for the tokens in the sequence to the field.
        if token_representations is not None:
            fields["token_representations"] = ArrayField(
                token_representations.numpy())

        if labels is not None:
            fields["labels"] = SequenceLabelField(labels, raw_text_field)
        return Instance(fields)
