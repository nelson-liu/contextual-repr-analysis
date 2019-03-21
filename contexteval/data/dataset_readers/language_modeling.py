from typing import Optional, Set, Union
import logging
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from contexteval.contextualizers import Contextualizer
from contexteval.data.dataset_readers import TaggingDatasetReader

logger = logging.getLogger(__name__)


@DatasetReader.register("billion_word_benchmark_language_modeling")
class LanguageModelingDatasetReader(TaggingDatasetReader):
    """
    Reads a file with a sentence per line (billion-word benchmark format), and
    returns instances for language modeling. Each instances is a line in the dataset,
    and they are predicted independently of each other.

    Parameters
    ----------
    max_length: int, optional (default=50)
        The maximum length of the sequences to use in the LM task. Any sequences that are
        longer than this value will be discarded.
    backward: bool, optional (default=False)
        If so, generate instances suitable for evaluating the a backward language model.
        For example, if the sentence is [a, b, c, d], the forward instance would have tokens of
        [a, b, c] and labels of [b, c, d], whereaas the backward instance would have tokens of
        [b, c, d] and labels of [a, b, c].
    vocabulary_path: str, optional (default=None)
        If provided, words in the input files that are not in this vocabulary are set to "<UNK>".
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
                 max_length: int = 50,
                 backward: bool = False,
                 vocabulary_path: Optional[str] = None,
                 contextualizer: Optional[Contextualizer] = None,
                 max_instances: Optional[Union[int, float]] = None,
                 seed: int = 0,
                 lazy: bool = False) -> None:
        super().__init__(
            contextualizer=contextualizer,
            max_instances=max_instances,
            seed=seed,
            lazy=lazy)

        self._max_length = max_length
        self._vocabulary_path = vocabulary_path
        self._vocabulary: Set[str] = set()

        if vocabulary_path:
            # Load the vocabulary
            cached_vocabulary_path = cached_path(vocabulary_path)
            with open(cached_vocabulary_path) as cached_vocabulary_file:
                for line in cached_vocabulary_file:
                    token = line.rstrip("\n")
                    self._vocabulary.add(token)

        self._backward = backward

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
            logger.info("Counting instances (backward: %s) in LM file at: %s",
                        self._backward, file_path)
        else:
            logger.info("Reading instances (backward: %s) from lines in LM file at: %s",
                        self._backward, file_path)
        index = 0
        with open(file_path) as input_file:
            for line in input_file:
                clean_line = line.rstrip("\n")
                if line.startswith("#"):
                    continue
                # Get tokens and the labels of the instance
                tokenized_line = clean_line.split(" ")

                if not tokenized_line or len(tokenized_line) > self._max_length:
                    continue
                if count_only:
                    yield 1
                    continue
                if keep_idx is not None and index not in keep_idx:
                    index += 1
                    continue

                # Replace OOV tokens in tokenized_line
                if self._vocabulary:
                    tokenized_line = [word if word in self._vocabulary else "<UNK>" for
                                      word in tokenized_line]

                if self._backward:
                    # Tokens are all tokens, labels are a BOS indicator + all except last token
                    labels = ["<S>"] + tokenized_line[:-1]
                else:
                    # Tokens are all tokens, and labels
                    # are all except first token + a EOS indicator
                    labels = tokenized_line[1:] + ["</S>"]

                # Contextualize the tokens if a Contextualizer was provided.
                if self._contextualizer:
                    token_representations = self._contextualizer([tokenized_line])[0]
                else:
                    token_representations = None

                yield self.text_to_instance(tokenized_line,
                                            token_representations,
                                            labels)
                index += 1
