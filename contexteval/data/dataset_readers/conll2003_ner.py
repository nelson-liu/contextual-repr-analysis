from typing import Optional, Set, Union
import itertools
import logging

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from overrides import overrides

from contexteval.contextualizers import Contextualizer
from contexteval.data.dataset_readers import TaggingDatasetReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":
            return True
        else:
            return False


@DatasetReader.register("conll2003_ner")
class Conll2003NERDatasetReader(TaggingDatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD POS-TAG CHUNK-TAG NER-TAG

    with a blank line indicating the end of each sentence
    and '-DOCSTART- -X- -X- O' indicating the end of each article,
    and converts it into a ``Dataset`` suitable for sequence tagging.


    This dataset reader ignores the "article" divisions and simply treats
    each sentence as an independent ``Instance``. (Technically the reader splits sentences
    on any combination of blank lines and "DOCSTART" tags; in particular, it does the right
    thing on well formed inputs.)

    Parameters
    ----------
    label_encoding : ``str``, optional (default=``IOB1``)
        Label encoding to use in data . Valid options are "BIOUL" or "IOB1".
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
                 label_encoding: str = "IOB1",
                 contextualizer: Optional[Contextualizer] = None,
                 max_instances: Optional[Union[int, float]] = None,
                 seed: int = 0,
                 lazy: bool = False) -> None:
        super().__init__(
            max_instances=max_instances,
            seed=seed,
            lazy=lazy)
        if label_encoding not in ("IOB1", "BIOUL"):
            raise ConfigurationError("unknown label_encoding: {}".format(
                label_encoding))
        self._label_encoding = label_encoding
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

        with open(file_path, "r") as data_file:
            if count_only:
                logger.info("Counting instances in file at: %s", file_path)
            else:
                logger.info("Reading instances from lines in file at: %s", file_path)

            index = 0
            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    if count_only:
                        yield 1
                        continue
                    if keep_idx is not None and index not in keep_idx:
                        index += 1
                        continue
                    fields = [line.strip().split() for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    tokens, _, _, ner_tags = [list(field) for field in zip(*fields)]

                    # Contextualize the tokens if a Contextualizer was provided.
                    # TODO (nfliu): How can we make this batched?
                    # Would make contextualizers that use the GPU much faster.
                    if self._contextualizer:
                        token_representations = self._contextualizer([tokens])[0]
                    else:
                        token_representations = None

                    # Recode the labels if necessary.
                    if self._label_encoding == "BIOUL":
                        coded_ner = to_bioul(ner_tags) if ner_tags is not None else None
                    else:
                        coded_ner = ner_tags

                    yield self.text_to_instance(
                        tokens,
                        token_representations,
                        coded_ner)
                    index += 1
