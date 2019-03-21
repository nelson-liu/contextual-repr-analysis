from typing import Optional, Set
import logging
from overrides import overrides
import re

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.common.file_utils import cached_path

from contexteval.data.dataset_readers import TaggingDatasetReader

logger = logging.getLogger(__name__)


@DatasetReader.register("ccg_supertagging")
class CcgSupertaggingDatasetReader(TaggingDatasetReader):
    """
    Reads a file with concatenated trees from CCGBank format and produces instances
    suitable for use by an auxiliary classifier that aims to predict, given word representations
    for each token in a sequence, the CCG supertag of each.

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
            logger.info("Counting instances in file at: %s", file_path)
        else:
            logger.info("Reading instances from lines in file at: %s", file_path)
        index = 0
        with open(file_path) as input_file:
            for line in input_file:
                if line.startswith("(<"):
                    if count_only:
                        yield 1
                        continue
                    if keep_idx is not None and index not in keep_idx:
                        index += 1
                        continue

                    # Each leaf looks like
                    # (<L ccg_category modified_pos original_pos token predicate_arg_category>)
                    leaves = re.findall("<L (.*?)>", line)

                    # Use magic unzipping trick to split into tuples
                    tuples = zip(*[leaf.split() for leaf in leaves])

                    # Convert to lists and assign to variables.
                    ccg_categories, _, _, tokens, _ = \
                        [list(result) for result in tuples]
                    # Contextualize the tokens if a Contextualizer was provided.
                    # TODO (nfliu): How can we make this batched?
                    # Would make contextualizers that use the GPU much faster.
                    if self._contextualizer:
                        token_representations = self._contextualizer([tokens])[0]
                    else:
                        token_representations = None

                    yield self.text_to_instance(tokens,
                                                token_representations,
                                                ccg_categories)
                    index += 1
