from typing import Optional, Set, Union
import logging
from overrides import overrides
import random

from allennlp.data.dataset_readers.dataset_reader import DatasetReader

logger = logging.getLogger(__name__)


class TruncatableDatasetReader(DatasetReader):
    """
    A base DatasetReader with the ability to return only a subset of the generated
    instances.

    Parameters
    ----------
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
                 max_instances: Union[int, float] = None,
                 seed: int = 0,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._max_instances = max_instances
        if max_instances is not None and not isinstance(max_instances, (int, float)):
            # Coerce max_num_instances into an int or a float
            try:
                self._max_instances = int(max_instances)
            except ValueError:
                self._max_instances = float(max_instances)
        if self._max_instances is not None:
            if self._max_instances <= 0:
                raise ValueError("max_instances must be greater than 0. "
                                 "Got value {}".format(self._max_instances))
            if isinstance(self._max_instances, float) and self._max_instances > 1.0:
                raise ValueError(
                    "If float, max_instances cannot be greater than 1. "
                    "Got value {}".format(self._max_instances))
        self._keep_idx: Set[int] = set()
        self._seed = seed
        self._rng = random.Random(seed)

    def _reseed(self, seed: int = 0):
        """
        Reseed the Random instance underlying the dataset reader.

        Parameters
        ----------
        seed: int, optional (default=``1``)
            The random seed to use.
        """
        self._rng = random.Random(seed)

    @overrides
    def _read(self, file_path: str):
        self._reseed(self._seed)
        if self._max_instances is None or self._max_instances == 1.0:
            yield from self._read_dataset(file_path)
        else:
            # We want to truncate the dataset.
            if not self._keep_idx:
                # Figure out which instances to keep and which to discard.
                # Count the total number of instances in the dataset.
                total_num_instances = self._count_instances_in_dataset(file_path)

                # Generate the indices to keep
                dataset_indices = list(range(total_num_instances))
                self._rng.shuffle(dataset_indices)
                if isinstance(self._max_instances, int):
                    num_instances_to_keep = self._max_instances
                else:
                    num_instances_to_keep = int(self._max_instances * total_num_instances)
                if num_instances_to_keep > total_num_instances:
                    logger.warning("Raw number of instances to keep is %s, but total "
                                   "number of instances in dataset is %s. Keeping "
                                   "all instances...", num_instances_to_keep, total_num_instances)
                self._keep_idx.update(dataset_indices[:num_instances_to_keep])
                logger.info("Keeping %s instances", len(self._keep_idx))

            # We know which instances we want to keep, so yield from the reader,
            # taking only those instances.
            yield from self._read_dataset(file_path=file_path, keep_idx=self._keep_idx)

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

    def _count_instances_in_dataset(self, file_path: str):
        num_instances = 0
        for instance in self._read_dataset(file_path=file_path, count_only=True):
            num_instances += 1
        return num_instances
